#!/usr/bin/env python3
"""Extract active BMX riding intervals as lossless horizontal clips."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_REGISTRY = {
    "qwen2-7b": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "class_name": "Qwen2VLForConditionalGeneration",
    },
    "qwen25-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
    },
    "qwen25-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
    },
}

PROMPT = (
    "Look at this video frame. Is this active BMX riding? "
    "Active BMX riding means a rider on a BMX bike approaching a feature, "
    "pedaling with purpose, pumping, jumping, grinding, doing a trick, "
    "landing, rolling away, or bailing during a riding attempt. "
    "NOT active riding means standing still, holding or walking the bike, "
    "talking, resting, setting up, filming, an empty scene, or no active "
    "rider visible. Reply with ONLY one word: YES or NO."
)

DEFAULT_SAMPLE_INTERVAL = 2.0
DEFAULT_GAP_TOLERANCE = 4.0
DEFAULT_MIN_CLIP = 3.0
DEFAULT_PAD_START = 2.5
DEFAULT_PAD_END = 1.0
DEFAULT_LINE_CONTEXT_PAD_START = 8.0
DEFAULT_LINE_CONTEXT_PAD_END = 4.0
DEFAULT_INFERENCE_WIDTH = 1280
DEFAULT_INFERENCE_HEIGHT = 720


def resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    windows_default = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
    if windows_default.is_file():
        return str(windows_default)
    raise RuntimeError("ffmpeg was not found on PATH or at C:\\ffmpeg\\bin\\ffmpeg.exe")


def get_model_class(class_name: str):
    import transformers

    try:
        return getattr(transformers, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Installed transformers {transformers.__version__} does not expose "
            f"{class_name}. Upgrade transformers."
        ) from exc


def load_model(model_key: str):
    config = MODEL_REGISTRY[model_key]
    model_id = config["model_id"]
    model_class = get_model_class(config["class_name"])
    print(f"[model] Loading {model_id} on cuda:0...", flush=True)
    started = time.time()
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=720 * 28 * 28,
    )
    allocated = torch.cuda.memory_allocated() / 1e9
    print(
        f"[model] Ready in {time.time() - started:.1f}s | "
        f"VRAM allocated: {allocated:.1f} GB",
        flush=True,
    )
    return model, processor, model_id


def classify_frame(image: Image.Image, model, processor) -> tuple[bool, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    input_length = inputs["input_ids"].shape[1]
    answer = processor.decode(
        output_ids[0][input_length:], skip_special_tokens=True
    ).strip().upper()
    return answer.startswith("YES"), answer


def probe_video(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    if width <= height:
        raise RuntimeError(
            f"Horizontal source required; received {width}x{height}. "
            "This product never rotates, crops, or converts vertical footage."
        )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
        "width": width,
        "height": height,
    }


def scan_video(
    video_path: Path,
    model,
    processor,
    sample_interval: float,
    start_offset: float,
) -> tuple[list[dict], dict]:
    metadata = probe_video(video_path)
    capture = cv2.VideoCapture(str(video_path))
    results = []
    timestamp = start_offset
    started = time.time()
    count = 0
    expected = max(0, int((metadata["duration"] - start_offset) / sample_interval))
    print(
        f"[scan] {video_path.name} -- {metadata['duration']:.1f}s @ "
        f"{metadata['fps']:.3f}fps, source {metadata['width']}x{metadata['height']}, "
        f"inference {DEFAULT_INFERENCE_WIDTH}x{DEFAULT_INFERENCE_HEIGHT}",
        flush=True,
    )
    print(f"[scan] About {expected} frames to classify", flush=True)
    try:
        while timestamp < metadata["duration"]:
            frame_index = int(timestamp * metadata["fps"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break
            frame_small = cv2.resize(
                frame,
                (DEFAULT_INFERENCE_WIDTH, DEFAULT_INFERENCE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            image = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
            riding, answer = classify_frame(image, model, processor)
            results.append(
                {"time": round(timestamp, 3), "riding": riding, "answer": answer}
            )
            count += 1
            rate = count / max(time.time() - started, 0.001)
            status = "RIDING" if riding else "skip"
            print(
                f"  t={timestamp:7.2f}s  {status:<6}  [{rate:.2f} frames/s]",
                flush=True,
            )
            timestamp += sample_interval
    finally:
        capture.release()
    return results, metadata


def build_segments(
    scan_results: list[dict],
    sample_interval: float,
    duration: float,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    min_clip: float = DEFAULT_MIN_CLIP,
    pad_start: float = DEFAULT_PAD_START,
    pad_end: float = DEFAULT_PAD_END,
) -> list[dict]:
    riding_times = [row["time"] for row in scan_results if row["riding"]]
    if not riding_times:
        return []

    raw_segments = []
    segment_start = riding_times[0]
    segment_end = riding_times[0] + sample_interval
    for timestamp in riding_times[1:]:
        if timestamp - segment_end <= gap_tolerance:
            segment_end = timestamp + sample_interval
        else:
            raw_segments.append((segment_start, segment_end))
            segment_start = timestamp
            segment_end = timestamp + sample_interval
    raw_segments.append((segment_start, segment_end))

    segments = []
    for start, end in raw_segments:
        padded_start = max(0.0, start - pad_start)
        padded_end = min(duration, end + pad_end)
        if padded_end - padded_start >= min_clip:
            segments.append(
                {
                    "start": round(padded_start, 3),
                    "end": round(padded_end, 3),
                    "duration": round(padded_end - padded_start, 3),
                }
            )
    return segments


def cut_segments(
    ffmpeg: str,
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    label: str = "riding",
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, segment in enumerate(segments, 1):
        output_path = output_dir / f"{video_path.stem}_{label}_{index:02d}.MP4"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{segment['start']:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{segment['duration']:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        print(
            f"[cut] {index}/{len(segments)}: {segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s -> {output_path.name}",
            flush=True,
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {output_path.name}: {completed.stderr[-1000:]}"
            )
        outputs.append(str(output_path))
    return outputs


def load_scan_log(scan_log: Path) -> dict:
    with scan_log.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "scan" not in data:
        raise RuntimeError(f"{scan_log} is missing required key: scan")
    if not isinstance(data["scan"], list):
        raise RuntimeError(f"{scan_log} has an invalid scan array")
    if "video" in data and isinstance(data.get("source"), dict):
        return data
    if "source" in data and "source_duration" in data:
        return {
            "video": data["source"],
            "source": {
                "duration": data["source_duration"],
                "fps": data.get("source_fps"),
            },
            "settings": {
                "sample_interval": data.get(
                    "sample_interval", DEFAULT_SAMPLE_INTERVAL
                ),
                "gap_tolerance": data.get(
                    "gap_tolerance", DEFAULT_GAP_TOLERANCE
                ),
            },
            "scan": data["scan"],
        }
    required = ("video", "source", "settings", "scan")
    missing = [key for key in required if key not in data]
    raise RuntimeError(f"{scan_log} is missing required keys: {', '.join(missing)}")


def recut_from_scan_log(
    scan_log: Path,
    output_dir: Path,
    ffmpeg: str,
    gap_tolerance: float | None = None,
    min_clip: float | None = None,
    pad_start: float | None = None,
    pad_end: float | None = None,
    label: str = "line",
) -> int:
    data = load_scan_log(scan_log)
    video_path = Path(data["video"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    settings = data.get("settings", {})
    source = data.get("source", {})
    sample_interval = float(settings.get("sample_interval", DEFAULT_SAMPLE_INTERVAL))
    duration = float(source.get("duration", 0.0))
    if duration <= 0:
        metadata = probe_video(video_path)
        duration = float(metadata["duration"])
        source = metadata

    segments = build_segments(
        data["scan"],
        sample_interval,
        duration,
        gap_tolerance=(
            DEFAULT_GAP_TOLERANCE if gap_tolerance is None else gap_tolerance
        ),
        min_clip=DEFAULT_MIN_CLIP if min_clip is None else min_clip,
        pad_start=DEFAULT_LINE_CONTEXT_PAD_START if pad_start is None else pad_start,
        pad_end=DEFAULT_LINE_CONTEXT_PAD_END if pad_end is None else pad_end,
    )

    print(f"[segments] Found {len(segments)} {label} intervals", flush=True)
    for index, segment in enumerate(segments, 1):
        print(
            f"  {index:2d}. {segment['start']:7.2f}s -> "
            f"{segment['end']:7.2f}s ({segment['duration']:.1f}s)",
            flush=True,
        )

    outputs = cut_segments(ffmpeg, video_path, segments, output_dir, label=label)
    recut_log = output_dir / f"{video_path.stem}_{label}_recut_log.json"
    recut_log.write_text(
        json.dumps(
            {
                "source_log": str(scan_log),
                "video": str(video_path),
                "source": source,
                "settings": {
                    "sample_interval": sample_interval,
                    "gap_tolerance": (
                        DEFAULT_GAP_TOLERANCE
                        if gap_tolerance is None
                        else gap_tolerance
                    ),
                    "min_clip": DEFAULT_MIN_CLIP if min_clip is None else min_clip,
                    "pad_start": (
                        DEFAULT_LINE_CONTEXT_PAD_START
                        if pad_start is None
                        else pad_start
                    ),
                    "pad_end": DEFAULT_LINE_CONTEXT_PAD_END if pad_end is None else pad_end,
                    "label": label,
                },
                "segments": segments,
                "outputs": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[log] {recut_log}", flush=True)
    print(f"[done] {len(outputs)} clips written", flush=True)
    return len(outputs)


def unload_model(model, processor) -> None:
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def list_models() -> None:
    for key, config in MODEL_REGISTRY.items():
        print(f"{key:<11} {config['model_id']}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract multiple lossless horizontal BMX riding clips with a "
            "local Qwen vision model"
        )
    )
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--output-dir", "-o", type=Path)
    parser.add_argument(
        "--recut-log",
        type=Path,
        help=(
            "Reuse an existing *_scan_log.json and cut wider line-context clips "
            "without loading Qwen again"
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="qwen2-7b",
        help="Vision model alias (default: qwen2-7b, the tested configuration)",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--gap-tolerance", type=float, default=DEFAULT_GAP_TOLERANCE)
    parser.add_argument("--min-clip", type=float, default=DEFAULT_MIN_CLIP)
    parser.add_argument("--pad-start", type=float, default=DEFAULT_PAD_START)
    parser.add_argument("--pad-end", type=float, default=DEFAULT_PAD_END)
    parser.add_argument(
        "--line-context",
        action="store_true",
        help="Use wider defaults for full trick lines: 8s pre-roll and 4s rollout",
    )
    parser.add_argument(
        "--clip-label",
        default=None,
        help="Output filename label, for example riding or line",
    )
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args(argv)
    if args.list_models:
        return args
    if args.recut_log is None and args.video is None:
        parser.error("video is required unless --list-models is used")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --list-models is used")
    if args.line_context:
        if args.pad_start == DEFAULT_PAD_START:
            args.pad_start = DEFAULT_LINE_CONTEXT_PAD_START
        if args.pad_end == DEFAULT_PAD_END:
            args.pad_end = DEFAULT_LINE_CONTEXT_PAD_END
    for name in (
        "sample_interval",
        "gap_tolerance",
        "min_clip",
        "pad_start",
        "pad_end",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.sample_interval == 0:
        parser.error("--sample-interval must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        list_models()
        return 0
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    if args.recut_log is not None:
        if not args.recut_log.is_file():
            raise FileNotFoundError(args.recut_log)
        label = args.clip_label or ("line" if args.line_context else "riding")
        recut_from_scan_log(
            args.recut_log,
            args.output_dir,
            ffmpeg,
            gap_tolerance=args.gap_tolerance,
            min_clip=args.min_clip,
            pad_start=args.pad_start,
            pad_end=args.pad_end,
            label=label,
        )
        return 0

    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this product")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = processor = None
    try:
        model, processor, model_id = load_model(args.model)
        scan_results, metadata = scan_video(
            args.video,
            model,
            processor,
            args.sample_interval,
            args.start_offset,
        )
        segments = build_segments(
            scan_results,
            args.sample_interval,
            metadata["duration"],
            gap_tolerance=args.gap_tolerance,
            min_clip=args.min_clip,
            pad_start=args.pad_start,
            pad_end=args.pad_end,
        )
        print(f"[segments] Found {len(segments)} riding segments", flush=True)
        for index, segment in enumerate(segments, 1):
            print(
                f"  {index:2d}. {segment['start']:7.2f}s -> "
                f"{segment['end']:7.2f}s ({segment['duration']:.1f}s)",
                flush=True,
            )
        outputs = []
        if not args.scan_only and segments:
            outputs = cut_segments(
                ffmpeg,
                args.video,
                segments,
                args.output_dir,
                label=args.clip_label or "riding",
            )
        manifest_path = args.output_dir / f"{args.video.stem}_scan_log.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "video": str(args.video),
                    "model_alias": args.model,
                    "model_id": model_id,
                    "source": metadata,
                    "settings": {
                        "sample_interval": args.sample_interval,
                        "gap_tolerance": args.gap_tolerance,
                        "min_clip": args.min_clip,
                        "pad_start": args.pad_start,
                        "pad_end": args.pad_end,
                    },
                    "segments": segments,
                    "outputs": outputs,
                    "scan": scan_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[log] {manifest_path}", flush=True)
        print(f"[done] {len(outputs)} clips written", flush=True)
        return 0
    finally:
        if model is not None and processor is not None:
            unload_model(model, processor)


if __name__ == "__main__":
    raise SystemExit(main())
