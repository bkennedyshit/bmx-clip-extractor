---
name: extract-bmx-riding-clips
description: Scan long horizontal BMX source videos with a local Qwen vision model and extract every active riding interval as a separate lossless horizontal clip. Use for the first stage of a YouTube BMX workflow when standing, talking, setup, and empty time must be removed before later editing. Do not use for reels, vertical crops, sequencing, music, or publishing.
---

# Extract BMX Riding Clips

Run `bmx_riding_extractor.py` to perform one job:

```text
long horizontal source -> Qwen riding decisions -> multiple horizontal clips
```

The script samples frames across the full timeline, asks Qwen whether each
sample contains active riding, merges nearby positive samples, adds approach
and rollout padding, and cuts each interval with FFmpeg stream copy.

If the Qwen scan catches the action but cuts too tight, use `--recut-log` with
`--line-context` to reuse the existing scan log and make wider full-line cuts
without loading the model again.

## Defaults

- Model: `qwen2-7b` (tested on `C1120.MP4`)
- Frame sampling: every 2 seconds
- Merge gap: 4 seconds
- Pre-roll: 2.5 seconds
- Post-roll: 1 second
- Line-context recut: 8 seconds pre-roll, 4 seconds post-roll
- Output: original horizontal resolution and codecs

## Models

```powershell
python workspace\bmx_riding_extractor.py --list-models
```

- `qwen2-7b`: Qwen2-VL 7B, tested
- `qwen25-3b`: Qwen2.5-VL 3B, configurable but not yet clip-validated
- `qwen25-7b`: Qwen2.5-VL 7B, configurable but not yet clip-validated

Use the product virtual environment. Qwen2.5 aliases require
`transformers>=4.51.3`, installed by the setup scripts.

## Run

```powershell
python workspace\bmx_riding_extractor.py INPUT.MP4 `
  --output-dir OUTPUT_FOLDER `
  --model qwen2-7b
```

## Recut Existing Scan

```powershell
python workspace\bmx_riding_extractor.py `
  --recut-log OUTPUT_FOLDER\INPUT_scan_log.json `
  --output-dir OUTPUT_FOLDER\line-context `
  --line-context `
  --clip-label line
```

Use this when the first pass catches only the end of a trick line, such as a
halfcab or rollout, and needs more time before the detected frame to include
the earlier 180, fakie, manual, or setup move.

Never crop, rotate, reframe, concatenate, or convert the output to vertical.
Downstream reel and composition work belongs to separate skills.
