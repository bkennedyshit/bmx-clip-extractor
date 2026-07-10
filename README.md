# BMX Riding Extractor

First-stage horizontal BMX footage extraction using local Qwen vision models
and lossless FFmpeg cuts.

## Scope

Input is one long horizontal video containing riding and non-riding time.
Output is multiple independent horizontal riding clips. The product does not
create reels, crop vertically, sequence a final edit, add music, or publish.

## Tested behavior

Validated on `C1120.MP4`:

- 3840x2160 HEVC source
- Qwen2-VL 7B
- 2-second frame sampling
- 4-second gap merging
- 2.5-second pre-roll
- 1-second post-roll
- Five extracted riding clips

## Configurable models

| Alias | Hugging Face model | Status |
| --- | --- | --- |
| `qwen2-7b` | `Qwen/Qwen2-VL-7B-Instruct` | Tested |
| `qwen25-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | Packaged, needs clip validation |
| `qwen25-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Packaged, needs clip validation |

The loader selects the correct Transformers model class for each generation.
Qwen2.5-VL requires `transformers>=4.51.3`; the setup scripts install that
requirement inside the product virtual environment.

## Setup

Windows:

```bat
setup-windows.bat
```

Linux:

```bash
bash setup-linux.sh
```

FFmpeg must be on `PATH`. Windows also checks `C:\ffmpeg\bin\ffmpeg.exe`.

## Run

```powershell
venv\Scripts\python workspace\bmx_riding_extractor.py `
  D:\AGENT\input\bmx\C1120.MP4 `
  --output-dir D:\AGENT\output\bmx\broll\06-14-26 `
  --model qwen2-7b
```

List available models:

```powershell
venv\Scripts\python workspace\bmx_riding_extractor.py --list-models
```

Every run writes a JSON scan log containing the model, settings, frame
decisions, segment boundaries, and output files.

## Fast line-context recuts

When a scan finds the action but cuts too tight, reuse the existing
`*_scan_log.json` instead of loading Qwen again. Recut mode also accepts the
installed Codex skill's `*_riding_manifest.json` shape.

```powershell
venv\Scripts\python workspace\bmx_riding_extractor.py `
  --recut-log D:\AGENT\output\bmx\C1120_scan_log.json `
  --output-dir D:\AGENT\output\bmx\line-context `
  --line-context `
  --clip-label line
```

`--line-context` uses the proven full-line defaults from the July 2026 BMX
test pass: 8 seconds before detected riding and 4 seconds after. This is meant
for cases where the model catches the halfcab or rollout but the setup trick,
manual, fakie, or earlier 180 happened farther back in the line. Recut mode
does not require CUDA because it only reads the scan log and asks FFmpeg to make
new stream-copy cuts.
