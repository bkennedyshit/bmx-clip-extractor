import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "workspace"))

from bmx_riding_extractor import (
    DEFAULT_LINE_CONTEXT_PAD_END,
    DEFAULT_LINE_CONTEXT_PAD_START,
    build_segments,
    load_scan_log,
    parse_args,
)


class BuildSegmentsTest(unittest.TestCase):
    def test_tested_c1120_boundaries(self):
        riding_times = [30, 36, 48, 50, 58, 60, 62, 72, 74, 78, 80, 82, 84, 86, 102, 104]
        scan = [{"time": value, "riding": True} for value in riding_times]
        segments = build_segments(scan, sample_interval=2.0, duration=109.109)
        self.assertEqual(
            segments,
            [
                {"start": 27.5, "end": 39.0, "duration": 11.5},
                {"start": 45.5, "end": 53.0, "duration": 7.5},
                {"start": 55.5, "end": 65.0, "duration": 9.5},
                {"start": 69.5, "end": 89.0, "duration": 19.5},
                {"start": 99.5, "end": 107.0, "duration": 7.5},
            ],
        )

    def test_line_context_recut_defaults(self):
        args = parse_args(
            [
                "--recut-log",
                "scan_log.json",
                "--output-dir",
                "out",
                "--line-context",
            ]
        )
        self.assertEqual(args.pad_start, DEFAULT_LINE_CONTEXT_PAD_START)
        self.assertEqual(args.pad_end, DEFAULT_LINE_CONTEXT_PAD_END)

    def test_line_context_keeps_explicit_padding(self):
        args = parse_args(
            [
                "--recut-log",
                "scan_log.json",
                "--output-dir",
                "out",
                "--line-context",
                "--pad-start",
                "12",
                "--pad-end",
                "6",
            ]
        )
        self.assertEqual(args.pad_start, 12)
        self.assertEqual(args.pad_end, 6)

    def test_load_scan_log_accepts_installed_skill_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "source": r"D:\input.mov",
                        "source_duration": 10.0,
                        "source_fps": 30.0,
                        "sample_interval": 1.0,
                        "gap_tolerance": 2.5,
                        "scan": [{"time": 4.0, "riding": True}],
                    }
                ),
                encoding="utf-8",
            )
            data = load_scan_log(path)
        self.assertEqual(data["video"], r"D:\input.mov")
        self.assertEqual(data["source"]["duration"], 10.0)
        self.assertEqual(data["settings"]["sample_interval"], 1.0)


if __name__ == "__main__":
    unittest.main()
