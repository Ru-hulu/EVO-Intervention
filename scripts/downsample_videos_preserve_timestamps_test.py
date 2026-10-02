"""End-to-end checks for exact frame selection and non-divisible clip endings."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("downsample_videos_preserve_timestamps.py")
SPEC = importlib.util.spec_from_file_location("timestamp_downsampler", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class DownsampleIntegrationTest(unittest.TestCase):
    def test_frame_identity_timestamps_and_exact_end_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame_count in (1, 6, 7, 17, 18, 19):
                with self.subTest(frame_count=frame_count):
                    source, output = root / "source.mp4", root / "output.mp4"
                    subprocess.run([
                        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30",
                        "-frames:v", str(frame_count), "-c:v", "libx264", "-crf", "0", str(source),
                    ], check=True)
                    before = MODULE.probe_video(source)
                    MODULE.validate_source(before, 30, source)
                    MODULE.convert_video(source, output, before, 6, crf=0, overwrite=True)
                    after = MODULE.probe_video(output)
                    self.assertEqual(after.pts, before.pts[::6])
                    self.assertEqual(after.duration, before.duration)
                    self.assertEqual(len(after.pts), (frame_count + 5) // 6)

                    def decoded_hashes(path, select=False):
                        command = ["ffmpeg", "-v", "error", "-i", str(path)]
                        if select:
                            command += ["-vf", "select=not(mod(n\\,6))"]
                        command += ["-fps_mode", "passthrough", "-f", "framemd5", "-"]
                        text = subprocess.check_output(command, text=True)
                        return [line.rsplit(",", 1)[1].strip() for line in text.splitlines() if not line.startswith("#")]

                    # Lossless encoding lets us prove that frame 0, 6, 12, ...
                    # was selected, rather than the nearby frames of an fps filter.
                    self.assertEqual(decoded_hashes(source, select=True), decoded_hashes(output))

    def test_rejects_in_place_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--input-dir", directory, "--output-dir", directory,
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Input and output directories must differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
