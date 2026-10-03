"""Transfer regression checks; no device or inference required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from echoguard.evaluation.device_eval import upload_dataset


class DatasetTransferTests(unittest.TestCase):
    def test_large_unicode_dataset_is_verified_byte_for_byte(self):
        payload = ('[{"text":"नमस्ते"}]\n' * 10000).encode("utf-8")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.json"
            path.write_bytes(payload)
            with patch("echoguard.evaluation.device_eval.subprocess.run") as run, patch(
                "echoguard.evaluation.device_eval.subprocess.check_output", return_value=payload
            ):
                self.assertEqual(upload_dataset(["adb", "-s", "test"], path), payload)
            self.assertEqual(run.call_args_list[0].args[0][3], "push")
            self.assertEqual(run.call_args_list[-1].args[0][3:6], ["shell", "rm", "-f"])
            self.assertFalse(any("exec-in" in call.args[0] for call in run.call_args_list))

    def test_truncated_transfer_fails_and_cleans_staging_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.json"
            path.write_bytes(b'["complete"]')
            with patch("echoguard.evaluation.device_eval.subprocess.run") as run, patch(
                "echoguard.evaluation.device_eval.subprocess.check_output", return_value=b'["comp'
            ):
                with self.assertRaisesRegex(RuntimeError, "Evaluation not started"):
                    upload_dataset(["adb"], path)
            self.assertEqual(run.call_args_list[-1].args[0][1:4], ["shell", "rm", "-f"])


if __name__ == "__main__":
    unittest.main()
