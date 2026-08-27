import json
import tempfile
import unittest
from pathlib import Path

from roentgenv2.train_code.metrics_logger import MetricsLogger


class MetricsLoggerTests(unittest.TestCase):
    def test_resume_preserves_history_and_trims_points_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            initial = MetricsLogger(metrics_path, meta={"max_train_steps": 300})
            for step in (100, 200, 300):
                initial.log_train(step, step / 1000, 1.0e-4)
                initial.log_val(step, step / 2000)
            initial.save()

            resumed = MetricsLogger(
                metrics_path,
                meta={"max_train_steps": 400, "train_text_encoder_lora": False},
            )
            self.assertTrue(resumed.load_for_resume(200))
            resumed.log_train(300, 0.25, 9.0e-5)
            resumed.log_val(300, 0.12)
            resumed.save()

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["train"]["step"], [100, 200, 300])
            self.assertEqual(payload["train"]["loss"], [0.1, 0.2, 0.25])
            self.assertEqual(payload["val"]["step"], [100, 200, 300])
            self.assertEqual(payload["val"]["loss"], [0.05, 0.1, 0.12])
            self.assertEqual(payload["meta"]["max_train_steps"], 400)
            self.assertFalse(metrics_path.with_name(".metrics.json.tmp").exists())

    def test_logging_the_same_step_updates_instead_of_duplicating(self):
        logger = MetricsLogger("unused.json")
        logger.log_train(10, 0.5, 1.0e-4, 5.0e-6)
        logger.log_train(10, 0.4, 9.0e-5, 4.0e-6)
        logger.log_val(10, 0.3)
        logger.log_val(10, 0.2)

        self.assertEqual(logger.data["train"]["step"], [10])
        self.assertEqual(logger.data["train"]["loss"], [0.4])
        self.assertEqual(logger.data["train"]["text_encoder_lr"], [4.0e-6])
        self.assertEqual(logger.data["val"]["step"], [10])
        self.assertEqual(logger.data["val"]["loss"], [0.2])

    def test_resume_rejects_misaligned_history_instead_of_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            original = {
                "meta": {},
                "train": {"step": [1, 2], "loss": [0.5], "lr": [1.0e-4, 1.0e-4]},
                "val": {"step": [], "loss": []},
            }
            metrics_path.write_text(json.dumps(original), encoding="utf-8")

            logger = MetricsLogger(metrics_path)
            with self.assertRaisesRegex(ValueError, "train.loss has 1 values for 2 steps"):
                logger.load_for_resume(2)

            self.assertEqual(json.loads(metrics_path.read_text(encoding="utf-8")), original)

    def test_missing_metrics_file_starts_history_at_resumed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            logger = MetricsLogger(metrics_path)

            self.assertFalse(logger.load_for_resume(100))
            logger.log_train(101, 0.4, 1.0e-4)

            self.assertEqual(logger.data["train"]["step"], [101])


if __name__ == "__main__":
    unittest.main()
