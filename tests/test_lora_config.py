import tempfile
import unittest
from pathlib import Path

import yaml

from roentgenv2.train_code.lora_config import load_config


class LoRAConfigTests(unittest.TestCase):
    def test_joint_text_encoder_config_has_separate_hyperparameters(self):
        config = load_config("configs/train_lora_roentgen_text_encoder.yaml")

        self.assertTrue(config.train_text_encoder_lora)
        self.assertEqual(config.unet_lora_rank, 16)
        self.assertEqual(config.text_encoder_lora_rank, 4)
        self.assertEqual(config.unet_learning_rate, 3.0e-5)
        self.assertEqual(config.text_encoder_learning_rate, 5.0e-6)
        self.assertTrue(config.output_dir.endswith("/train_07"))

    def test_legacy_unet_keys_still_load(self):
        config = load_config("configs/train_lora_roentgen.yaml")

        self.assertEqual(config.unet_lora_rank, 8)
        self.assertEqual(config.unet_lora_alpha, 8)
        self.assertEqual(config.unet_learning_rate, 1.0e-4)
        self.assertFalse(config.train_text_encoder_lora)

    def test_mixed_legacy_and_new_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "learning_rate": 1.0e-4,
                        "unet_learning_rate": 3.0e-5,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "use only 'unet_learning_rate'"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
