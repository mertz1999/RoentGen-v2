import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roentgenv2.evaluation.medclip_alignment import (
    binary_roc_auc,
    derangement,
    load_prompt_dir_pairs,
    preprocess_medclip_image,
    stem_from_value,
)


class MedclipAlignmentTests(unittest.TestCase):
    def test_numeric_csv_id_maps_to_image_stem(self):
        self.assertEqual(stem_from_value(1002), "1002")
        self.assertEqual(stem_from_value(1002.0), "1002")
        self.assertEqual(stem_from_value("CXR1002"), "CXR1002")

    def test_negative_mapping_has_no_matched_pairs(self):
        negative = derangement(100, np.random.default_rng(873))
        self.assertFalse(np.any(negative == np.arange(100)))

    def test_auc_is_one_when_every_matched_score_is_higher(self):
        self.assertEqual(binary_roc_auc(np.array([0.9, 0.8]), np.array([0.1, 0.2, 0.3])), 1.0)

    def test_prompt_directory_matches_exact_and_inference_image_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "labels"
            images = root / "predicted"
            prompts.mkdir()
            images.mkdir()
            (prompts / "1001.txt").write_text("No acute disease.", encoding="utf-8")
            (prompts / "1002.txt").write_text("Mild cardiomegaly.", encoding="utf-8")
            (images / "1001.jpg").touch()
            (images / "1002_0.png").touch()

            frame = load_prompt_dir_pairs(prompts, images)

            self.assertEqual(frame["folder_stem"].tolist(), ["1001", "1002"])
            self.assertEqual(frame["prompt"].tolist(), ["No acute disease.", "Mild cardiomegaly."])
            self.assertEqual(
                [path.name for path in frame["image_path"]],
                ["1001.jpg", "1002_0.png"],
            )

    def test_prompt_directory_rejects_missing_generated_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "labels"
            images = root / "predicted"
            prompts.mkdir()
            images.mkdir()
            (prompts / "1001.txt").write_text("No acute disease.", encoding="utf-8")
            (images / "different.jpg").touch()

            with self.assertRaisesRegex(ValueError, r"prompt files: \['1001'\]"):
                load_prompt_dir_pairs(prompts, images)

    def test_prompt_directory_rejects_ambiguous_generated_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = root / "labels"
            images = root / "predicted"
            prompts.mkdir()
            images.mkdir()
            (prompts / "1001.txt").write_text("No acute disease.", encoding="utf-8")
            (images / "1001.jpg").touch()
            (images / "1001_0.jpg").touch()

            with self.assertRaisesRegex(ValueError, "Multiple generated images match"):
                load_prompt_dir_pairs(prompts, images)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is only required in the MedCLIP runtime")
    def test_medclip_preprocessing_returns_single_channel_224_square(self):
        from PIL import Image

        image = Image.new("L", (100, 300), 127)
        processed = preprocess_medclip_image(image)
        self.assertEqual(processed.shape, (1, 224, 224))
        self.assertTrue(np.isfinite(processed).all())


if __name__ == "__main__":
    unittest.main()
