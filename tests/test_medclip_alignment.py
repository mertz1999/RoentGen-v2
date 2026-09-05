import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roentgenv2.evaluation.medclip_alignment import (
    all_other_label_diagnostics,
    load_prompt_dir_pairs,
    paired_similarities,
    preprocess_medclip_image,
    stem_from_value,
)


class MedclipAlignmentTests(unittest.TestCase):
    def test_numeric_csv_id_maps_to_image_stem(self):
        self.assertEqual(stem_from_value(1002), "1002")
        self.assertEqual(stem_from_value(1002.0), "1002")
        self.assertEqual(stem_from_value("CXR1002"), "CXR1002")

    def test_paired_similarities_return_raw_cosine_and_scaled_logits(self):
        image_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        text_embeddings = np.array([[0.6, 0.8], [0.8, 0.6]])

        raw_cosine, scaled_logits = paired_similarities(
            image_embeddings,
            text_embeddings,
            logit_scale=10.0,
        )

        np.testing.assert_allclose(raw_cosine, [0.6, 0.6])
        np.testing.assert_allclose(scaled_logits, [6.0, 6.0])

    def test_paired_similarities_reject_mismatched_embedding_shapes(self):
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            paired_similarities(np.zeros((2, 3)), np.zeros((1, 3)), 10.0)

    def test_all_other_label_diagnostics_use_every_unpaired_label(self):
        image_embeddings = np.eye(3)
        text_embeddings = np.array(
            [
                [0.8, 0.1, 0.2],
                [0.3, 0.7, 0.4],
                [0.5, 0.6, 0.9],
            ]
        )

        diagnostics = all_other_label_diagnostics(
            image_embeddings,
            text_embeddings,
            logit_scale=10.0,
        )

        np.testing.assert_allclose(diagnostics["raw_cosine_similarity"], [0.8, 0.7, 0.9])
        np.testing.assert_allclose(diagnostics["mean_unpaired_cosine_similarity"], [0.4, 0.35, 0.3])
        np.testing.assert_allclose(diagnostics["mean_unpaired_cosine_distance"], [0.6, 0.65, 0.7])
        np.testing.assert_allclose(
            diagnostics["matched_minus_unpaired_cosine_margin"],
            [0.4, 0.35, 0.6],
        )
        np.testing.assert_allclose(
            diagnostics["mean_unpaired_scaled_medclip_logit"],
            [4.0, 3.5, 3.0],
        )
        np.testing.assert_allclose(
            diagnostics["matched_minus_unpaired_scaled_logit_margin"],
            [4.0, 3.5, 6.0],
        )
        np.testing.assert_allclose(diagnostics["correct_label_percentile"], [100.0] * 3)
        np.testing.assert_allclose(diagnostics["correct_label_rank"], [1.0] * 3)

    def test_all_other_label_diagnostics_use_midranks_for_tied_labels(self):
        image_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        text_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

        diagnostics = all_other_label_diagnostics(
            image_embeddings,
            text_embeddings,
            logit_scale=10.0,
        )

        np.testing.assert_allclose(diagnostics["matched_minus_unpaired_cosine_margin"], [0.0, 0.0])
        np.testing.assert_allclose(diagnostics["correct_label_percentile"], [50.0, 50.0])
        np.testing.assert_allclose(diagnostics["correct_label_rank"], [1.5, 1.5])

    def test_all_other_label_diagnostics_require_two_pairs(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            all_other_label_diagnostics(
                np.array([[1.0, 0.0]]),
                np.array([[1.0, 0.0]]),
                logit_scale=10.0,
            )

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
