import importlib.util
import unittest

import numpy as np

from roentgenv2.evaluation.medclip_alignment import (
    binary_roc_auc,
    derangement,
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

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is only required in the MedCLIP runtime")
    def test_medclip_preprocessing_returns_single_channel_224_square(self):
        from PIL import Image

        image = Image.new("L", (100, 300), 127)
        processed = preprocess_medclip_image(image)
        self.assertEqual(processed.shape, (1, 224, 224))
        self.assertTrue(np.isfinite(processed).all())


if __name__ == "__main__":
    unittest.main()
