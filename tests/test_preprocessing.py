import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from preprocessing import crop_document, preprocess


class PreprocessingTests(unittest.TestCase):
    def test_blank_image_falls_back_without_changing_pixels(self):
        image = np.full((200, 300, 3), 240, dtype=np.uint8)
        result, detected = crop_document(image)
        self.assertFalse(detected)
        np.testing.assert_array_equal(result, image)

    def test_large_document_restores_original_scale_and_preserves_content(self):
        image = np.zeros((1800, 1500, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (1300, 1600), (255, 255, 255), -1)
        cv2.rectangle(image, (600, 700), (900, 1100), (30, 90, 170), -1)
        before = image.copy()
        result, detected = crop_document(image)
        self.assertTrue(detected)
        self.assertGreater(result.shape[0], 1300)
        self.assertGreater(result.shape[1], 1000)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_allclose(result[result.shape[0] // 2, result.shape[1] // 2],
                                   [30, 90, 170], atol=3)

    def test_binary_input_is_three_channel_and_original_is_unchanged(self):
        image = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        before = image.copy()
        result = preprocess(image, "binary")
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        np.testing.assert_array_equal(image, before)


if __name__ == "__main__":
    unittest.main()
