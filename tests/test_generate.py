import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import config
import generate as g


class ImageTests(unittest.TestCase):
    def test_threshold_and_nearest_resize(self):
        mask = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8))
        actual = np.asarray(g.binarize_mask(mask, (8, 2)))
        np.testing.assert_array_equal(actual, [[0, 0, 0, 0, 255, 255, 255, 255]] * 2)

    def test_composite_exact_after_png_round_trip(self):
        original = np.random.default_rng(3).integers(0, 256, (11, 17, 3), dtype=np.uint8)
        generated = 255 - original
        mask = np.zeros((11, 17), dtype=np.uint8)
        mask[2:7, 4:9] = 255
        image = g.composite_patch(Image.fromarray(original), Image.fromarray(generated), Image.fromarray(mask))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        stream.seek(0)
        actual = np.asarray(Image.open(stream))
        np.testing.assert_array_equal(actual[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(actual[mask > 0], generated[mask > 0])

    def test_composite_shape_guard(self):
        with self.assertRaises(ValueError):
            g.composite_patch(Image.new("RGB", (2, 2)), Image.new("RGB", (3, 2)), Image.new("L", (2, 2)))

    def test_overlay_known_pixel(self):
        actual = np.asarray(g.overlay_mask_on_image(Image.new("RGB", (2, 1), (100, 100, 100)),
                          Image.fromarray(np.array([[255, 0]], dtype=np.uint8))))
        np.testing.assert_array_equal(actual, [[[154, 65, 65], [100, 100, 100]]])

    def test_palette_against_independent_distance_search(self):
        pixels = np.random.default_rng(7).integers(0, 256, (7, 11, 3), dtype=np.uint8)
        palette = g.PALETTE.tolist()
        expected = [[min(palette, key=lambda c: sum((int(a)-b)**2 for a,b in zip(pixel,c)))
                     for pixel in row] for row in pixels]
        np.testing.assert_array_equal(g.quantize_to_palette(Image.fromarray(pixels), (11, 7)), expected)

    def test_response_bytes_base64_and_thinking_filter(self):
        stream = io.BytesIO()
        Image.new("RGB", (2, 3), (17, 33, 100)).save(stream, format="PNG")
        for data in (stream.getvalue(), base64.b64encode(stream.getvalue()).decode()):
            response = SimpleNamespace(parts=[SimpleNamespace(text="hello"),
                SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png", data=data)),
                SimpleNamespace(thought=True, inline_data=SimpleNamespace(mime_type="image/png", data=b"bad"))])
            self.assertEqual(g.response_last_image_as_pil(response).getpixel((0, 0)), (17, 33, 100))

    def test_no_image_response(self):
        for parts in (None, [], [SimpleNamespace(text="no image")]):
            with self.assertRaisesRegex(ValueError, "no image"):
                g.response_last_image_as_pil(SimpleNamespace(parts=parts))

    def test_collision_free_names(self):
        root = Path("inputs")
        names = ["a/frame.jpg", "b/frame.jpg", "a/frame.png"]
        self.assertEqual(len({g.output_key(root / n, root) for n in names}), 3)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source = self.inputs / "road.png"
        Image.new("RGB", (8, 6), (100, 100, 100)).save(self.source)
        self.output = self.root / "outputs"
        self.calls = []

    def fake_generate(self, prompt, images):
        self.calls.append(prompt)
        if prompt == config.MASK_PROMPT:
            mask = Image.new("L", (8, 6))
            mask.paste(255, (2, 2, 6, 4))
            return mask
        return Image.new("RGB", (8, 6), (33, 44, 55))

    def run_image(self, **kwargs):
        return g.process_image(self.source, self.inputs, self.output, self.fake_generate, **kwargs)

    def test_two_calls_resume_and_optional_stages(self):
        paths = self.run_image()
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(paths["semseg"].exists())
        self.assertFalse(paths["evening"].exists())
        self.run_image()
        self.assertEqual(len(self.calls), 2)
        self.run_image(evening=True, semseg=True)
        self.assertEqual(len(self.calls), 4)
        self.run_image(evening=True, semseg=True)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(g.load_rgb(paths["patches"]).getpixel((0, 0)), (100, 100, 100))
        self.assertEqual(g.load_rgb(paths["patches"]).getpixel((3, 3)), (33, 44, 55))

    def test_changed_source_refused_before_api(self):
        self.run_image()
        Image.new("RGB", (8, 6), (7, 8, 9)).save(self.source)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_image()
        self.assertEqual(len(self.calls), 2)

    def test_changed_prompt_refused(self):
        self.run_image()
        with patch.object(config, "PATCH_PROMPT", "new prompt"):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_image()
        self.assertEqual(len(self.calls), 2)

    def test_failed_patch_resumes_from_saved_mask(self):
        def fail_patch(prompt, images):
            if prompt == config.PATCH_PROMPT:
                raise ValueError("No image")
            return self.fake_generate(prompt, images)
        with self.assertRaises(ValueError):
            g.process_image(self.source, self.inputs, self.output, fail_patch)
        self.run_image()
        self.assertEqual(self.calls.count(config.MASK_PROMPT), 1)

    def test_empty_mask_stops_before_patch_generation(self):
        calls = []
        def empty(prompt, images):
            calls.append(prompt)
            return Image.new("L", (8, 6))
        with self.assertRaisesRegex(ValueError, "Empty"):
            g.process_image(self.source, self.inputs, self.output, empty)
        self.assertEqual(calls, [config.MASK_PROMPT])

    def test_discovery_excludes_nested_outputs(self):
        nested = self.inputs / "outputs"
        nested.mkdir()
        Image.new("RGB", (2, 2)).save(nested / "generated.png")
        self.assertEqual(g.discover_inputs(self.inputs, nested), [self.source])
        with self.assertRaises(ValueError):
            g.discover_inputs(self.inputs, self.inputs)
        with self.assertRaises(ValueError):
            g.discover_inputs(self.inputs, self.root)


if __name__ == "__main__":
    unittest.main()
