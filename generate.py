"""Generate road-patch candidates; inspect every pair before using it as a label."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PALETTE = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0],
                    [255, 0, 0], [255, 255, 255]], dtype=np.int32)


def response_last_image_as_pil(response):
    """Ignore text/thinking parts and decode the last actual image."""
    result = None
    for part in getattr(response, "parts", None) or []:
        if getattr(part, "thought", False):
            continue
        blob = getattr(part, "inline_data", None)
        if blob is None or not (getattr(blob, "mime_type", "") or "").startswith("image/"):
            continue
        data = blob.data
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            result = image.convert("RGB")
    if result is None:
        raise ValueError("The API returned no image. No output was saved for this stage.")
    return result


def binarize_mask(mask_img, size_wh):
    mask = mask_img.convert("L").resize(size_wh, Image.Resampling.NEAREST)
    return Image.fromarray((np.asarray(mask) >= 128).astype(np.uint8) * 255)


def ensure_size(image, size_wh):
    return image if image.size == size_wh else image.resize(size_wh, Image.Resampling.LANCZOS)


def composite_patch(original, generated, mask):
    """Preserve decoded RGB pixels outside the binary mask exactly."""
    if original.size != generated.size or original.size != mask.size:
        raise ValueError("Original, generated image and mask must have equal sizes.")
    return Image.composite(generated.convert("RGB"), original.convert("RGB"),
                           binarize_mask(mask, original.size))


def overlay_mask_on_image(image, mask_l, color=(255, 0, 0), alpha=0.35):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if image.size != mask_l.size:
        raise ValueError("Image and mask must have equal sizes.")
    out = np.array(image.convert("RGB"), dtype=np.float32)
    mask = np.asarray(mask_l.convert("L")) >= 128
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def quantize_to_palette(mask_rgb, size_wh):
    arr = np.asarray(mask_rgb.convert("RGB").resize(size_wh, Image.Resampling.NEAREST), dtype=np.int32)
    # Work a row at a time to avoid an H*W*5*3 temporary on large images.
    out = np.empty(arr.shape, dtype=np.uint8)
    for y, row in enumerate(arr):
        distances = np.sum((row[:, None, :] - PALETTE[None, :, :]) ** 2, axis=-1)
        out[y] = PALETTE[np.argmin(distances, axis=1)]
    return Image.fromarray(out)


def output_key(path, input_dir):
    # Retain both the relative folder and original extension: a.jpg != a.png.
    relative = path.relative_to(input_dir)
    return relative.parent / (relative.name + ".png")


def discover_inputs(input_dir, output_dir):
    root, out = input_dir.resolve(), output_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    if root == out or root.is_relative_to(out):
        raise ValueError("The output directory must not contain the input directory.")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
                   and not p.resolve().is_relative_to(out))
    if not paths:
        raise ValueError(f"No images found in {root}")
    return paths


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def process_image(path, input_dir, output_dir, generate_image, *, evening=False, semseg=False):
    """Resume completed stages; refuse stale outputs after an input/prompt change."""
    original = load_rgb(path)
    key = output_key(path, input_dir)
    folders = {name: output_dir / name / key for name in
               ("masks", "generated", "patches", "overlay", "evening", "semseg")}
    manifest_path = output_dir / "metadata" / key.with_suffix(".json")
    identity = {"source": key.as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": config.MODEL, "image_size": "1K", "processing_version": 1,
        "mask_prompt": config.MASK_PROMPT, "patch_prompt": config.PATCH_PROMPT,
        "evening_prompt": config.EVENING_PROMPT, "semseg_prompt": config.SEMSEG_PROMPT}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Input or settings changed for {path.name}. Use a new --output directory.")
    elif any(p.exists() for p in folders.values()):
        raise ValueError(f"Outputs without provenance for {path.name}. Use a new --output directory.")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")
    if not folders["masks"].exists():
        mask = binarize_mask(generate_image(config.MASK_PROMPT, [original]), original.size)
        if not np.any(np.asarray(mask)):
            raise ValueError(f"Empty patch mask for {path.name}; inspect the input before retrying.")
        save_image(mask, folders["masks"])
    with Image.open(folders["masks"]) as image:
        mask = binarize_mask(image, original.size)
    if not folders["generated"].exists():
        generated = ensure_size(generate_image(config.PATCH_PROMPT, [original, mask.convert("RGB")]), original.size)
        save_image(generated, folders["generated"])
    if not folders["patches"].exists():
        save_image(composite_patch(original, load_rgb(folders["generated"]), mask), folders["patches"])
    patched = load_rgb(folders["patches"])
    if not folders["overlay"].exists():
        save_image(overlay_mask_on_image(patched, mask), folders["overlay"])
    if evening and not folders["evening"].exists():
        save_image(ensure_size(generate_image(config.EVENING_PROMPT, [patched]), original.size), folders["evening"])
    if semseg and not folders["semseg"].exists():
        save_image(quantize_to_palette(generate_image(config.SEMSEG_PROMPT, [original]), original.size), folders["semseg"])
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(config.INPUT_DIR))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--evening", action="store_true", help="One additional generation per input")
    parser.add_argument("--semseg", action="store_true", help="One additional experimental segmentation per input")
    parser.add_argument("--limit", type=int, help="Process at most this many input images")
    parser.add_argument("--dry-run", action="store_true", help="List inputs and maximum generation count; no API calls")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.input, args.output = args.input.resolve(), args.output.resolve()
    try:
        paths = discover_inputs(args.input, args.output)
    except ValueError as error:
        parser.error(str(error))
    paths = paths[:args.limit]
    if args.dry_run:
        for path in paths:
            print(path.relative_to(args.input))
        print(f"At most {len(paths) * (2 + args.evening + args.semseg)} image generations; existing stages are reused.")
        return
    if not config.GEMINI_API_KEY:
        parser.error("Set the GEMINI_API_KEY environment variable before generating images.")
    from google import genai
    from google.genai import types
    from tqdm.auto import tqdm

    def request_image(prompt, images):
        response = client.models.generate_content(
            model=config.MODEL, contents=[prompt, *images],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"],
                                                image_config=types.ImageConfig(image_size="1K")))
        return response_last_image_as_pil(response)

    with genai.Client(api_key=config.GEMINI_API_KEY) as client:
        for path in tqdm(paths, desc="Processing"):
            process_image(path, args.input, args.output, request_image,
                          evening=args.evening, semseg=args.semseg)
    print(f"Saved to {args.output}. Review masks, alignment and seams before training.")


if __name__ == "__main__":
    main()
