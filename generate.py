import os
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm.auto import tqdm

from google import genai

from config import (
    GEMINI_API_KEY,
    MODEL,
    INPUT_DIR,
    OUT_MASKS_NAME,
    OUT_PATCHED_NAME,
    OUT_EVENING_NAME,
    OUT_OVERLAY_NAME,
    OUT_SEMSEG_NAME,
    MASK_PROMPT,
    PATCH_PROMPT,
    EVENING_PROMPT,
    SEMSEG_PROMPT,
)


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def response_last_image_as_pil(response):
    last_img = None

    for part in response.parts:
        if getattr(part, "inline_data", None) is None:
            continue

        try:
            img = part.as_image()
            if hasattr(img, "convert"):
                last_img = img
                continue
        except Exception:
            img = None

        blob = part.inline_data
        data = getattr(blob, "data", None)

        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception:
                data = None

        if isinstance(data, (bytes, bytearray)):
            last_img = PILImage.open(io.BytesIO(data))
            continue

        if img is not None:
            for attr in ("image_bytes", "data", "bytes"):
                b = getattr(img, attr, None)
                if isinstance(b, str):
                    try:
                        b = base64.b64decode(b)
                    except Exception:
                        b = None
                if isinstance(b, (bytes, bytearray)):
                    last_img = PILImage.open(io.BytesIO(b))
                    break

    return last_img


def binarize_mask(mask_img: PILImage.Image, size_wh):
    w, h = size_wh
    m = mask_img.convert("L").resize((w, h), PILImage.NEAREST)
    arr = np.array(m, dtype=np.uint8)
    arr = (arr >= 128).astype(np.uint8) * 255
    return PILImage.fromarray(arr, mode="L")


def ensure_size(img: PILImage.Image, size_wh):
    w, h = size_wh
    if img.size == (w, h):
        return img
    return img.resize((w, h), PILImage.LANCZOS)


def to_rgb(img: PILImage.Image):
    return img.convert("RGB")


def overlay_mask_on_image(
    img_rgb: PILImage.Image,
    mask_l: PILImage.Image,
    color=(255, 0, 0),
    alpha=0.35,
):
    img = to_rgb(img_rgb)
    m = mask_l.convert("L")
    base = np.array(img, dtype=np.uint8)
    mask = (np.array(m, dtype=np.uint8) >= 128)

    out = base.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * c

    out = np.clip(out, 0, 255).astype(np.uint8)
    return PILImage.fromarray(out, mode="RGB")


_PALETTE = np.array(
    [
        [0, 0, 0],         # background
        [0, 0, 255],       # curb
        [0, 255, 0],       # road markings
        [255, 0, 0],       # crosswalk
        [255, 255, 255],   # sign
    ],
    dtype=np.int32,
)


def quantize_to_palette(mask_rgb: PILImage.Image, size_wh):
    w, h = size_wh

    m = to_rgb(mask_rgb).resize((w, h), PILImage.NEAREST)

    arr = np.array(m, dtype=np.int32)

    flat = arr.reshape(-1, 3)[:, None, :]
    pal = _PALETTE[None, :, :]

    d2 = np.sum((flat - pal) ** 2, axis=-1)
    idx = np.argmin(d2, axis=1)

    out = _PALETTE[idx].astype(np.uint8).reshape(h, w, 3)

    return PILImage.fromarray(out, mode="RGB")


def main():
    assert GEMINI_API_KEY, "Set GEMINI_API_KEY in config.py"

    client = genai.Client(api_key=GEMINI_API_KEY)

    in_dir = Path(INPUT_DIR)

    script_dir = Path(__file__).resolve().parent

    masks_dir = script_dir / OUT_MASKS_NAME
    patched_dir = script_dir / OUT_PATCHED_NAME
    evening_dir = script_dir / OUT_EVENING_NAME
    overlay_dir = script_dir / OUT_OVERLAY_NAME
    semseg_dir = script_dir / OUT_SEMSEG_NAME

    masks_dir.mkdir(parents=True, exist_ok=True)
    patched_dir.mkdir(parents=True, exist_ok=True)
    evening_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    semseg_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(in_dir.glob("**/*")) if p.is_file() and p.suffix.lower() in EXTS]
    assert paths, f"No images found in {INPUT_DIR}"

    for p in tqdm(paths, desc="Processing"):
        stem = p.stem

        out_mask_path = masks_dir / f"{stem}.png"
        out_patched_path = patched_dir / f"{stem}.png"
        out_evening_path = evening_dir / f"{stem}.png"
        out_overlay_path = overlay_dir / f"{stem}.png"
        out_semseg_path = semseg_dir / f"{stem}.png"

        if (
            out_mask_path.exists()
            and out_patched_path.exists()
            and out_evening_path.exists()
            and out_overlay_path.exists()
            and out_semseg_path.exists()
        ):
            continue

        orig = to_rgb(PILImage.open(p))
        W, H = orig.size

        # semseg mask
        if not out_semseg_path.exists():
            resp = client.models.generate_content(
                model=MODEL,
                contents=[SEMSEG_PROMPT, orig],
            )
            raw_sem = response_last_image_as_pil(resp)
            sem = quantize_to_palette(raw_sem, (W, H))
            sem.save(out_semseg_path)

        # patch mask
        if not out_mask_path.exists():
            resp = client.models.generate_content(
                model=MODEL,
                contents=[MASK_PROMPT, orig],
            )
            raw_mask = response_last_image_as_pil(resp)
            mask_l = binarize_mask(raw_mask, (W, H))
            mask_l.save(out_mask_path)
        else:
            mask_l = binarize_mask(PILImage.open(out_mask_path), (W, H))

        # patches
        if not out_patched_path.exists():
            mask_rgb = mask_l.convert("RGB")
            resp = client.models.generate_content(
                model=MODEL,
                contents=[PATCH_PROMPT, orig, mask_rgb],
            )
            patched = ensure_size(to_rgb(response_last_image_as_pil(resp)), (W, H))
            patched.save(out_patched_path)

        # overlay
        if not out_overlay_path.exists():
            patched_img = to_rgb(PILImage.open(out_patched_path))
            overlay = overlay_mask_on_image(patched_img, mask_l, color=(255, 0, 0), alpha=0.35)
            overlay.save(out_overlay_path)

        # evening
        if not out_evening_path.exists():
            patched_img = to_rgb(PILImage.open(out_patched_path))
            resp = client.models.generate_content(
                model=MODEL,
                contents=[EVENING_PROMPT, patched_img],
            )
            evening = ensure_size(to_rgb(response_last_image_as_pil(resp)), (W, H))
            evening.save(out_evening_path)

    print("Done.")
    print("Masks:   ", str(masks_dir))
    print("Patched: ", str(patched_dir))
    print("Overlay: ", str(overlay_dir))
    print("SemSeg:  ", str(semseg_dir))
    print("Evening: ", str(evening_dir))


if __name__ == "__main__":
    main()
