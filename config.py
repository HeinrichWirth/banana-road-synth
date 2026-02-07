# API Key and model
GEMINI_API_KEY = "key_api"
MODEL = "gemini-3-pro-image-preview"

# path photo
INPUT_DIR = "path"

OUT_MASKS_NAME = "out_masks"
OUT_PATCHED_NAME = "out_patches"
OUT_EVENING_NAME = "out_evening"
OUT_OVERLAY_NAME = "out_overlay"
OUT_SEMSEG_NAME = "out_semseg"

#prompts
MASK_PROMPT = """Generate a binary segmentation mask of asphalt road repair patches.
The mask should represent realistic, randomly shaped road patches that DO NOT exist in the original image.
Patches must appear only on the asphalt road surface.
No patches on sidewalks, curbs, grass, buildings, cars, poles, or road markings.
Patch shapes should be irregular, organic, and varied in size, similar to real asphalt repairs.
The mask must be white (255) for patches and black (0) for everything else.
Do not modify the original image.
Do not add textures, colors, lighting, or shadows.
Output only the segmentation mask image (no text)."""

PATCH_PROMPT = """You are given two images:
(1) the original photo
(2) a binary mask where WHITE (255) indicates the exact pixels to edit and BLACK (0) indicates pixels that must remain unchanged.

Task: Generate realistic asphalt road repair patches ONLY where the mask is white.
The patches must look OLD and worn (not fresh): low-contrast, slightly blended into the existing road, almost the same color as the road.
Do NOT change anything outside the white mask.
Do NOT touch sidewalks, curbs, grass, buildings, cars, poles, or road markings.
No new objects, no extra shadows, no added text.
Output only the edited photo (no text)."""

EVENING_PROMPT = """Change the time of day to early evening / dusk.
Keep the scene, geometry, and all objects exactly the same; only change lighting and color grading realistically.
Output only the edited photo (no text)."""

SEMSEG_PROMPT = """Segment the image and output a single RGB mask image with EXACT colors:
- Curb / road curb edges (бордюр): (0, 0, 255)
- Road lane markings / road markings EXCLUDING crosswalk: (0, 255, 0)
- Crosswalk markings (пешеходный переход / "zebra"): (255, 0, 0)
- Road Sign: (255, 255, 255)
- Everything else: (0, 0, 0)

Rules:
- Output ONLY the mask image, no text.
- Use flat solid colors only, no gradients, no anti-aliasing.
- Do not modify the original photo.
"""
