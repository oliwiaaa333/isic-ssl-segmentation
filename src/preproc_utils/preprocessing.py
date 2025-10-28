from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")

def remove_hairs(img: Image.Image) -> Image.Image:
    # Placeholder do usuwania włosów; na razie filtr medianowy
    return img.filter(ImageFilter.MedianFilter(size=3))

def resize_image(img: Image.Image, size: tuple[int, int], is_mask: bool = False) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize(size, resample=resample)

def normalize_image(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def save_image(arr: np.ndarray, path: Path, fmt: str = "png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path.with_suffix(".npy"), arr)
    else:
        if arr.ndim == 3:
            img = Image.fromarray((arr * 255).astype(np.uint8))
        else:
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img.save(path.with_suffix(".png"))
