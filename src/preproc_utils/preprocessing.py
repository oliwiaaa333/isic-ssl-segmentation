from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def remove_hairs(img: Image.Image) -> Image.Image:
    # na podstawie publikacji PMC10969337
    # Konwersja PIL do OpenCV (numpy)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # utworzenie maski wlosow za pomoca blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # binaryzacja maski wlosow
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpainting do usuniecia wlosow
    inpainted = cv2.inpaint(img_cv, hair_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    # konwersja do pil
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


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
