from pathlib import Path
import cv2
import numpy as np
import torch


def _to_uint8_img(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CxHxW -> HxWxC
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # normalizacja, jeśli jest w [0,1]
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype("uint8")
    else:
        arr = np.clip(arr, 0, 255).astype("uint8")

    return arr


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    img = _to_uint8_img(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    m = _to_uint8_img(mask)
    if m.ndim == 3:
        m = m[:, :, 0]

    m_bin = (m > 0).astype("uint8")

    overlay = img.copy()
    overlay[m_bin == 1] = color

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


def save_teacher_student_example(
        image_path,
        teacher_mask_path,
        student_probs,
        out_dir,
        prefix,
        thr=0.5,
        gt_mask_path=None,
):
    """
    Zapisuje zestaw obrazków:
      - image.png
      - teacher_pseudo.png
      - student_prob.png
      - student_bin.png
      - overlay_teacher.png
      - overlay_student.png
      - (opcjonalnie) gt_mask.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Nie udalo sie wczytac obrazu: {image_path}")

    teacher = cv2.imread(str(teacher_mask_path), cv2.IMREAD_GRAYSCALE)
    if teacher is None:
        raise FileNotFoundError(f"Nie udalo sie wczytac pseudo-maski teachera: {teacher_mask_path}")

    if isinstance(student_probs, torch.Tensor):
        probs = student_probs.detach().cpu().float()
        if probs.ndim == 4:
            probs = probs[0, 0]
        elif probs.ndim == 3:
            probs = probs[0]
    else:
        probs = np.asarray(student_probs, dtype=np.float32)

    probs = probs.clip(0.0, 1.0)
    probs_uint8 = (probs * 255.0).round().astype("uint8")
    student_bin = (probs >= thr).astype("uint8") * 255

    overlay_teacher = overlay_mask_on_image(img, teacher)
    overlay_student = overlay_mask_on_image(img, student_bin)

    cv2.imwrite(str(out_dir / f"{prefix}_image.png"), img)
    cv2.imwrite(str(out_dir / f"{prefix}_teacher_pseudo.png"), teacher)
    cv2.imwrite(str(out_dir / f"{prefix}_student_prob.png"), probs_uint8)
    cv2.imwrite(str(out_dir / f"{prefix}_student_bin.png"), student_bin)
    cv2.imwrite(str(out_dir / f"{prefix}_overlay_teacher.png"), overlay_teacher)
    cv2.imwrite(str(out_dir / f"{prefix}_overlay_student.png"), overlay_student)

    if gt_mask_path is not None:
        gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            cv2.imwrite(str(out_dir / f"{prefix}_gt.png"), gt)
