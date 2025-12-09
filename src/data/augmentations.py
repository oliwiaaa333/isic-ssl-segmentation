import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentations(mode: str):
    if mode == "supervised":
        return get_augmentations_supervised()
    if mode == "teacher":
        return get_augmentations_teacher()
    if mode == "student":
        return get_augmentations_student()
    if mode == "teacher_dumm":
        return get_augmentations_teacher_dumm()
    if mode == "student_dumm":
        return get_augmentations_student_dumm()
    raise ValueError(f"Unknown augmentations mode: {mode}")


def get_augmentations_supervised():
    # augemntacje na podstawie publikacji PMID: 37189563
    return A.Compose([
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, shear=(-10,10), fit_output=False, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# augmentacje dla noisy-student i dumm
def get_augmentations_teacher():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.02,
            p=0.2
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_augmentations_student():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.1,
                           rotate_limit=20,
                           border_mode=0,
                           p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25,
                                   p=0.7),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30,
                             val_shift_limit=20,
                             p=0.5),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.CoarseDropout(max_holes=6,
                        max_height=32,
                        max_width=32,
                        fill_value=0,
                        p=0.5),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_augmentations_teacher_dumm():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),
        ToTensorV2(),
    ])


def get_augmentations_student_dumm():
    return A.Compose([
        A.Resize(256, 256),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=20,
            border_mode=0,
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.7
        ),

        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GaussNoise(p=0.3),

        A.CoarseDropout(
            max_holes=6,
            hole_height=32,
            hole_width=32,
            p=0.3
        ),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),

        ToTensorV2(),
    ])
