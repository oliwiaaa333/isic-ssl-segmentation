import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentations_maau():
    # augemntacje na podstawie publikacji PMID: 37189563
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, shear=(-10,10), fit_output=False, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_augmentations_noisy_student():
    # TODO: dodac augmentacje
    pass

def get_augmentations_dumm():
    # TODO: dodac augmentacje
    pass