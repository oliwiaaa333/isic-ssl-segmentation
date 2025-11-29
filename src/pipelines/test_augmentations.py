import cv2
import matplotlib.pyplot as plt
from src.data.augmentations import get_augmentations_supervised
import pandas as pd
from pathlib import Path

manifest_path = Path("data/metadata/isic2018_task1_train_processed_test_hair_removal_small.csv") #do uzupelnienia odpowiednia sciezka
manifest = pd.read_csv(manifest_path)

augment = get_augmentations_supervised()

for i in range(5):
    img_url = Path(manifest.iloc[i]['image_url'])
    mask_url = Path(manifest.iloc[i]['mask_url'])

    img = cv2.imread(str(img_url))
    mask = cv2.imread(str(mask_url), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Nie udalo sie wczytac: {img_url} lub {mask_url}")
        continue

    augmented = augment(image=img, mask=mask)
    aug_img, aug_mask = augmented['image'], augmented['mask']

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Oryginalny obraz")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Oryginalna maska")

    aug_img_vis = aug_img.permute(1,2,0).cpu().numpy()
    aug_img_vis = (aug_img_vis - aug_img_vis.min()) / (aug_img_vis.max() - aug_img_vis.min())
    axs[2].imshow(aug_img_vis)
    axs[2].set_title("Po augmentacji")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"reports/figs/augmentation_sample_{i}.png")
    plt.close()