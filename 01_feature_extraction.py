# Using PyRadiomics to extract features from a given image and mask

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import time
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

Image_Path = Path("")
Mask_Path = Path("")


# ---------------------------------------------------------------------
# PyRadiomics setup
# ---------------------------------------------------------------------

EXTRACTOR = featureextractor.RadiomicsFeatureExtractor()
IMAGE_TYPES = ["Original", "LoG", "Wavelet", "Square", "SquareRoot", "Logarithm", "Exponential", "Gradient", "LBP2D", "LBP3D"]
FEATURE_CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def main() -> None:
    start_total = time.time()

    if not Image_Path.exists():
        raise FileNotFoundError(f"Missing image: {Image_Path}")
    if not Mask_Path.exists():
        raise FileNotFoundError(f"Missing mask: {Mask_Path}")

    img = sitk.ReadImage(str(Image_Path))
    msk = sitk.ReadImage(str(Mask_Path))
    if sitk.GetArrayViewFromImage(msk).max() == 0:
        raise ValueError(f"Mask is empty: {Mask_Path}")

    all_features = {}
    for i_img, img_type in enumerate(IMAGE_TYPES, 1):
        print(f"[{i_img}/{len(IMAGE_TYPES)}] Image type: {img_type}")
        EXTRACTOR.disableAllImageTypes()
        EXTRACTOR.enableImageTypeByName(img_type)

        for i_fc, fc in enumerate(FEATURE_CLASSES, 1):
            print(f"[{i_fc}/{len(FEATURE_CLASSES)}] Feature class: {fc}")
            EXTRACTOR.disableAllFeatures()
            EXTRACTOR.enableFeatureClassByName(fc)

            feats = EXTRACTOR.execute(img, msk, label=1)
            feats = {k: v for k, v in feats.items() if not k.startswith("diagnostics")}
            all_features.update(feats)
            print(f"Done ({len(feats)} features)")

    print(f"Features extracted: {len(all_features)}")
    print(f"Total time: {(time.time() - start_total) / 60:.1f} minutes")
    #print(all_features)

if __name__ == "__main__":
    main()
