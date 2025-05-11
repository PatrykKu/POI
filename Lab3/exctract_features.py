import os
import numpy as np
import pandas as pd
from skimage import io, color, feature


def compute_features(patches_dir, output_csv):
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 degrees
    features_list = []

    for category in os.listdir(patches_dir):
        category_dir = os.path.join(patches_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for patch_file in os.listdir(category_dir):
            patch_path = os.path.join(category_dir, patch_file)
            image = io.imread(patch_path)
            gray = color.rgb2gray(image)
            gray_ubyte = (gray * 255).astype(np.uint8)
            quantized = gray_ubyte // 4  # Reduce to 64 levels (5 bits)

            glcm_features = []
            for d in distances:
                for angle in angles:
                    glcm = feature.graycomatrix(
                        quantized,
                        distances=[d],
                        angles=[angle],
                        levels=64,
                        symmetric=True
                    )
                    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
                    for prop in properties:
                        glcm_features.append(feature.graycoprops(glcm, prop)[0, 0])

            glcm_features.append(category)
            features_list.append(glcm_features)

    # Generate column names
    columns = []
    for d in distances:
        for a_deg in [0, 45, 90, 135]:
            for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
                columns.append(f'd{d}_a{a_deg}_{prop}')
    columns.append('category')

    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    patches_directory = "Patches"
    output_csv = "texture_features.csv"
    compute_features(patches_directory, output_csv)