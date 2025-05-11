import os
from PIL import Image


def extract_patches(input_dir, output_dir, patch_size=128):
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            with Image.open(img_path) as img:
                width, height = img.size
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        box = (x, y, x + patch_size, y + patch_size)
                        patch = img.crop(box)
                        if patch.size == (patch_size, patch_size):
                            patch.save(os.path.join(
                                output_category_dir,
                                f"{os.path.splitext(img_name)[0]}_patch_{y}_{x}.png"
                            ))


if __name__ == "__main__":
    input_directory = "Textures"  # Original texture images
    output_directory = "Patches"  # Output directory for patches
    extract_patches(input_directory, output_directory)