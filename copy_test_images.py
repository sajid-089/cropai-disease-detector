"""
Copy Test Images
================
Copies sample images from the dataset into test_images/ folder.
Run this before test_model.py.

Usage: python copy_test_images.py
"""

import os
import shutil


# --- Settings ---
DATA_DIR = "data"
OUTPUT_DIR = "test_images"
IMAGES_PER_CLASS = 2

CLASSES_TO_TEST = [
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]


def find_valid_folder():
    """Find the validation folder inside the dataset."""
    candidates = [
        os.path.join(DATA_DIR, "New Plant Diseases Dataset(Augmented)",
                     "New Plant Diseases Dataset(Augmented)", "valid"),
        os.path.join(DATA_DIR, "New Plant Diseases Dataset(Augmented)", "valid"),
        os.path.join(DATA_DIR, "valid"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def copy_images(valid_folder):
    """Copy sample images from each class into test_images/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0

    for class_name in CLASSES_TO_TEST:
        class_path = os.path.join(valid_folder, class_name)

        # Try to find the folder even if name is slightly different
        if not os.path.exists(class_path):
            for folder in os.listdir(valid_folder):
                if class_name.lower().replace("_", "") in folder.lower().replace("_", ""):
                    class_path = os.path.join(valid_folder, folder)
                    break

        if not os.path.exists(class_path):
            print(f"  [SKIP]  {class_name} - not found")
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        copied = 0
        for filename in images[:IMAGES_PER_CLASS]:
            src = os.path.join(class_path, filename)
            dst = os.path.join(OUTPUT_DIR, f"{class_name}__{filename}")
            shutil.copy2(src, dst)
            copied += 1
            total += 1

        label = "HEALTHY" if "healthy" in class_name.lower() else "DISEASE"
        print(f"  [{label}]  {class_name}  ->  {copied} images")

    return total


def main():
    print("=" * 60)
    print("  COPY TEST IMAGES")
    print("=" * 60)

    print("\nLooking for dataset validation folder...")
    valid_folder = find_valid_folder()

    if valid_folder is None:
        print("\nERROR: Could not find the validation folder.")
        print("Make sure your dataset is extracted inside the data/ folder.")
        return

    print(f"Found: {valid_folder}")
    print(f"\nCopying {IMAGES_PER_CLASS} images per class...\n")

    total = copy_images(valid_folder)

    print(f"\n{'-' * 60}")
    print(f"Done. {total} images copied to {OUTPUT_DIR}/")
    print(f"\nNext step: python test_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()