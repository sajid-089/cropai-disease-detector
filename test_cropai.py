"""
Test Model
==========
Loads your trained model and analyzes leaf images.
Put any leaf photos in test_images/ folder before running.

Usage: python test_model.py
"""

import os
import sys
import json
from PIL import Image
from transformers import pipeline


# --- Settings ---
MODEL_DIR = os.path.join("models", "final_model")
TEST_IMAGES_DIR = "test_images"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model not found at '{MODEL_DIR}'")
        print("Run 'python train.py' first.")
        sys.exit(1)

    print("Loading model...")
    classifier = pipeline(
        "image-classification",
        model=MODEL_DIR,
        device=-1,  # -1 means CPU
    )
    print("Model loaded.\n")
    return classifier


def show_model_info():
    """Print model accuracy and supported crops."""
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    if not os.path.exists(labels_path):
        return

    with open(labels_path, "r") as f:
        info = json.load(f)

    accuracy = round(info.get("accuracy", 0) * 100, 1)
    classes = info.get("num_classes", "?")
    crops = ", ".join(info.get("crops", []))

    print(f"  Accuracy : {accuracy}%")
    print(f"  Classes  : {classes}")
    print(f"  Crops    : {crops}")
    print()


def format_label(label):
    """
    Convert model label to readable text.
    Example: 'Tomato___Late_blight' -> 'Tomato  ->  Late blight'
    """
    if "___" in label:
        crop, disease = label.split("___", 1)
        crop = crop.replace("_", " ").replace(",", "").strip()
        disease = disease.replace("_", " ").strip()
        return f"{crop}  ->  {disease}"
    return label.replace("_", " ")


def analyze(image_path, classifier):
    """
    Analyze one leaf image and print the result.
    Shows top 5 predictions with confidence scores.
    """
    if not os.path.exists(image_path):
        print(f"  ERROR: File not found: {image_path}\n")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ERROR: Cannot open image: {e}\n")
        return

    # Run the model
    results = classifier(image)

    top = results[0]
    is_healthy = "healthy" in top["label"].lower()
    confidence = top["score"] * 100

    # Print header
    print(f"  File     : {os.path.basename(image_path)}")
    print(f"  Size     : {image.size[0]} x {image.size[1]} pixels")
    print()

    # Print main result
    if is_healthy:
        print(f"  RESULT   : HEALTHY CROP ({confidence:.1f}% confidence)")
    else:
        print(f"  RESULT   : DISEASE DETECTED ({confidence:.1f}% confidence)")
        print(f"  DISEASE  : {format_label(top['label'])}")

    # Print top 5 predictions
    print()
    print("  Top 5 Predictions:")
    print("  " + "-" * 52)

    for i, pred in enumerate(results[:5]):
        label = format_label(pred["label"])
        score = pred["score"] * 100
        bar = "#" * int(score / 100 * 20) + "." * (20 - int(score / 100 * 20))
        marker = "  <--" if i == 0 else ""
        print(f"  {i + 1}. {label}")
        print(f"     [{bar}] {score:.1f}%{marker}")

    print()
    print("  " + "=" * 52)
    print()


def main():
    print("=" * 60)
    print("  CROP DISEASE MODEL TESTER")
    print("=" * 60)
    print()

    # Load model
    classifier = load_model()

    # Show model info
    show_model_info()

    # Make sure test_images folder exists
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

    # Find all images in test_images/
    images = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Analyze all found images
    if images:
        print(f"Found {len(images)} image(s) in {TEST_IMAGES_DIR}/")
        print("Analyzing...\n")
        print("=" * 60)
        print()

        for filename in sorted(images):
            full_path = os.path.join(TEST_IMAGES_DIR, filename)
            analyze(full_path, classifier)

    else:
        print(f"No images found in {TEST_IMAGES_DIR}/")
        print()
        print("To get test images, run:")
        print("  python copy_test_images.py")
        print()
        print("Or manually put any leaf photo inside test_images/ folder.")
        print()

    # Interactive mode - let user type any image path
    print("=" * 60)
    print("  INTERACTIVE MODE")
    print("  Enter any image path to test it.")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("  Image path: ").strip().strip('"').strip("'")

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!")
                break

            print()
            analyze(user_input, classifier)

        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
            break


if __name__ == "__main__":
    main()