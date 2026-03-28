"""import os
import sys
import json


MODEL_DIR = os.path.join("models", "final_model")


def main():

    # ---------------------------------------------------
    # CHECK MODEL FOLDER
    # ---------------------------------------------------

    if not os.path.exists(MODEL_DIR):
        print("ERROR: Model folder not found!")
        print("Run 'python train.py' first.")
        sys.exit(1)

    print("Model folder found. Files:")
    for f in sorted(os.listdir(MODEL_DIR)):
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")


    # ---------------------------------------------------
    # FIX README (YE MAIN FIX HAI - INFERENCE API KE LIYE)
    # ---------------------------------------------------

    readme = """---
license: mit
tags:
- image-classification
- mobilenet_v2
- agriculture
- crop-disease
- pakistan
pipeline_tag: image-classification
library_name: transformers
metrics:
- accuracy
model-index:
- name: pakistan-crop-disease
  results:
  - task:
      type: image-classification
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.955
---

# Pakistan Crop Disease Detector

Fine-tuned MobileNetV2 model for detecting crop diseases.

## Performance
- **Accuracy**: 95.5%
- **Classes**: 20
- **Base Model**: google/mobilenet_v2_1.0_224

## Supported Crops
- Corn (Maize) - 4 classes
- Grape - 4 classes
- Pepper Bell - 2 classes
- Potato - 3 classes
- Tomato - 7 classes

## Usage

```python
from transformers import pipeline
classifier = pipeline("image-classification", model="REPO_PLACEHOLDER")
result = classifier("leaf_photo.jpg")
print(result)"""
readme_path = os.path.join(MODEL_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
print("\nREADME.md fixed with pipeline_tag!")


# ---------------------------------------------------
# TEST MODEL LOCALLY
# ---------------------------------------------------

print("\n" + "=" * 50)
test = input("Test model locally? (yes/no): ").strip().lower()

if test in ("yes", "y"):

    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image
        import requests
        from io import BytesIO

        print("Loading model...")
        classifier = hf_pipeline("image-classification", model=MODEL_DIR, top_k=3)
        print("Model loaded!\n")

        # Test with online images
        urls = {
            "Tomato leaf": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Tomato_je.jpg/800px-Tomato_je.jpg",
            "Plant leaf": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Acer_palmatum_leaf.jpg/800px-Acer_palmatum_leaf.jpg",
        }

        for name, url in urls.items():
            try:
                img = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
                results = classifier(img)
                print(f"{name}:")
                for r in results:
                    print(f"  {r['label']}: {r['score']*100:.1f}%")
                print()
            except Exception as e:
                print(f"{name}: Error - {e}\n")

        # Test with leaf.jpg if exists
        if os.path.exists("leaf.jpg"):
            img = Image.open("leaf.jpg").convert("RGB")
            results = classifier(img)
            print("leaf.jpg:")
            for r in results:
                print(f"  {r['label']}: {r['score']*100:.1f}%")
            print()

        # Test with test_images folder
        for folder in ["test_images", "test_image"]:
            if os.path.exists(folder):
                print(f"Testing {folder}/:")
                count = 0
                for root, dirs, files in os.walk(folder):
                    for fname in files:
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            path = os.path.join(root, fname)
                            try:
                                img = Image.open(path).convert("RGB")
                                results = classifier(img)
                                print(f"\n  {fname}:")
                                for r in results:
                                    print(f"    {r['label']}: {r['score']*100:.1f}%")
                            except Exception as e:
                                print(f"  {fname}: Error - {e}")
                            count += 1
                            if count >= 10:
                                break
                    if count >= 10:
                        break
                print()

        print("Testing complete!\n")

    except ImportError:
        print("Install required: pip install transformers torch pillow requests")
        print("Skipping test...\n")


# ---------------------------------------------------
# UPLOAD TO HUGGINGFACE
# ---------------------------------------------------

print("=" * 50)
upload = input("Upload to HuggingFace? (yes/no): ").strip().lower()

if upload not in ("yes", "y"):
    print("Done. Run again when ready to upload.")
    return

token = input("HuggingFace WRITE token (hf_...): ").strip()
username = input("HuggingFace username: ").strip()
repo_input = input("Repository name [pakistan-crop-disease]: ").strip()
repo_name = repo_input if repo_input else "pakistan-crop-disease"
repo_id = f"{username}/{repo_name}"

if not token or not username:
    print("ERROR: Token and username required!")
    sys.exit(1)

# Update README with repo id
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("REPO_PLACEHOLDER", repo_id)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nUploading to: https://huggingface.co/{repo_id}")
confirm = input("Confirm? (yes/no): ").strip().lower()

if confirm not in ("yes", "y"):
    print("Cancelled.")
    return

try:
    from huggingface_hub import HfApi, login

    print("\n[1/3] Logging in...")
    login(token=token)
    print("Done.")

    api = HfApi()

    print("[2/3] Creating repository...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    print("Done.")

    print("[3/3] Uploading files...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Deploy crop disease model with Inference API",
    )

    print(f"""==================================================
UPLOAD SUCCESSFUL!
Model URL: https://huggingface.co/{repo_id}

Wait 2-5 minutes for Inference API to activate.
Then upload an image on the model page to test!

Python usage:
from transformers import pipeline
classifier = pipeline("image-classification", model="{repo_id}")
result = classifier("leaf_photo.jpg")
print(result)
""")
if name == "main":
main()

"""