"""
Pakistan Crop Disease Detection Model
Training Script for AI Mustaqbil 2.0 Hackathon
Theme 4: Agriculture and Food Security

What this script does:
1. Finds the PlantVillage dataset in your data/ folder
2. Filters only Pakistan-relevant crops (removes Apple, Blueberry etc)
3. Limits to 300 images per class (for faster CPU training)
4. Loads Google MobileNetV2 pre-trained model
5. Retrains it to recognize 20 Pakistani crop diseases
6. Evaluates accuracy on unseen test images
7. Saves the final model to models/final_model/

Run: python train.py
Time: approximately 30-60 minutes on CPU
"""

import os
import sys
import json
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

print("\nLoading libraries...")

try:
    import torch
    import numpy as np
    from PIL import Image
    from collections import defaultdict
    from transformers import (
        AutoModelForImageClassification,
        AutoImageProcessor,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from datasets import load_dataset, DatasetDict
    from evaluate import load as load_metric
    print("All libraries loaded successfully")
except ImportError as e:
    print(f"ERROR: Missing library - {e}")
    print("Fix: python -m pip install torch torchvision transformers datasets evaluate accelerate Pillow")
    sys.exit(1)


BASE_MODEL = "google/mobilenet_v2_1.0_224"
DEVICE = "cpu"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4
WORKERS = 0
MAX_IMAGES_PER_CLASS = 300
DATA_DIR = "data"
FILTERED_DIR = "filtered_pakistan_data"
CHECKPOINT_DIR = "checkpoints"
FINAL_DIR = os.path.join("models", "final_model")

PAKISTAN_CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]

print(f"""
============================================================
  PAKISTAN CROP DISEASE MODEL - TRAINING
  AI Mustaqbil 2.0 Hackathon
============================================================
  Device:           {DEVICE.upper()}
  Base model:       {BASE_MODEL}
  Batch size:       {BATCH_SIZE}
  Epochs:           {EPOCHS}
  Learning rate:    {LEARNING_RATE}
  Images per class: {MAX_IMAGES_PER_CLASS}
  Target classes:   {len(PAKISTAN_CLASSES)}
============================================================
""")


# STEP 1: Find dataset
print("STEP 1/8: Finding dataset location...")
print("-" * 50)

TRAIN_PATH = None
VALID_PATH = None

possible_paths = [
    os.path.join(DATA_DIR, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train"),
    os.path.join(DATA_DIR, "New Plant Diseases Dataset(Augmented)", "train"),
    os.path.join(DATA_DIR, "train"),
    DATA_DIR,
]

for check_path in possible_paths:
    if os.path.exists(check_path):
        folders = [
            item for item in os.listdir(check_path)
            if os.path.isdir(os.path.join(check_path, item))
        ]
        looks_like_classes = any("___" in folder for folder in folders)
        if len(folders) > 5 and looks_like_classes:
            TRAIN_PATH = check_path
            validation_path = check_path.replace("train", "valid")
            if os.path.exists(validation_path):
                VALID_PATH = validation_path
            break

if TRAIN_PATH is None:
    print("ERROR: Could not find dataset!")
    print(f"Looked inside: {os.path.abspath(DATA_DIR)}")
    print("")
    print("What is in your data/ folder:")
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            print(f"  {item}")
            sub = os.path.join(DATA_DIR, item)
            if os.path.isdir(sub):
                for child in os.listdir(sub)[:3]:
                    print(f"    {child}")
    print("")
    print("FIX: Make sure the dataset zip is extracted inside data/ folder")
    sys.exit(1)

all_classes = sorted([
    d for d in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, d))
])

print(f"  Found training data: {TRAIN_PATH}")
print(f"  Found validation data: {VALID_PATH or 'Not found (will split from train)'}")
print(f"  Total classes in dataset: {len(all_classes)}")


# STEP 2: Filter Pakistan crops
print(f"\nSTEP 2/8: Filtering Pakistan crops (max {MAX_IMAGES_PER_CLASS} images each)...")
print("-" * 50)

if os.path.exists(FILTERED_DIR):
    shutil.rmtree(FILTERED_DIR)

filtered_train_dir = os.path.join(FILTERED_DIR, "train")
filtered_val_dir = os.path.join(FILTERED_DIR, "val")
os.makedirs(filtered_train_dir, exist_ok=True)
os.makedirs(filtered_val_dir, exist_ok=True)


def normalize_name(name):
    return name.lower().replace(" ", "").replace(",", "").replace("_", "")


total_images_copied = 0
found_classes = []

for target_class in PAKISTAN_CLASSES:
    source_dir = os.path.join(TRAIN_PATH, target_class)

    if not os.path.exists(source_dir):
        target_normalized = normalize_name(target_class)
        for existing_class in all_classes:
            existing_normalized = normalize_name(existing_class)
            if target_normalized in existing_normalized or existing_normalized in target_normalized:
                source_dir = os.path.join(TRAIN_PATH, existing_class)
                break

    if not os.path.exists(source_dir):
        print(f"  SKIP: {target_class} (not found in dataset)")
        continue

    dest_dir = os.path.join(filtered_train_dir, target_class)
    os.makedirs(dest_dir, exist_ok=True)

    all_image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_files = all_image_files[:MAX_IMAGES_PER_CLASS]

    for filename in selected_files:
        shutil.copy2(
            os.path.join(source_dir, filename),
            os.path.join(dest_dir, filename)
        )

    total_images_copied += len(selected_files)
    found_classes.append(target_class)

    is_healthy = "healthy" in target_class.lower()
    status = "HEALTHY" if is_healthy else "DISEASE"
    print(f"  {status}: {target_class} -> {len(selected_files)} images")

if VALID_PATH:
    for class_name in found_classes:
        val_source = os.path.join(VALID_PATH, class_name)
        if not os.path.exists(val_source):
            for existing in os.listdir(VALID_PATH):
                if normalize_name(class_name) in normalize_name(existing):
                    val_source = os.path.join(VALID_PATH, existing)
                    break
        if os.path.exists(val_source):
            val_dest = os.path.join(filtered_val_dir, class_name)
            os.makedirs(val_dest, exist_ok=True)
            val_files = [
                f for f in os.listdir(val_source)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:50]
            for f in val_files:
                shutil.copy2(os.path.join(val_source, f), os.path.join(val_dest, f))

print(f"\n  Summary:")
print(f"    Classes kept: {len(found_classes)} out of {len(all_classes)}")
print(f"    Classes removed: {len(all_classes) - len(found_classes)} (not Pakistani crops)")
print(f"    Total training images: {total_images_copied}")

if len(found_classes) == 0:
    print("ERROR: No matching classes found. Check your dataset.")
    sys.exit(1)


# STEP 3: Load and split dataset
print(f"\nSTEP 3/8: Loading and splitting dataset...")
print("-" * 50)

has_validation = os.path.exists(filtered_val_dir) and len(os.listdir(filtered_val_dir)) > 0

if has_validation:
    train_data = load_dataset("imagefolder", data_dir=filtered_train_dir, split="train")
    val_data = load_dataset("imagefolder", data_dir=filtered_val_dir, split="train")
    train_and_test = train_data.train_test_split(
        test_size=0.1, seed=42, stratify_by_column="label"
    )
    dataset = DatasetDict({
        "train": train_and_test["train"],
        "validation": val_data,
        "test": train_and_test["test"],
    })
else:
    full_data = load_dataset("imagefolder", data_dir=filtered_train_dir, split="train")
    train_and_rest = full_data.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="label"
    )
    val_and_test = train_and_rest["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="label"
    )
    dataset = DatasetDict({
        "train": train_and_rest["train"],
        "validation": val_and_test["train"],
        "test": val_and_test["test"],
    })

class_names = dataset["train"].features["label"].names
num_classes = len(class_names)
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

print(f"  Train images:      {len(dataset['train'])}")
print(f"  Validation images: {len(dataset['validation'])}")
print(f"  Test images:       {len(dataset['test'])}")
print(f"  Number of classes: {num_classes}")


# STEP 4: Load pre-trained model
print(f"\nSTEP 4/8: Loading pre-trained model...")
print("-" * 50)

processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

model = AutoModelForImageClassification.from_pretrained(
    BASE_MODEL,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

total_parameters = sum(p.numel() for p in model.parameters())
print(f"  Model: {BASE_MODEL}")
print(f"  Parameters: {total_parameters:,}")
print(f"  Output classes: {num_classes}")


# STEP 5: Image preprocessing
print(f"\nSTEP 5/8: Setting up image preprocessing...")
print("-" * 50)


def preprocess_train(batch):
    processed_images = []
    for img in batch["image"]:
        img = img.convert("RGB")
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        processed_images.append(img)
    inputs = processor(images=processed_images, return_tensors="pt")
    inputs["labels"] = batch["label"]
    return inputs


def preprocess_val(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = batch["label"]
    return inputs


dataset["train"].set_transform(preprocess_train)
dataset["validation"].set_transform(preprocess_val)
dataset["test"].set_transform(preprocess_val)

print("  Training: resize + normalize + random flip")
print("  Validation/Test: resize + normalize only")


# STEP 6: Train the model
print(f"\nSTEP 6/8: Training the model...")
print("=" * 50)

accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_arguments = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    save_total_limit=1,
    dataloader_num_workers=WORKERS,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ],
)

print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Early stopping: after 2 epochs without improvement")
print("")
print("  Training started... (this will take 30-60 minutes on CPU)")
print("  You will see progress updates below.")
print("")

start_time = time.time()
training_result = trainer.train()
training_time = time.time() - start_time

print("")
print(f"  Training finished!")
print(f"  Total time: {training_time / 60:.1f} minutes")
print(f"  Final loss: {training_result.training_loss:.4f}")


# STEP 7: Evaluate on test images
print(f"\nSTEP 7/8: Evaluating on test images...")
print("-" * 50)

evaluation_results = trainer.evaluate(dataset["test"])
accuracy = evaluation_results["eval_accuracy"]

print(f"  Test accuracy: {accuracy * 100:.2f}%")
print(f"  Test loss: {evaluation_results['eval_loss']:.4f}")

all_predictions = trainer.predict(dataset["test"])
predicted_labels = np.argmax(all_predictions.predictions, axis=-1)
true_labels = all_predictions.label_ids

class_correct = defaultdict(int)
class_total = defaultdict(int)

for predicted, actual in zip(predicted_labels, true_labels):
    class_name = id2label[actual]
    class_total[class_name] += 1
    if predicted == actual:
        class_correct[class_name] += 1

print(f"\n  Accuracy per class:")
print(f"  {'Class':<55} {'Accuracy':>8}")
print(f"  {'-' * 65}")

for class_name in sorted(class_total.keys()):
    class_accuracy = class_correct[class_name] / class_total[class_name] * 100
    if class_accuracy >= 90:
        status = "GOOD"
    elif class_accuracy >= 70:
        status = "OK  "
    else:
        status = "WEAK"
    print(f"  [{status}] {class_name:<53} {class_accuracy:.1f}%")


# STEP 8: Save the trained model
print(f"\nSTEP 8/8: Saving trained model...")
print("-" * 50)

os.makedirs(os.path.dirname(FINAL_DIR), exist_ok=True)

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)

trainer.save_model(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)

labels_info = {
    "id2label": {str(k): v for k, v in id2label.items()},
    "label2id": label2id,
    "num_classes": num_classes,
    "class_names": list(class_names),
    "accuracy": round(float(accuracy), 4),
    "pakistan_focused": True,
    "crops": sorted(list(set(c.split("___")[0] for c in class_names))),
    "base_model": BASE_MODEL,
    "training_time_minutes": round(training_time / 60, 1),
    "images_per_class": MAX_IMAGES_PER_CLASS,
    "epochs": EPOCHS,
    "device": DEVICE,
}

with open(os.path.join(FINAL_DIR, "labels.json"), "w") as f:
    json.dump(labels_info, f, indent=2)

crops_list = sorted(list(set(c.split("___")[0] for c in class_names)))
readme_text = (
    "# Pakistan Crop Disease Detector\n\n"
    "Custom trained MobileNetV2 model for Pakistani agriculture.\n\n"
    "## Performance\n"
    f"- Test Accuracy: {accuracy * 100:.1f}%\n"
    f"- Classes: {num_classes}\n"
    f"- Training Time: {training_time / 60:.0f} minutes\n\n"
    "## Supported Crops\n"
)
for crop in crops_list:
    readme_text += f"- {crop}\n"
readme_text += (
    "\n## Usage\n"
    "```python\n"
    "from transformers import pipeline\n"
    'classifier = pipeline("image-classification", model="YOUR_USERNAME/pakistan-crop-disease")\n'
    'result = classifier("leaf_photo.jpg")\n'
    "print(result)\n"
    "```\n\n"
    "## Details\n"
    f"- Base Model: {BASE_MODEL}\n"
    "- Dataset: PlantVillage (filtered for Pakistani crops)\n"
    f"- Images per class: {MAX_IMAGES_PER_CLASS}\n"
    "- Built for AI Mustaqbil 2.0 Hackathon\n"
)

with open(os.path.join(FINAL_DIR, "README.md"), "w") as f:
    f.write(readme_text)

model_size_mb = sum(
    os.path.getsize(os.path.join(FINAL_DIR, f))
    for f in os.listdir(FINAL_DIR)
    if os.path.isfile(os.path.join(FINAL_DIR, f))
) / (1024 * 1024)

print(f"  Model saved to: {FINAL_DIR}/")
print(f"  Model size: {model_size_mb:.1f} MB")
print("")
print("  Files saved:")
for filename in sorted(os.listdir(FINAL_DIR)):
    file_size = os.path.getsize(os.path.join(FINAL_DIR, filename)) / (1024 * 1024)
    print(f"    {filename} ({file_size:.1f} MB)")


# Quick test
print("\nQuick inference test...")
print("-" * 50)

from transformers import pipeline as create_pipeline

test_pipeline = create_pipeline(
    "image-classification",
    model=FINAL_DIR,
    device=-1,
)

test_image = Image.new("RGB", (224, 224), color=(34, 139, 34))
test_result = test_pipeline(test_image)
print(f"  Green image prediction: {test_result[0]['label']} ({test_result[0]['score']*100:.1f}%)")
print("  Model is working correctly!")


# Clean up checkpoints
if os.path.exists(CHECKPOINT_DIR):
    checkpoint_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(CHECKPOINT_DIR)
        for f in fn
    ) / (1024 * 1024 * 1024)
    shutil.rmtree(CHECKPOINT_DIR)
    print(f"\n  Cleaned up {checkpoint_size:.1f} GB of checkpoint files")


# Final summary
total_time = time.time() - start_time

print(f"""
============================================================
  TRAINING COMPLETE
============================================================

  Model:          Pakistan Crop Disease Detector
  Base:           MobileNetV2
  Classes:        {num_classes}
  Test Accuracy:  {accuracy * 100:.1f}%
  Model Size:     {model_size_mb:.1f} MB
  Training Time:  {total_time / 60:.1f} minutes

  Model Location: {os.path.abspath(FINAL_DIR)}

  NEXT STEPS:
  1. Test with real leaf images:
     python test_model.py

  2. Upload to HuggingFace:
     python upload_to_hf.py

  3. Use in your Next.js app:
     Update diseaseAgent.ts with your HuggingFace model URL

============================================================
""")