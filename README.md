---
language: en
license: mit
tags:
- image-classification
- pytorch
- vision-transformer
- plant-disease
- agriculture
datasets:
- custom-plant-village
metrics:
- accuracy
model-index:
- name: Pakistan Crop Disease Detector
  results:
  - task:
      type: image-classification
    metrics:
    - type: accuracy
      value: 95.6
      name: Test Accuracy
---

# 🌿 Pakistan Crop Disease Detector (AI Mustaqbil 2.0)

This model is designed to detect various diseases in crops common in Pakistan (Tomato, Potato, Corn, etc.). It was trained on a dataset of 10,000 images using PyTorch and fine-tuned for high accuracy.



## 🚀 Model Details
- **Accuracy:** 95.6% on Validation Set
- **Architecture:* Fine‑tuned **MobileNetV2** model to detect crop diseases
from leaf images (Tomato, Potato, Corn, Pepper, Grape)
- **Input Size:** 224x224 pixels
- **Training Epochs:** 10+ (Early stopping applied)

## 🛠 Usage in Next.js (Inference API)

You can call this model via the Hugging Face Inference API. In your `diseaseAgent.ts`:

```typescript
const DISEASE_MODEL = "YOUR_USERNAME/pakistan-crop-disease";

async function query(filename) {
    const data = fs.readFileSync(filename);
    const response = await fetch(
        `https://api-inference.huggingface.co/models/${DISEASE_MODEL}`,
        {
            headers: { Authorization: `Bearer ${HF_TOKEN}` },
            method: "POST",
            body: data,
        }
    );
    const result = await response.json();
    return result;
}