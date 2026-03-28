# Pakistan Crop Disease Detector

Custom trained MobileNetV2 model for Pakistani agriculture.

## Performance
- Test Accuracy: 95.5%
- Classes: 20
- Training Time: 106 minutes

## Supported Crops
- Corn_(maize)
- Grape
- Pepper,_bell
- Potato
- Tomato

## Usage
```python
from transformers import pipeline
classifier = pipeline("image-classification", model="YOUR_USERNAME/pakistan-crop-disease")
result = classifier("leaf_photo.jpg")
print(result)
```

## Details
- Base Model: google/mobilenet_v2_1.0_224
- Dataset: PlantVillage (filtered for Pakistani crops)
- Images per class: 300
- Built for AI Mustaqbil 2.0 Hackathon
