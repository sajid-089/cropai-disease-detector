import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline(
    "image-classification",
    model="models/final_model",
    top_k=5
)

def predict(image):
    if image is None:
        return {}
    results = classifier(image)
    return {r["label"]: round(r["score"], 4) for r in results}

# Create web app
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=5, label="Disease Prediction"),
    title="🌾 Pakistan Crop Disease Detector",
    description="Upload a leaf image to detect crop disease. Supports Corn, Grape, Pepper, Potato, and Tomato.",
    examples=["leaf.jpg"] if __import__("os").path.exists("leaf.jpg") else None,
)

demo.launch()