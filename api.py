from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Crop Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
classifier = pipeline(
    "image-classification",
    model="models/final_model",
    top_k=5
)
print("Model loaded!")


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "Pakistan Crop Disease Detector",
        "usage": "POST /predict with image file"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = classifier(image)

    return {
        "success": True,
        "predictions": [
            {
                "disease": r["label"],
                "confidence": round(r["score"] * 100, 2)
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
