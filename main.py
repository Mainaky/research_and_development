from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

from crop_pipeline import CropDiseasePipeline

# 🔥 Initialize app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],   # 🔥 VERY IMPORTANT
    allow_headers=["*"],   # 🔥 VERY IMPORTANT
)

# 🔥 Load pipeline
pipeline = CropDiseasePipeline(
    rice_model_path="model/rice_model.h5",
    wheat_model_path="model/wheat_model.h5",
    rice_labels=[
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    ],
    wheat_labels=[
        "Aphid", "Black Rust", "Blast", "Brown Rust",
        "Common Root Rot", "Fusarium Head Blight",
        "Healthy", "Leaf Blight", "Mildew", "Mite",
        "Septoria", "Smut", "Stem fly", "Tan spot", "Yellow Rust"
    ]
)

# 🔥 SIMPLE SEVERITY FUNCTION (NO TRAINING REQUIRED)
def calculate_severity(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Convert to grayscale
    gray = np.mean(img_array, axis=2)

    # Threshold (detect dark/infected areas)
    infected_pixels = np.sum(gray < 100)
    total_pixels = gray.size

    severity = infected_pixels / total_pixels

    return round(float(severity), 4)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        print("✅ File received")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("✅ Image loaded")

        result = pipeline.run_inference_image(image)
        print("✅ Pipeline done")

        severity = calculate_severity(image)
        print("✅ Severity calculated")

        result["severity"] = severity

        return result

    except Exception as e:
        print("❌ ERROR:", str(e))
        import traceback
        traceback.print_exc()

        return {"error": str(e)}