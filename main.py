from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

from crop_pipeline import CropDiseasePipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = CropDiseasePipeline(
    crop_model_path="model/crop_classifier.pth",
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
        "Aphid","Black Rust","Blast","Brown Rust",
        "Common Root Rot","Fusarium Head Blight",
        "Healthy","Leaf Blight","Mildew","Mite",
        "Septoria","Smut","Stem fly","Tan spot","Yellow Rust"
    ]
)

# 🔥 NEW ENDPOINT (FIXED)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    # ✅ Load image correctly
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ✅ Call new function
    result = pipeline.run_inference_image(image)

    return result