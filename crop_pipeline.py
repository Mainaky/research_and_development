import os
import torch
import torch.nn as nn
from torchvision import transforms, models as torch_models
from PIL import Image
import numpy as np
import tensorflow as tf

class CropDiseasePipeline:
    def __init__(self, rice_model_path, wheat_model_path, rice_labels, wheat_labels):
        print("--- Initializing Pipeline ---")

        self.rice_model = tf.keras.models.load_model(rice_model_path)
        self.wheat_model = tf.keras.models.load_model(wheat_model_path)

        self.rice_labels = rice_labels
        self.wheat_labels = wheat_labels

        print("✅ Models loaded successfully")

    def _preprocess(self, image):
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def _predict_crop(self, image):
        img_tensor = self.py_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.pytorch_model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        return self.crop_classes[pred.item()], conf.item()

    def _predict_disease(self, image, crop_type):
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if crop_type.lower() == 'rice':
            preds = self.rice_model.predict(img_array, verbose=0)
            return self.rice_labels[np.argmax(preds)], np.max(preds)

        elif crop_type.lower() == 'wheat':
            preds = self.wheat_model.predict(img_array, verbose=0)
            return self.wheat_labels[np.argmax(preds)], np.max(preds)

        return None, 0
    
    def run_inference_image(self, image):
        img_array = self._preprocess(image)

        rice_preds = self.rice_model.predict(img_array, verbose=0)
        wheat_preds = self.wheat_model.predict(img_array, verbose=0)

        rice_conf = np.max(rice_preds)
        wheat_conf = np.max(wheat_preds)

        if rice_conf > wheat_conf:
            crop_name = "Rice"
            disease_name = self.rice_labels[np.argmax(rice_preds)]
            disease_conf = rice_conf
        else:
            crop_name = "Wheat"
            disease_name = self.wheat_labels[np.argmax(wheat_preds)]
            disease_conf = wheat_conf

        return {
            "status": "Success",
            "crop": crop_name,
            "disease": disease_name,
            "disease_confidence": f"{disease_conf:.2%}"
        }

    def run_inference(self, image_path):
        if not os.path.exists(image_path):
            return {"error": "File not found"}

        image = Image.open(image_path).convert("RGB")

        crop_name, crop_conf = self._predict_crop(image)

        print("Crop:", crop_name)
        print("Confidence:", crop_conf)

        if crop_name.lower() == "random":
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            rice_preds = self.rice_model.predict(img_array, verbose=0)
            wheat_preds = self.wheat_model.predict(img_array, verbose=0)

            if np.max(rice_preds) > np.max(wheat_preds):
                crop_name = "Rice"
                disease_name = self.rice_labels[np.argmax(rice_preds)]
                disease_conf = np.max(rice_preds)
            else:
                crop_name = "Wheat"
                disease_name = self.wheat_labels[np.argmax(wheat_preds)]
                disease_conf = np.max(wheat_preds)

        else:
            disease_name, disease_conf = self._predict_disease(image, crop_name)

        return {
            "status": "Success",
            "crop": crop_name,
            "crop_confidence": f"{crop_conf:.2%}",
            "disease": disease_name,
            "disease_confidence": f"{disease_conf:.2%}"
        }




# ==========================================
# 🛠️ HOW TO USE IT
# ==========================================
if __name__ == "__main__":

    RICE_DISEASES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]
    WHEAT_DISEASES = [
    "Aphid",
    "Black Rust",
    "Blast",
    "Brown Rust",
    "Common Root Rot",
    "Fusarium Head Blight",
    "Healthy",
    "Leaf Blight",
    "Mildew",
    "Mite",
    "Septoria",
    "Smut",
    "Stem fly",
    "Tan spot",
    "Yellow Rust"
]

    pipeline = CropDiseasePipeline(
        crop_model_path="model/crop_classifier.pth",
        rice_model_path="model/rice_model.h5",
        wheat_model_path="model/wheat_model.h5",
        rice_labels=RICE_DISEASES,
        wheat_labels=WHEAT_DISEASES
    )

    # Test the pipeline
    test_image = r"C:\Users\maina\OneDrive\Desktop\R&ND\anomaly_module\disease_prediction\data\Rice\Brown Spot\aug_0_17.jpg" # Change to your image path
    result = pipeline.run_inference(test_image)
    
    print("\n--- FINAL RESULT ---")
    print(result)