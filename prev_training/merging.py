import os
import shutil

base_dir = r"C:\Users\maina\OneDrive\Desktop\R&ND\anomaly_module\disease_prediction\data"

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# 👉 New merged directory
merged_dir = os.path.join(base_dir, "merged_train")

# Create merged directory if not exists
os.makedirs(merged_dir, exist_ok=True)

# Get all class folders (from train)
classes = os.listdir(train_dir)

for cls in classes:
    train_class_path = os.path.join(train_dir, cls)
    valid_class_path = os.path.join(valid_dir, cls)
    
    merged_class_path = os.path.join(merged_dir, cls)
    os.makedirs(merged_class_path, exist_ok=True)
    
    # Copy from train
    if os.path.exists(train_class_path):
        for file in os.listdir(train_class_path):
            src = os.path.join(train_class_path, file)
            dst = os.path.join(merged_class_path, file)
            shutil.copy2(src, dst)
    
    # Copy from valid
    if os.path.exists(valid_class_path):
        for file in os.listdir(valid_class_path):
            src = os.path.join(valid_class_path, file)
            dst = os.path.join(merged_class_path, file)
            
            # Handle duplicate filenames
            if os.path.exists(dst):
                name, ext = os.path.splitext(file)
                dst = os.path.join(merged_class_path, f"{name}_valid{ext}")
            
            shutil.copy2(src, dst)

print("✅ Train and Valid merged successfully into 'merged_train'")