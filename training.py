import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 🔥 Paths
base_dir = r"C:\Users\maina\OneDrive\Desktop\R&ND\anomaly_module\disease_prediction\data"

rice_dir = os.path.join(base_dir, "Rice")
wheat_train_dir = os.path.join(base_dir, "wheat", "train")
wheat_test_dir = os.path.join(base_dir, "wheat", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# 🔁 DATA AUGMENTATION (CONTROLLED)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# =========================
# 🌾 LOAD DATA
# =========================
rice_train = train_datagen.flow_from_directory(
    rice_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

rice_val = val_datagen.flow_from_directory(
    rice_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

wheat_train = train_datagen.flow_from_directory(
    wheat_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

wheat_val = val_datagen.flow_from_directory(
    wheat_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

wheat_test = ImageDataGenerator(rescale=1./255).flow_from_directory(
    wheat_test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# 🧠 MODEL (TRANSFER LEARNING + DROPOUT)
# =========================
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)   # 🔥 strong regularization

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# 🛑 CALLBACKS (IMPORTANT)
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3
)

# =========================
# 🌾 TRAIN RICE MODEL
# =========================
print("\n🚀 Training Rice Model...")

rice_model = build_model(rice_train.num_classes)

rice_model.fit(
    rice_train,
    validation_data=rice_val,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

rice_model.save("rice_model.h5")
print("✅ Rice model saved")

# =========================
# 🌾 TRAIN WHEAT MODEL
# =========================
print("\n🚀 Training Wheat Model...")

wheat_model = build_model(wheat_train.num_classes)

wheat_model.fit(
    wheat_train,
    validation_data=wheat_val,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# 🧪 TEST
loss, acc = wheat_model.evaluate(wheat_test)
print(f"✅ Wheat Test Accuracy: {acc*100:.2f}%")

wheat_model.save("wheat_model.h5")
print("✅ Wheat model saved")