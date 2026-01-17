import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import traceback

# Debugging info
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())

try:
    # Paths
    train_dir = "dataset/train"
    val_dir = "dataset/val"

    # Parameters
    img_height, img_width = 128, 128  # Reduced size for speed
    batch_size = 32                  # Reduced for stability
    AUTOTUNE = tf.data.AUTOTUNE

    print("\n[INFO] Loading datasets...")

    # Load datasets
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Get class names before prefetch
    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Classes found: {class_names}\n")

    # Optimize dataset pipeline
    train_ds = (raw_train_ds
                .cache()
                .shuffle(1000)
                .prefetch(buffer_size=AUTOTUNE))

    val_ds = (raw_val_ds
              .cache()
              .prefetch(buffer_size=AUTOTUNE))

    # Model
    print("[INFO] Building model...")
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\n[INFO] Starting training...\n")
    sys.stdout.flush()

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1
    )

    # Save the model
    model.save("food_waste_model.h5")
    print("\n[INFO] Training complete. Model saved as food_waste_model.h5")

except Exception as e:
    print("\n[ERROR] Training failed!")
    traceback.print_exc()
