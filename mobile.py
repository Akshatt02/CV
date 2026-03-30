import tensorflow as tf                      # TensorFlow for deep learning
import tensorflow_datasets as tfds           # Built-in datasets (cats vs dogs)
from tensorflow.keras import layers, models  # Keras API for building models

# ===================== PERFORMANCE SETTINGS =====================

AUTOTUNE = tf.data.AUTOTUNE  
# Automatically tunes data pipeline for best performance (CPU/GPU optimization)

# ===================== DATASET LOADING =====================

# Load dataset (80% train, 10% validation)
# as_supervised=True → returns (image, label)
(ds_train, ds_test), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]"],
    as_supervised=True,
    with_info=True
)

# ===================== PREPROCESSING =====================

def preprocess(img, label):
    img = tf.image.resize(img, (64, 64))        # Resize image to fixed size
    img = tf.cast(img, tf.float32) / 255.0      # Normalize pixel values (0–1)
    return img, label                           # Return processed image + label

# ===================== DATA PIPELINE =====================

train_dataset = (
    ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)  # Apply preprocessing in parallel
    .shuffle(1000)                                # Shuffle data → better generalization
    .batch(32)                                    # Process 32 images at a time
    .prefetch(AUTOTUNE)                           # Prepare next batch while training
)

test_dataset = (
    ds_test
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

# ===================== LOAD PRETRAINED MODEL =====================

# MobileNetV2 pretrained on ImageNet
# include_top=False → removes original classifier
# input_shape must match our resized images
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(64, 64, 3)
)

# Freeze base model → do not update pretrained weights
base_model.trainable = False

# ===================== BUILD CUSTOM MODEL =====================

model = models.Sequential([
    
    base_model,  # Feature extractor (learns edges, shapes, textures)

    layers.GlobalAveragePooling2D(),  
    # Converts feature maps → single vector (reduces parameters)

    layers.BatchNormalization(),  
    # Normalizes activations → faster & stable training

    layers.Dense(128, activation="relu"),  
    # Learns task-specific features (cat vs dog)

    layers.Dropout(0.5),  
    # Randomly drops neurons → prevents overfitting

    layers.Dense(1, activation="sigmoid")  
    # Output layer → binary classification (0 or 1)
])

# ===================== COMPILATION =====================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
    # Adam optimizer with tuned learning rate

    loss="binary_crossentropy",  
    # Loss for binary classification

    metrics=["accuracy"]  
    # Track accuracy during training
)

# Print model architecture (layers + parameters)
model.summary()

# ===================== CALLBACKS =====================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",        # Watch validation loss
        patience=2,               # Stop if no improvement for 2 epochs
        restore_best_weights=True # Restore best model
    ),
    
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",       # Monitor validation loss
        factor=0.3,              # Reduce learning rate
        patience=1               # If no improvement → reduce LR
    )
]

# ===================== TRAINING =====================

history = model.fit(
    train_dataset,               # Training data
    validation_data=test_dataset,# Validation data
    epochs=10,                   # Max epochs (EarlyStopping may stop early)
    callbacks=callbacks          # Use callbacks for optimization
)

# ===================== EVALUATION =====================

loss, acc = model.evaluate(test_dataset)  # Evaluate on unseen data

print(f"Final Test Accuracy: {acc:.4f}")  # Print final accuracy