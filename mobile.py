import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load dataset
ds_train, ds_test = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]"],
    as_supervised=True
)

# Preprocess images
def preprocess(img, label):
    img = tf.image.resize(img,(64,64))/255.0
    return img, label

train_dataset = ds_train.map(preprocess).batch(32)
test_dataset = ds_test.map(preprocess).batch(32)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(64,64,3)
)

base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.summary()

history = model.fit(train_dataset,validation_data=test_dataset,epochs=1)

print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
