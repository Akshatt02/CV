import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load dataset
ds_train, ds_test = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]"],
    as_supervised=True,
)

# Preprocessing
def preprocess(img, label):
    img = tf.image.resize(img,(64,64)) / 255.0
    return img, label

train_dataset = ds_train.map(preprocess).batch(32)
test_dataset = ds_test.map(preprocess).batch(32)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(64,64,3)),
    layers.MaxPool2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(train_dataset, validation_data=test_dataset, epochs=1)

# Plot results
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.show()
