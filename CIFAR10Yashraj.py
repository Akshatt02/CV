import tensorflow as tf
from tensorflow.keras import layers,models,regularizers
import numpy as np

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.cifar100.load_data()
train_images,test_images = train_images/255.0,test_images/255.0

IMG_SIZE = train_images.shape[1]
NUM_CLASSES = len(np.unique(train_labels))

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

def build_cnn(input_shape,num_classes):
  initializer = tf.keras.initializers.HeNormal()
  regularizer = tf.keras.regularizers.l2(1e-4)

  inputs = layers.Input(shape=input_shape)
  x = data_augmentation(inputs)

  #block1 conv-conv-pool
  shortcut = layers.Conv2D(32,(1,1),padding='same')(x)

  x = layers.Conv2D(32,(3,3),padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(32,(3,3),padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer)(x)
  x = layers.BatchNormalization()(x)

  x = layers.add([shortcut,x])
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2,2))(x)

  shortcut = layers.Conv2D(64,(1,1),padding='same')(x)

  x = layers.Conv2D(64,(3,3),padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(64,(3,3),padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer)(x)
  x = layers.BatchNormalization()(x)

  x = layers.add([shortcut,x])
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2,2))(x)

  #final fc layer
  x = layers.Flatten()(x)
  x = layers.Dense(256,kernel_initializer=initializer)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.5)(x)

  output = layers.Dense(num_classes)(x)
  return models.Model(inputs,output)

model = build_cnn((IMG_SIZE,IMG_SIZE,3),NUM_CLASSES)

#compile
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_images,
    train_labels,
    epochs = 50,
    batch_size=128,
    validation_data=(test_images,test_labels)
)