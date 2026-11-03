# --------------------------------------------------
# Author Name : POTAKARI SMART
# USN         : 1AM23CI107
# Branch      : CSE - AIML
# College     : AMC Engineering College, Bengaluru
# Subject     : Deep Learning Laboratory
# --------------------------------------------------

"""
Program 7 (Pretrained Model / Transfer Learning)
Enable a pre-trained model (MobileNetV2) to classify a given image
dataset (CIFAR-10).
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert labels to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# 2. Resize images
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# 3. Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pre-trained layers
base_model.trainable = False

# 4. Create model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# 5. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 6. Train
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
)

# 7. Test model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# 8. Predict one image
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
prediction = model.predict(x_test[:1])
predicted_class = np.argmax(prediction)
print("Actual:", classes[y_test[0]])
print("Predicted:", classes[predicted_class])

# Print final training, validation and test accuracy
print("Training Accuracy:", history.history["accuracy"][-1] * 100, "%")
print("Validation Accuracy:", history.history["val_accuracy"][-1] * 100, "%")
print("Test Accuracy:", test_accuracy * 100, "%")

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()
