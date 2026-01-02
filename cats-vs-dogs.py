import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# ---------------------------
# 1) Config
# ---------------------------
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 5

# ---------------------------
# 2) Load dataset
# ---------------------------
(ds_train, ds_val), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

# ---------------------------
# 3) Preprocess (resize + normalize)
# ---------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_val   = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# ---------------------------
# 4) Data augmentation
# ---------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

# ---------------------------
# 5) Prepare dataset pipeline
# ---------------------------
ds_train = ds_train.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val   = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# 6) Build CNN model
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ---------------------------
# 7) Compile
# ---------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# 8) Train
# ---------------------------
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS
)

# ---------------------------
# 9) Evaluate
# ---------------------------
val_loss, val_acc = model.evaluate(ds_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# ---------------------------
# 10) Save
# ---------------------------
model.save("cats_vs_dogs_cnn.keras")
print("Saved model to cats_vs_dogs_cnn.keras")

# ---------------------------
# 11) Predict on a new image (optional)
# Put a file named 'test.jpg' in the same folder
# ---------------------------
def predict_image(path: str):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prob = float(model.predict(x, verbose=0)[0][0])
    label = "DOG" if prob >= 0.5 else "CAT"
    print(f"Prediction: {label} (dog_prob={prob:.3f})")

# Uncomment if you have a test image:
# predict_image("test.jpg")
