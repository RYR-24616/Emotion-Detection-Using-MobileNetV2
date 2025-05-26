# Emotion Detection using MobileNetV2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
import os

# Set GPU memory growth (optional for safety)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Paths
train_dir = r"train" #insert your path here
test_dir = r"test" #insert your path here

# Parameters
image_size = (48, 48)
batch_size = 32

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    label_mode='categorical',
    shuffle=True
)

val_ds = image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    label_mode='categorical',
    shuffle=True
)

# Class names
class_names = train_ds.class_names

# Preprocess
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Base model
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze

# Build model
inputs = Input(shape=(48, 48, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

# Save model (with full custom object compatibility)
save_path = r"emotion_mobilenet_model.keras" #insert your path here
model.save(save_path)
