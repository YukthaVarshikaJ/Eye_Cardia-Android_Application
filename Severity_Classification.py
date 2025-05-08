import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


input_folder = r"C:\Users\Yuktha Varshika\Music\Diabetic Retinopathy"
classes = [cls for cls in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, cls))]
print("Detected Classes:", classes)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return enhanced_image

def gamma_correction(image, gamma=1.5):
  
    image = image.astype(np.float32) / 255.0  # Normalize to 0-1
    corrected_image = np.power(image, gamma)  # Apply gamma transformation
    corrected_image = np.clip(corrected_image * 255, 0, 255).astype(np.uint8)  # Scale back
    return corrected_image


def resize_and_normalize(image, target_size=(224, 224)):
    
    
 
    h, w, _ = image.shape
    scale = min(target_size[0] / w, target_size[1] / h)  # Scale to maintain aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    normalized_image = canvas.astype(np.float32) / 255.0
    return normalized_image
fig, axes = plt.subplots(nrows=len(classes), ncols=2, figsize=(10, len(classes) * 4))


for idx, cls in enumerate(classes):
    class_path = os.path.join(input_folder, cls)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        image_path = os.path.join(class_path, image_files[0])  # Take the first image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        clahe_image = apply_clahe(image)
        gamma_corrected_image = gamma_correction(clahe_image, gamma=1.5)
        final_image = resize_and_normalize(gamma_corrected_image)
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Original - {cls}")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(final_image)
        axes[idx, 1].set_title(f"Processed - {cls}")
        axes[idx, 1].axis("off")
plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Concatenate, Input
)
from tensorflow.keras.models import Model

input_layer = Input(shape=(224, 224, 3))
resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
resnet.trainable = False  
resnet_out = GlobalAveragePooling2D()(resnet.output)
cnn = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)
cnn = Conv2D(64, (3,3), activation='relu', padding='same')(cnn)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)
cnn = Conv2D(128, (3,3), activation='relu', padding='same')(cnn)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)
cnn = Flatten()(cnn)  
combined_features = Concatenate()([resnet_out, cnn])
x = Dense(256, activation='relu')(combined_features)
x = Dropout(0.3)(x)
output_layer = Dense(4, activation='softmax', name='disease_output')(x)  # Assuming 5 classes
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
train_image_paths = []
train_labels = []
for cls in classes:
    class_path = os.path.join(input_folder, cls)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        train_image_paths.append(os.path.join(class_path, img_file))
        train_labels.append(cls)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_categorical = to_categorical(train_labels_encoded, num_classes=len(classes))

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, input_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        images, labels = self.__data_generation(batch_image_paths, batch_labels)
        return images, labels

    def __data_generation(self, batch_image_paths, batch_labels):
        batch_images = []
        for img_path in batch_image_paths:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            image = cv2.resize(image, self.input_size)
            image = image / 255.0  # Normalize to [0,1]

            batch_images.append(image)

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32)
    )
).prefetch(buffer_size=tf.data.AUTOTUNE)

train_generator = CustomDataGenerator(train_image_paths, train_labels_categorical, batch_size=16)
history=model.fit(
    train_generator, 
    epochs=30,
    steps_per_epoch=len(train_generator),
    verbose=1)

import os
save_directory = r"C:\Users\Yuktha Varshika\Music\Models"


os.makedirs(save_directory, exist_ok=True)

model_path = os.path.join(save_directory, "Retinal_Severity_Classification.h5")
model.save(model_path)

print(f"✅ Model saved successfully at: {model_path}")
