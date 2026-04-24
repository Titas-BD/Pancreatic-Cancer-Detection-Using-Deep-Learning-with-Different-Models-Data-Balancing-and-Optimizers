import os
import pydicom
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import gc

# ----------------- DICOM Loader -----------------
def load_and_preprocess_dicom(dcm_path, img_size=(128,128)):
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, 'pixel_array'):
            return None
        img = dcm.pixel_array
        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]  # middle slice
        img = img.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope'):
            img *= float(dcm.RescaleSlope)
        if hasattr(dcm, 'RescaleIntercept'):
            img += float(dcm.RescaleIntercept)
        img_min, img_max = np.min(img), np.max(img)
        if img_max <= img_min:
            return None
        img = (img - img_min) / (img_max - img_min)
        img = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, axis=-1)
    except:
        return None

def load_dataset(base_path, max_images=18000, img_size=(128,128)):
    images, labels, count = [], [], 0
    for root, _, files in os.walk(base_path):
        class_name = os.path.basename(root)
        for file in files:
            if count >= max_images:
                break
            if file.lower().endswith('.dcm'):
                img = load_and_preprocess_dicom(os.path.join(root, file), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                    count += 1
                    if count % 500 == 0:
                        gc.collect()
    return np.array(images, dtype=np.float16), np.array(labels)

# ----------------- ResNet18 -----------------
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=3, strides=1):
    y = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    return y

def residual_block(x, filters, downsample=False):
    strides = 2 if downsample else 1
    y = conv_block(x, filters, strides=strides)
    y = Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    if downsample or x.shape[-1] != filters:
        x = Conv2D(filters, 1, strides=strides, use_bias=False)(x)
        x = BatchNormalization()(x)
    y = Add()([x, y])
    y = ReLU()(y)
    return y

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 64, 7, strides=2)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model  # No optimizer, untrained

# ----------------- Main -----------------
def main():
    base_path = r"E:\Thesis\Thesis_test\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128,128)
    max_images = 18000
    test_size = 0.2

    # Load dataset
    images, labels = load_dataset(base_path, max_images, img_size)
    unique_labels = np.unique(labels)
    label_to_idx = {label:i for i,label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    y = tf.keras.utils.to_categorical(y)

    # Split dataset (no data balancing)
    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=test_size, random_state=42)

    del images, labels
    gc.collect()

    # Build untrained ResNet18
    model = build_resnet18((img_size[0], img_size[1],1), len(unique_labels))
    model.summary()

    # ---------------- Evaluation ----------------
    y_pred_classes = np.argmax(model.predict(X_test, batch_size=16), axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nTest Accuracy (random untrained model): {acc:.4f}")

    # Handle only classes present in test set
    actual_classes_in_test = np.unique(y_true_classes)
    actual_labels_in_test = [unique_labels[i] for i in actual_classes_in_test]

    print("\nClassification Report (random untrained model):")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=actual_classes_in_test,
        target_names=actual_labels_in_test
    ))

if __name__ == "__main__":
    main()
