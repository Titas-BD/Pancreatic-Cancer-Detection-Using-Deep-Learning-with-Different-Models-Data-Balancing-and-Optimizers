import os
import pydicom
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import gc

from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from imblearn.combine import SMOTEENN

# ----------------- DICOM Loader -----------------
def load_and_preprocess_dicom(dcm_path, img_size=(128, 128)):
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, "pixel_array"):
            return None

        img = dcm.pixel_array

        # middle slice if 3D
        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]

        img = img.astype(np.float32)

        if hasattr(dcm, "RescaleSlope"):
            img *= float(dcm.RescaleSlope)
        if hasattr(dcm, "RescaleIntercept"):
            img += float(dcm.RescaleIntercept)

        img_min, img_max = np.min(img), np.max(img)
        if img_max <= img_min:
            return None

        img = (img - img_min) / (img_max - img_min)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

        return np.expand_dims(img, axis=-1)
    except:
        return None


def load_dataset(base_path, max_images=18000, img_size=(128, 128)):
    images, labels, count = [], [], 0
    for root, _, files in os.walk(base_path):
        class_name = os.path.basename(root)
        for file in files:
            if count >= max_images:
                break
            if file.lower().endswith(".dcm"):
                img = load_and_preprocess_dicom(os.path.join(root, file), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                    count += 1
                    if count % 500 == 0:
                        gc.collect()
    return np.array(images, dtype=np.float16), np.array(labels)


# ----------------- ResNet18 -----------------
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    Add,
    GlobalAveragePooling2D,
    Dense,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def conv_block(x, filters, kernel_size=3, strides=1):
    y = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    return y


def residual_block(x, filters, downsample=False):
    strides = 2 if downsample else 1
    y = conv_block(x, filters, strides=strides)
    y = Conv2D(filters, 3, padding="same", use_bias=False)(y)
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
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


# ----------------- Main -----------------
def main():
    base_path = r"E:\Thesis\Thesis_test\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128, 128)
    max_images = 18000
    test_size = 0.2
    batch_size = 32
    epochs = 30
    random_state = 42

    # Load dataset
    images, labels = load_dataset(base_path, max_images, img_size)
    print(f"Loaded images: {images.shape[0]}")

    unique_labels = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_int = np.array([label_to_idx[label] for label in labels])

    # ------------------ SMOTE+ENN ------------------
    print("[INFO] Applying SMOTE+ENN on flattened image representation...")
    n_samples, h, w, c = images.shape
    X_flat = images.reshape(n_samples, -1)

    smote_enn = SMOTEENN(random_state=random_state)
    X_res_flat, y_res = smote_enn.fit_resample(X_flat, y_int)

    X_res = X_res_flat.reshape(-1, img_size[0], img_size[1], 1)
    y_int_resampled = y_res

    print(f"Original dataset size: {images.shape[0]}")
    print(f"Resampled dataset size (SMOTE+ENN): {X_res.shape[0]}")

    num_classes = len(unique_labels)
    y_resampled = to_categorical(y_int_resampled, num_classes=num_classes)

    # Train / test split on resampled dataset
    X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
        X_res,
        y_resampled,
        y_int_resampled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_int_resampled,
    )

    del images, labels, X_res, y_resampled
    gc.collect()

    # Build ResNet18 with RAdam optimizer
    model = build_resnet18((img_size[0], img_size[1], 1), num_classes)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Train model (no class weights; imbalance handled by SMOTE+ENN)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    # Evaluation
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = y_test_int

    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nTest Accuracy (ResNet18 + RAdam + SMOTE+ENN): {acc:.4f}")

    actual_classes_in_test = np.unique(y_true_classes)
    actual_labels_in_test = [unique_labels[i] for i in actual_classes_in_test]

    print("\nClassification Report (ResNet18 + RAdam + SMOTE+ENN):")
    print(
        classification_report(
            y_true_classes,
            y_pred_classes,
            labels=actual_classes_in_test,
            target_names=actual_labels_in_test,
        )
    )

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("ResNet18 Accuracy (RAdam + SMOTE+ENN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("ResNet18 Loss (RAdam + SMOTE+ENN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
