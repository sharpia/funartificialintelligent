"""
MobileNetV3-Large Transfer Learning Training Script
Trains a custom image classifier and exports to TFLite (float16 quantized)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV3 Transfer Learning Trainer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory (ImageFolder format: data_dir/class_name/image.jpg)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for model and TFLite files")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if not specified)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=20,
                        help="Additional epochs for fine-tuning")
    parser.add_argument("--initial_lr", type=float, default=1e-3,
                        help="Initial learning rate for head training")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--fine_tune_layers", type=int, default=30,
                        help="Number of last layers to unfreeze for fine-tuning")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split fraction")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate for classification head")
    return parser.parse_args()


def build_model(num_classes: int, image_size: int, dropout_rate: float) -> Model:
    """Build MobileNetV3-Large model with custom classification head."""
    input_shape = (image_size, image_size, 3)

    # Load pretrained MobileNetV3-Large backbone
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True  # built-in preprocessing (scales to [-1, 1])
    )

    # Freeze the base model initially
    base_model.trainable = False

    # Build classification head
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="mobilenetv3_classifier")
    return model, base_model


def create_data_generators(data_dir: str, image_size: int, batch_size: int, val_split: float):
    """Create training and validation data generators with augmentation."""

    train_datagen = ImageDataGenerator(
        validation_split=val_split,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode="nearest",
        rescale=None  # MobileNetV3 has built-in preprocessing
    )

    val_datagen = ImageDataGenerator(
        validation_split=val_split,
        rescale=None
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        interpolation="bilinear"
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        interpolation="bilinear"
    )

    return train_gen, val_gen


def unfreeze_top_layers(base_model: Model, num_layers: int):
    """Unfreeze the top N layers of the base model for fine-tuning."""
    base_model.trainable = True
    # Freeze all but the last `num_layers` layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    # Keep BatchNorm layers frozen to avoid corrupting learned statistics
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze {trainable_count} layers for fine-tuning")


def export_tflite_float16(model: Model, output_path: str):
    """Convert Keras model to TFLite with float16 quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to: {output_path} ({size_mb:.2f} MB)")
    return tflite_model


def verify_tflite_model(tflite_model: bytes, image_size: int, num_classes: int):
    """Run a quick sanity check on the exported TFLite model."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nTFLite Model Verification:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    # Run inference on a random image
    dummy_input = np.random.randint(0, 255, (1, image_size, image_size, 3), dtype=np.uint8).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    assert output.shape == (1, num_classes), f"Expected output shape (1, {num_classes}), got {output.shape}"
    assert abs(output.sum() - 1.0) < 1e-3, f"Softmax outputs don't sum to 1: {output.sum()}"
    print(f"  Verification: PASSED (output sums to {output.sum():.4f})")


def save_class_labels(class_indices: dict, output_dir: str):
    """Save class label mapping as JSON and plain text (for Android assets)."""
    # Invert the dict: index -> class_name
    labels = {str(v): k for k, v in class_indices.items()}

    json_path = os.path.join(output_dir, "labels.json")
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=2)

    # Plain text labels file (one per line, sorted by index)
    txt_path = os.path.join(output_dir, "labels.txt")
    sorted_labels = [labels[str(i)] for i in range(len(labels))]
    with open(txt_path, "w") as f:
        f.write("\n".join(sorted_labels))

    print(f"Labels saved to: {json_path} and {txt_path}")
    return sorted_labels


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup GPU memory growth to avoid OOM
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(gpus)}")

    # Create data generators
    print(f"\nLoading dataset from: {args.data_dir}")
    train_gen, val_gen = create_data_generators(
        args.data_dir, args.image_size, args.batch_size, args.val_split
    )

    num_classes = args.num_classes or train_gen.num_classes
    class_names = save_class_labels(train_gen.class_indices, args.output_dir)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # Build model
    print("\nBuilding MobileNetV3-Large model...")
    model, base_model = build_model(num_classes, args.image_size, args.dropout_rate)
    model.summary()

    # Phase 1: Train classification head with frozen backbone
    print("\n=== Phase 1: Training classification head ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_phase1.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join(args.output_dir, "logs", "phase1"))
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_phase1,
        verbose=1
    )

    print(f"Phase 1 best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # Phase 2: Fine-tune top layers of backbone
    print(f"\n=== Phase 2: Fine-tuning top {args.fine_tune_layers} backbone layers ===")
    unfreeze_top_layers(base_model, args.fine_tune_layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    callbacks_phase2 = [
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-8, verbose=1),
        TensorBoard(log_dir=os.path.join(args.output_dir, "logs", "phase2"))
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks_phase2,
        verbose=1
    )

    print(f"Phase 2 best val_accuracy: {max(history2.history['val_accuracy']):.4f}")

    # Save full Keras model
    keras_path = os.path.join(args.output_dir, "mobilenetv3_classifier.h5")
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")

    # Export TFLite (float16 quantized)
    print("\nExporting TFLite model (float16 quantized)...")
    tflite_path = os.path.join(args.output_dir, "mobilenetv3_classifier.tflite")
    tflite_model = export_tflite_float16(model, tflite_path)

    # Verify the exported model
    verify_tflite_model(tflite_model, args.image_size, num_classes)

    # Save training config summary
    config = {
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": args.image_size,
        "model_architecture": "MobileNetV3Large",
        "tflite_quantization": "float16",
        "phase1_best_val_acc": float(max(history1.history["val_accuracy"])),
        "phase2_best_val_acc": float(max(history2.history["val_accuracy"])),
        "training_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining config saved to: {config_path}")
    print("\nDone! Copy the following files to your Android app's assets/ folder:")
    print(f"  {tflite_path}")
    print(f"  {os.path.join(args.output_dir, 'labels.txt')}")


if __name__ == "__main__":
    main()
