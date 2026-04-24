"""
Phase 2: ANN Model Design & Training
Multi-Modal Stroke Severity & Progression Prediction
Improved version with better regularization and data augmentation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    X_train = np.load("data/splits/X_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    X_test = np.load("data/splits/X_test.npy")

    y_sev_train = np.load("data/splits/y_sev_train.npy")
    y_sev_val = np.load("data/splits/y_sev_val.npy")
    y_sev_test = np.load("data/splits/y_sev_test.npy")

    y_prog_train = np.load("data/splits/y_prog_train.npy")
    y_prog_val = np.load("data/splits/y_prog_val.npy")
    y_prog_test = np.load("data/splits/y_prog_test.npy")

    return X_train, X_val, X_test, y_sev_train, y_sev_val, y_sev_test, y_prog_train, y_prog_val, y_prog_test

def augment_data(X, y_sev, y_prog, augmentation_factor=2):
    """
    Simple data augmentation by adding noise to features
    """
    augmented_X = [X]
    augmented_y_sev = [y_sev]
    augmented_y_prog = [y_prog]

    for _ in range(augmentation_factor - 1):
        # Add small random noise
        noise = np.random.normal(0, 0.05, X.shape)
        X_aug = X + noise
        augmented_X.append(X_aug)
        augmented_y_sev.append(y_sev)
        augmented_y_prog.append(y_prog)

    return np.concatenate(augmented_X), np.concatenate(augmented_y_sev), np.concatenate(augmented_y_prog)

def build_improved_model(clinical_input_dim):
    """
    Improved model with regularization and better architecture
    """
    # Clinical branch with regularization
    clinical_input = layers.Input(shape=(clinical_input_dim,), name="clinical_input")
    x = layers.Dropout(0.2)(clinical_input)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    clinical_features = layers.Dense(32, activation='relu')(x)

    # Fusion (simplified - no MRI for now)
    z = layers.Dense(64, activation='relu')(clinical_features)

    # Outputs
    sev_output = layers.Dense(3, activation='softmax', name="severity")(z)
    prog_output = layers.Dense(1, activation='linear', name="progression")(z)

    model = models.Model(inputs=clinical_input, outputs=[sev_output, prog_output])
    return model

def main():
    print("Phase 2: Training Improved ANN Model")
    print("=" * 50)

    # Load data
    X_train, X_val, X_test, y_sev_train, y_sev_val, y_sev_test, y_prog_train, y_prog_val, y_prog_test = load_data()

    print(f"Original dataset sizes:")
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # Apply data augmentation if dataset is small
    if len(X_train) < 100:
        print(f"Dataset small ({len(X_train)} samples), applying augmentation...")
        X_train, y_sev_train, y_prog_train = augment_data(X_train, y_sev_train, y_prog_train, augmentation_factor=3)
        print(f"After augmentation: {len(X_train)} samples")

    # Convert to categorical
    y_sev_train_cat = tf.keras.utils.to_categorical(y_sev_train, num_classes=3)
    y_sev_val_cat = tf.keras.utils.to_categorical(y_sev_val, num_classes=3)
    y_sev_test_cat = tf.keras.utils.to_categorical(y_sev_test, num_classes=3)

    # Build improved model
    clinical_input_dim = X_train.shape[1]
    model = build_improved_model(clinical_input_dim)

    print(f"Model input dimension: {clinical_input_dim}")
    model.summary()

    # Compile with improved optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss={"severity": "categorical_crossentropy", "progression": "mse"},
        loss_weights={"severity": 1.0, "progression": 0.5},
        metrics={"severity": "accuracy", "progression": "mae"}
    )

    # Callbacks for better training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_severity_accuracy',
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_severity_accuracy',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    # Train with more epochs and callbacks
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train,
        [y_sev_train_cat, y_prog_train],
        validation_data=(X_val, [y_sev_val_cat, y_prog_val]),
        epochs=200,  # More epochs with early stopping
        batch_size=min(32, len(X_train)),  # Adaptive batch size
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = model.evaluate(X_test, [y_sev_test_cat, y_prog_test], verbose=0)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Severity accuracy: {test_results[3]:.4f}")
    print(f"Progression MAE: {test_results[4]:.4f}")

    # Predictions for severity
    sev_pred, prog_pred = model.predict(X_test)
    y_sev_pred_classes = np.argmax(sev_pred, axis=1)

    # Classification report
    print("\n📈 Severity Classification Report:")
    target_names = ['Mild', 'Moderate', 'Severe']
    print(classification_report(y_sev_test, y_sev_pred_classes, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_sev_test, y_sev_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Stroke Severity Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/improved_confusion_matrix.png')
    plt.close()

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/improved_model.keras")
    print("✅ Improved model saved!")

    # Save training history
    history_dict = {
        'severity_accuracy': history.history.get('severity_accuracy', []),
        'val_severity_accuracy': history.history.get('val_severity_accuracy', []),
        'severity_loss': history.history.get('severity_loss', []),
        'val_severity_loss': history.history.get('val_severity_loss', []),
        'progression_mae': history.history.get('progression_mae', []),
        'val_progression_mae': history.history.get('val_progression_mae', []),
        'test_severity_accuracy': test_results[3],
        'test_progression_mae': test_results[4]
    }

    with open('results/improved_training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"📊 Results saved to results/ directory")
    print(f"🎯 Final test severity accuracy: {test_results[3]:.4f}")

    return model

if __name__ == "__main__":
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Train improved model
    model = main()