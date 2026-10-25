"""
MODULE 8: Model Training
Trains the Deep Neural Network model with callbacks, early stopping, LR scheduling, and tensor monitoring.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DNNTrainer:
    """
    Handles training of the Deep Neural Network with enterprise-grade callbacks:
    - EarlyStopping with patience
    - ModelCheckpoint for best validation AUC
    - ReduceLROnPlateau adaptive learning rate
    - Loss & metric history visualization
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
        }
        self.history = None
        self.model = None
        logger.info("DNNTrainer initialized")
    
    def _get_callbacks(self) -> list:
        import tensorflow as tf
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/saved/best_dnn_model.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'logs/training_log.csv',
                separator=',',
                append=False
            )
        ]
        return callbacks
    
    def train(self, model, X_train, y_train, X_val, y_val) -> dict:
        """
        Train the DNN model with stratified class weighting and monitoring.
        """
        logger.info("=" * 60)
        logger.info("MODULE 8: Model Training (Deep Neural Network)")
        logger.info("=" * 60)
        
        self.model = model
        os.makedirs("models/saved", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        batch_size = self.config.get('batch_size', 64)
        epochs = self.config.get('epochs', 100)
        
        print("\n" + "=" * 60)
        print("DNN TRAINING EXECUTION")
        print("=" * 60)
        print(f"Batch size: {batch_size}")
        print(f"Target Epochs: {epochs}")
        print(f"Train Set Size: {len(X_train)}")
        print(f"Val Set Size: {len(X_val)}")
        print(f"Feature Dimension: {X_train.shape[1]}")
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Class weighting: {class_weight_dict}")
        
        callbacks = self._get_callbacks()
        
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save final weights and full architecture
        model.save("models/saved/phishing_dnn_final.h5")
        logger.info("Final model saved to models/saved/phishing_dnn_final.h5")
        
        history_dict = {k: [float(v) for v in values] for k, values in self.history.history.items()}
        with open("models/saved/training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.plot_training_history()
        self._print_training_summary()
        
        return self.history.history
    
    def plot_training_history(self):
        """Plot loss, accuracy, and ROC-AUC curves."""
        if self.history is None:
            return
        
        history = self.history.history
        os.makedirs("reports/figures", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', color='#2980b9', lw=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#e74c3c', lw=2, linestyle='--')
        axes[0, 0].set_title('Cross-Entropy Loss Curve', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy', color='#27ae60', lw=2)
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='#8e44ad', lw=2, linestyle='--')
        axes[0, 1].set_title('Classification Accuracy Curve', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history and 'val_auc' in history:
            axes[1, 0].plot(history['auc'], label='Train AUC', color='#f39c12', lw=2)
            axes[1, 0].plot(history['val_auc'], label='Val AUC', color='#d35400', lw=2, linestyle='--')
            axes[1, 0].set_title('Area Under ROC Curve (AUC)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        if 'precision' in history and 'recall' in history:
            axes[1, 1].plot(history['precision'], label='Train Precision', color='#16a085', lw=1.5)
            axes[1, 1].plot(history['recall'], label='Train Recall', color='#c0392b', lw=1.5)
            if 'val_precision' in history:
                axes[1, 1].plot(history['val_precision'], label='Val Precision', color='#1abc9c', lw=1.5, linestyle='--')
            if 'val_recall' in history:
                axes[1, 1].plot(history['val_recall'], label='Val Recall', color='#e67e22', lw=1.5, linestyle='--')
            axes[1, 1].set_title('Precision & Recall Evolution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Metric')
            axes[1, 1].legend(fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Deep Neural Network Training Metrics History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved reports/figures/training_history.png")
    
    def _print_training_summary(self):
        history = self.history.history
        epochs_done = len(history['loss'])
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        print("\n" + "=" * 60)
        print("TRAINING PROCESS SUMMARY")
        print("=" * 60)
        print(f"Total epochs run: {epochs_done}")
        print(f"Optimal epoch: #{best_epoch}")
        print(f"Best Val Loss: {min(history['val_loss']):.4f}")
        print(f"Best Val Accuracy: {max(history['val_accuracy']):.4f}")
        if 'val_auc' in history:
            print(f"Best Val AUC: {max(history['val_auc']):.4f}")


def main():
    from src.models.deep_neural_network import PhishingDNNModel
    try:
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_val = np.load("data/processed/y_val.npy")
    except FileNotFoundError:
        print("Preprocessed data not found. Please run feature engineering first.")
        sys.exit(1)
        
    dnn = PhishingDNNModel(input_dim=X_train.shape[1], model_type='standard')
    trainer = DNNTrainer(config={'batch_size': 64, 'epochs': 40, 'learning_rate': 0.001, 'early_stopping_patience': 8})
    trainer.train(dnn.model, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
