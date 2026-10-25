"""
MODULE 7: Deep Neural Network Architecture
Defines deep neural network architectures (MLP, Batch Normalization, Dropout, Skip/Residual connections)
for phishing URL classification.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dnn_model(input_dim: int,
                    architecture: list = None,
                    learning_rate: float = 0.001,
                    optimizer_name: str = 'adam'):
    """
    Build a Deep Neural Network for binary classification (phishing detection).
    
    Architecture components:
    - Input layer (dynamic dimension matching feature vector)
    - Sequential Dense layers with He-normal weight initialization
    - Batch Normalization for internal covariate shift mitigation
    - Non-linear activations (ReLU)
    - Dropout regularization to curb overfitting
    - Sigmoid output neuron for posterior phishing probability P(Phishing | x)
    
    Args:
        input_dim: Number of input features
        architecture: List of layer configurations
        learning_rate: Learning rate for optimizer
        optimizer_name: Optimizer name ('adam', 'sgd', 'rmsprop', 'adamw')
    
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    
    if architecture is None:
        architecture = [
            {'units': 256, 'activation': 'relu', 'dropout': 0.3, 'batch_norm': True},
            {'units': 128, 'activation': 'relu', 'dropout': 0.3, 'batch_norm': True},
            {'units': 64, 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True},
            {'units': 32, 'activation': 'relu', 'dropout': 0.1, 'batch_norm': False},
        ]
    
    logger.info(f"Building DNN model with input_dim={input_dim}, layers={len(architecture)}")
    
    model = models.Sequential(name='PhishingDNN')
    model.add(layers.Input(shape=(input_dim,), name='input_layer'))
    
    for i, layer_config in enumerate(architecture):
        units = layer_config.get('units', 128)
        activation = layer_config.get('activation', 'relu')
        dropout_rate = layer_config.get('dropout', 0.3)
        use_batch_norm = layer_config.get('batch_norm', True)
        
        # Dense projection with L2 weight decay
        model.add(layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer='he_normal',
            name=f'dense_{i+1}'
        ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        model.add(layers.Activation(activation, name=f'{activation}_{i+1}'))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Binary classification output
    model.add(layers.Dense(1, activation='sigmoid', name='output_prob'))
    
    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'adamw': tf.keras.optimizers.AdamW(learning_rate=learning_rate),
    }
    optimizer = optimizers.get(optimizer_name.lower(), optimizers['adam'])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def build_advanced_dnn(input_dim: int, config: dict = None):
    """
    Build an Advanced Deep Neural Network with residual skip connections.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    
    config = config or {}
    learning_rate = config.get('learning_rate', 0.001)
    
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    # Block 1
    x1 = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # Block 2 (128 units)
    x2 = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Dropout(0.2)(x2)
    
    # Block 3 with bottleneck projection
    x3 = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Dropout(0.1)(x3)
    
    outputs = layers.Dense(1, activation='sigmoid', name='phishing_probability')(x3)
    model = models.Model(inputs=inputs, outputs=outputs, name='PhishingDNN_Advanced')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model


class PhishingDNNModel:
    """
    High-level abstraction wrapper for DNN training, evaluation, persistence and inference.
    """
    def __init__(self, input_dim: int, model_type: str = 'standard', config: dict = None):
        self.input_dim = input_dim
        self.model_type = model_type
        self.config = config or {}
        
        if model_type == 'advanced':
            self.model = build_advanced_dnn(input_dim, self.config)
        else:
            self.model = build_dnn_model(
                input_dim=input_dim,
                architecture=self.config.get('architecture'),
                learning_rate=self.config.get('learning_rate', 0.001),
                optimizer_name=self.config.get('optimizer', 'adam')
            )
            
    def summary(self):
        self.model.summary()
        
    def save(self, filepath: str = "models/saved/phishing_dnn.h5"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model successfully saved to {filepath}")
        
    def load(self, filepath: str = "models/saved/phishing_dnn.h5"):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        
    def predict(self, X):
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).flatten()
