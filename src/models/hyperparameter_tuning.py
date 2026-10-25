"""
MODULE 10: Hyperparameter Optimization
Automated Bayesian Hyperparameter Search using Optuna for Deep Neural Networks.
Optimizes learning rate, layer depths, hidden units, dropout, and batch size.
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Optuna Bayesian Optimization framework for Phishing Detection Deep Neural Networks.
    """
    def __init__(self, X_train, y_train, X_val, y_val, n_trials: int = 20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.input_dim = X_train.shape[1]
        self.best_params = None
        self.study = None
        logger.info(f"HyperparameterOptimizer initialized with n_trials={n_trials}")
        
    def _create_model(self, trial):
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
        
        n_layers = trial.suggest_int('n_layers', 2, 4)
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adamw'])
        use_bn = trial.suggest_categorical('use_batch_norm', [True, False])
        
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        
        for i in range(n_layers):
            units = trial.suggest_categorical(f'units_l{i}', [32, 64, 128, 256])
            drop = trial.suggest_float(f'dropout_l{i}', 0.1, 0.4)
            model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal'))
            if use_bn:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(drop))
            
        model.add(layers.Dense(1, activation='sigmoid'))
        
        opt_dict = {
            'adam': tf.keras.optimizers.Adam(learning_rate=lr),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=lr),
            'adamw': tf.keras.optimizers.AdamW(learning_rate=lr)
        }
        model.compile(
            optimizer=opt_dict[optimizer_choice],
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def _objective(self, trial) -> float:
        import tensorflow as tf
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        model = self._create_model(trial)
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=4, mode='max', restore_best_weights=True
        )
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=25,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        best_val_auc = float(max(history.history['val_auc']))
        return best_val_auc

    def optimize(self) -> dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info("=" * 60)
        logger.info("MODULE 10: Hyperparameter Optimization")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        self.study = optuna.create_study(direction='maximize', study_name='phishing_opt')
        self.study.optimize(self._objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        best_val = self.study.best_value
        
        print(f"\nBest Trial: #{self.study.best_trial.number}")
        print(f"Optimal Validation AUC: {best_val:.4f}")
        print("Optimal Parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
            
        res = {
            'best_params': self.best_params,
            'best_auc': float(best_val),
            'n_trials': self.n_trials
        }
        os.makedirs("models/saved", exist_ok=True)
        with open("models/saved/optimization_results.json", 'w') as f:
            json.dump(res, f, indent=2)
            
        return res

    def train_best_model(self):
        import tensorflow as tf
        from src.models.deep_neural_network import build_dnn_model
        
        if not self.best_params:
            return None
            
        n_layers = self.best_params['n_layers']
        arch = []
        for i in range(n_layers):
            arch.append({
                'units': self.best_params[f'units_l{i}'],
                'activation': 'relu',
                'dropout': self.best_params[f'dropout_l{i}'],
                'batch_norm': self.best_params['use_batch_norm']
            })
            
        model = build_dnn_model(
            input_dim=self.input_dim,
            architecture=arch,
            learning_rate=self.best_params['learning_rate'],
            optimizer_name=self.best_params['optimizer']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('models/saved/best_optimized_dnn.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=self.best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        model.save("models/saved/best_optimized_dnn.h5")
        logger.info("Saved models/saved/best_optimized_dnn.h5")
        return model


def main():
    try:
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_val = np.load("data/processed/y_val.npy")
    except FileNotFoundError:
        print("Data files not found.")
        sys.exit(1)
        
    optimizer = HyperparameterOptimizer(X_train, y_train, X_val, y_val, n_trials=15)
    optimizer.optimize()
    optimizer.train_best_model()


if __name__ == "__main__":
    main()
