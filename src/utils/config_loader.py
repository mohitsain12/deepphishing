"""
Configuration loader utility.
Loads and provides access to project configuration from YAML files.
"""

import os
import yaml
from typing import Any, Dict, Optional


class ConfigLoader:
    """Loads and manages project configuration."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        if self._config is None:
            if config_path is None:
                # Find config relative to project root
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                config_path = os.path.join(project_root, "config", "config.yaml")
            
            self.config_path = config_path
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.deep_learning.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def config(self) -> Dict:
        """Return the full configuration dictionary."""
        return self._config
    
    def reload(self):
        """Reload configuration from file."""
        self._config = None
        self._load_config()


def load_config(config_path: str = None) -> ConfigLoader:
    """
    Convenience function to load and return configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)
