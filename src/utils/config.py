"""
Configuration Handler
-------------------
Manages configuration for the HLA-ProtBERT system.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for HLA-ProtBERT system
    
    Handles loading and saving configuration from JSON/YAML files,
    providing default values, and managing environment variables.
    """
    
    DEFAULT_CONFIG = {
        "data": {
            "raw_dir": "./data/raw",
            "processed_dir": "./data/processed",
            "embeddings_dir": "./data/embeddings"
        },
        "model": {
            "protbert_model": "Rostlab/prot_bert",
            "pooling_strategy": "mean",
            "use_peptide_binding_region": True,
            "batch_size": 8
        },
        "encoder": {
            "cache_embeddings": True,
            "default_device": "auto"  # "auto", "cpu", or "cuda"
        },
        "matching": {
            "loci": ["A", "B", "C", "DRB1", "DQB1", "DPB1"],
            "similarity_threshold": 0.9
        },
        "prediction": {
            "default_model_type": "mlp",
            "default_clinical_variables": {
                "transplant": ["recipient_age", "donor_age", "disease", "donor_type"],
                "gvhd": ["recipient_age", "donor_age", "gender_match", "conditioning"]
            }
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path:
            self.load_config(config_path)
            
        # Override with environment variables
        self._load_from_env()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
            
        try:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return
                
            # Update configuration with loaded values
            self._update_nested_dict(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        
        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(exist_ok=True, parents=True)
            
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return
                
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., "model.batch_size")
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., "model.batch_size")
            value: Value to set
        """
        keys = key_path.split('.')
        
        if not keys:
            return
            
        config = self.config
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                # Convert non-dict to dict, overwriting previous value
                config[key] = {}
                
            config = config[key]
            
        config[keys[-1]] = value
    
    def _update_nested_dict(self, target: Dict, source: Dict) -> None:
        """Update nested dictionary with values from another dictionary
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(target[key], value)
            else:
                # Update/add value
                target[key] = value
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables
        
        Environment variables should follow the pattern:
        HLA_SECTION_KEY=value
        
        For example:
        HLA_MODEL_BATCH_SIZE=16
        """
        prefix = "HLA_"
        
        for name, value in os.environ.items():
            if name.startswith(prefix):
                # Remove prefix
                key = name[len(prefix):]
                
                # Convert to lowercase and replace underscores with dots
                key_path = key.lower().replace('_', '.')
                
                # Attempt to parse as number or boolean
                try:
                    if value.lower() in ["true", "yes", "y", "1"]:
                        value = True
                    elif value.lower() in ["false", "no", "n", "0"]:
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value)
                except (ValueError, AttributeError):
                    pass
                    
                # Set in config
                self.set(key_path, value)
                logger.debug(f"Config set from environment: {key_path}={value}")
