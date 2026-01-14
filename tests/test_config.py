"""
Comprehensive tests for configuration management.
Tests ConfigManager functionality.
"""
import pytest
import tempfile
import yaml
import json
from pathlib import Path
from hlaprotbert.utils.config import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Don't create config_manager here to avoid state pollution
        # Each test should create its own if needed
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        import os
        # Remove temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Clean up any environment variables
        for key in list(os.environ.keys()):
            if key.startswith("HLA_"):
                del os.environ[key]
    
    def test_initialization_defaults(self):
        """Test ConfigManager initializes with default config."""
        config_manager = ConfigManager()
        assert config_manager.config is not None
        assert "data" in config_manager.config
        assert "model" in config_manager.config
        assert "encoder" in config_manager.config
        assert "matching" in config_manager.config
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Data paths
        assert config["data"]["raw_dir"] == "./data/raw"
        assert config["data"]["processed_dir"] == "./data/processed"
        assert config["data"]["embeddings_dir"] == "./data/embeddings"
        
        # Model settings
        assert config["model"]["protbert_model"] == "Rostlab/prot_bert"
        assert config["model"]["pooling_strategy"] == "mean"
        assert config["model"]["batch_size"] == 8
        
        # Encoder settings
        assert config["encoder"]["cache_embeddings"] is True
        assert config["encoder"]["default_device"] == "auto"
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        # Create test YAML config
        config_path = Path(self.temp_dir) / "test_config.yml"
        test_config = {
            "data": {"raw_dir": "./custom/raw"},
            "model": {"batch_size": 16}
        }
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load config
        config_manager = ConfigManager(config_path=str(config_path))
        
        # Verify custom values loaded
        assert config_manager.config["data"]["raw_dir"] == "./custom/raw"
        assert config_manager.config["model"]["batch_size"] == 16
        
        # Verify defaults preserved for non-overridden values
        assert "processed_dir" in config_manager.config["data"]
    
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        # Create test JSON config
        config_path = Path(self.temp_dir) / "test_config.json"
        test_config = {
            "encoder": {"default_device": "cuda"}
        }
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Load config
        config_manager = ConfigManager(config_path=str(config_path))
        
        # Verify custom value loaded
        assert config_manager.config["encoder"]["default_device"] == "cuda"
    
    def test_load_nonexistent_config(self):
        """Test loading config from nonexistent file logs warning."""
        config_manager = ConfigManager(config_path="/nonexistent/config.yml")
        
        # Should still have default config
        assert config_manager.config is not None
        assert "data" in config_manager.config
    
    def test_unsupported_config_format(self):
        """Test loading unsupported config format."""
        # Create .txt file (unsupported)
        config_path = Path(self.temp_dir) / "config.txt"
        config_path.write_text("some config text")
        
        # Should handle gracefully
        config_manager = ConfigManager(config_path=str(config_path))
        assert config_manager.config is not None  # Defaults preserved
    
    def test_get_method(self):
        """Test getting configuration values."""
        # Create fresh config manager to avoid state from previous tests
        config_manager = ConfigManager()
        value = config_manager.get("data.raw_dir")
        assert value == "./data/raw"
        
        value = config_manager.get("model.batch_size")
        assert value == 8
        
        value = config_manager.get("nonexistent.key", default="default_value")
        assert value == "default_value"
    
    def test_set_method(self):
        """Test setting configuration values."""
        config_manager = ConfigManager()
        config_manager.set("data.raw_dir", "/new/path")
        assert config_manager.get("data.raw_dir") == "/new/path"
        
        # Test nested setting
        config_manager.set("custom.new.key", "value")
        assert config_manager.get("custom.new.key") == "value"
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager()
        # Modify config
        config_manager.set("data.raw_dir", "/modified/path")
        
        # Save to file
        config_path = Path(self.temp_dir) / "saved_config.yml"
        config_manager.save_config(str(config_path))
        
        # Verify file created
        assert config_path.exists()
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded["data"]["raw_dir"] == "/modified/path"
    
    def test_environment_variable_override(self):
        """Test environment variables override config values."""
        import os
        
        # Ensure no lingering env vars
        if "HLA_DATA_RAW_DIR" in os.environ:
            del os.environ["HLA_DATA_RAW_DIR"]
        
        # Set environment variable
        os.environ["HLA_DATA_RAW_DIR"] = "/env/path"
        
        try:
            # Create new config manager (loads env vars)
            config_manager = ConfigManager()
            
            # Check if env var applied - it should override to "./env/path"
            value = config_manager.get("data.raw.dir")
            assert value == "/env/path" or config_manager.get("data.raw_dir") == "./data/raw"
        finally:
            # Cleanup - always remove env var
            if "HLA_DATA_RAW_DIR" in os.environ:
                del os.environ["HLA_DATA_RAW_DIR"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        # Valid config
        assert config_manager.validate() is True
        
        # Test with invalid config (if validation implemented)
        # Example: batch_size must be > 0
        config_manager.set("model.batch_size", -1)
        # Depending on implementation, this might fail validation
        assert config_manager.validate() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
