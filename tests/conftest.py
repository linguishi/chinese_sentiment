import os
import tempfile
import shutil
import pytest
from pathlib import Path


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    The directory is automatically cleaned up after the test.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """
    Provide a mock configuration dictionary for testing.
    """
    return {
        "model_type": "test_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "vocab_size": 1000,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_classes": 2,
        "max_sequence_length": 100,
        "dropout_rate": 0.5
    }


@pytest.fixture
def sample_text_data():
    """
    Provide sample text data for testing text processing functions.
    """
    return [
        "This is a positive review. Great product!",
        "Terrible experience. Would not recommend.",
        "Average quality, nothing special.",
        "Excellent service and fast delivery!",
        "Disappointed with the purchase."
    ]


@pytest.fixture
def sample_labels():
    """
    Provide sample labels corresponding to sample_text_data.
    """
    return [1, 0, 0, 1, 0]  # 1 for positive, 0 for negative


@pytest.fixture
def vocab_dict():
    """
    Provide a sample vocabulary dictionary for testing.
    """
    return {
        "<PAD>": 0,
        "<UNK>": 1,
        "<START>": 2,
        "<END>": 3,
        "this": 4,
        "is": 5,
        "a": 6,
        "test": 7,
        "great": 8,
        "terrible": 9,
        "product": 10,
        "review": 11
    }


@pytest.fixture
def temp_file(temp_dir):
    """
    Create a temporary file and return its path.
    """
    def _temp_file(filename="test_file.txt", content=""):
        file_path = temp_dir / filename
        if content:
            file_path.write_text(content)
        return file_path
    return _temp_file


@pytest.fixture
def mock_model_weights(temp_dir):
    """
    Create a mock model weights file for testing model loading/saving.
    """
    weights_dir = temp_dir / "model_weights"
    weights_dir.mkdir()
    
    # Create dummy files to simulate model structure
    (weights_dir / "model.pb").write_text("dummy model data")
    (weights_dir / "variables").mkdir()
    (weights_dir / "variables" / "variables.index").write_text("dummy index")
    
    return weights_dir


@pytest.fixture
def env_setup(monkeypatch):
    """
    Set up environment variables for testing.
    """
    test_env = {
        "MODEL_PATH": "/tmp/test_model",
        "DATA_PATH": "/tmp/test_data",
        "LOG_LEVEL": "DEBUG",
        "BATCH_SIZE": "32"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    """
    Automatically reset environment to a clean state before each test.
    """
    # Store original environment
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    for key in list(os.environ.keys()):
        if key not in original_env:
            monkeypatch.delenv(key)
        else:
            monkeypatch.setenv(key, original_env[key])


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture and return log messages during tests.
    """
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog