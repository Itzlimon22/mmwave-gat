import torch
import numpy as np
import random
import logging

def set_seed(seed=42):
    """
    Standard JSDoc: Ensures reproducibility by pinning all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name="MIST-mmWave-GAT"):
    """
    Standard JSDoc: Initializes a professional logger for the engineering pipeline.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def early_return_check(data_path):
    """
    Logic Flow: Verifies data existence before execution to prevent runtime crashes.
    """
    import os
    if not os.path.exists(data_path):
        logging.error(f"Critical Path Missing: {data_path}")
        return False
    return True