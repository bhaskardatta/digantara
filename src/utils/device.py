"""
Device selection utilities
"""

import torch
import logging


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get appropriate device for computation.
    
    Args:
        device_str: Device specification ('auto', 'cpu', 'cuda')
        
    Returns:
        PyTorch device
    """
    logger = logging.getLogger(__name__)
    
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def get_memory_info() -> dict:
    """
    Get memory usage information.
    
    Returns:
        Memory information dictionary
    """
    info = {}
    
    if torch.cuda.is_available():
        info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3    # GB
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    import psutil
    info["ram_usage"] = psutil.virtual_memory().percent
    info["ram_available"] = psutil.virtual_memory().available / 1024**3  # GB
    
    return info
