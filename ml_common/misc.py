import torch
import numpy as np
import random
import os
from torch.nn import Module
from torch.backends import cudnn


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def enable_gpu_benchmarking():
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(model: Module, path: str = "./"):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), path)


def load_model(model: Module, path: str = "./"):
    model.load_state_dict(torch.load(path))

import time

def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  

        execution_time = end_time - start_time
        
        return result, execution_time

    return wrapper


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  

        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        
        return result

    return wrapper