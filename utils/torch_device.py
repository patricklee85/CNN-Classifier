import torch
gpu_id = 0
def get_device():
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")