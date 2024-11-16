import torch

if torch.cuda.is_available():
    print("CUDA está disponible en tu sistema.")
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")