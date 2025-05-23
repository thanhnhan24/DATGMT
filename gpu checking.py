import torch
print(torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Số GPU:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Tên GPU:", torch.cuda.get_device_name(0))
    print("Thiết bị mặc định:", torch.cuda.current_device())
