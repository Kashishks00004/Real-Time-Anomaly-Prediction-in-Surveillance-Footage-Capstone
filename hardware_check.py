
import torch
import platform
import os
import cv2

print("=" * 50)
print("LAPTOP HARDWARE INFORMATION")
print("=" * 50)

print("Operating System:", platform.system())
print("Processor:", platform.processor())
print("CPU Cores:", os.cpu_count())

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CPU Threads:", torch.get_num_threads())
print("MKLDNN:", torch.backends.mkldnn.is_available())

print("OpenCV:", cv2.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Inference Device: CPU")

print("=" * 50)
