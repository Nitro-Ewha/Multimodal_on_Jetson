# run_transformer_block.py
import torch
import torch.nn as nn

device = "cuda"

layer = nn.TransformerEncoderLayer(
    d_model=768,
    nhead=12,
    dim_feedforward=3072,
    batch_first=True
).to(device)

layer.eval()

# 다양한 config 실험하려면 여기만 바꾸면 됨
B = 8
SEQ = 1024
HID = 768

x = torch.randn(B, SEQ, HID).to(device)

# warmup
for _ in range(10):
    with torch.no_grad():
        layer(x)

torch.cuda.synchronize()

# 실제 실행
for _ in range(50):
    with torch.no_grad():
        layer(x)

torch.cuda.synchronize()