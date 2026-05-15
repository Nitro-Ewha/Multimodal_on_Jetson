import torch
import torch.nn as nn

device = "cuda"

# ✅ 직접 만든 Transformer block (ONNX-friendly)
class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)

        self.ffn = nn.Sequential(
            nn.Linear(hidden, 3072),
            nn.ReLU(),
            nn.Linear(3072, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        # MHA
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


model = SimpleTransformerBlock().to(device)
model.eval()

B, SEQ, HID = 8, 1024, 768
x = torch.randn(B, SEQ, HID).to(device)

torch.onnx.export(
    model,
    x,
    "transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 1: "seq"},
        "output": {0: "batch", 1: "seq"}
    },
    opset_version=17
)

print("✅ ONNX export success")