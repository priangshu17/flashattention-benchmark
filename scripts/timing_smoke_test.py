import torch 
from src.model.transformer_block import TransformerBlock
from src.benchmark.timing import time_forward

device = "cuda"
B, T, E, H = 4, 128, 512, 8

x = torch.randn(B, T, E, device = device, dtype = torch.float16)
model = TransformerBlock(E, H, "flash").to(device).half()

t_ms = time_forward(model, x, causal = True)
print(f"FlashAttention forward: {t_ms:.3f} ms")