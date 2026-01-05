import torch 
from src.model.transformer_block import TransformerBlock
from src.benchmark.memory import memory_forward

device = "cuda"
B, T, E, H = 2, 512, 512, 8

x = torch.randn(B, T, E, device=device, dtype=torch.float16)
model = TransformerBlock(E, H, "flash").to(device).half()

alloc_mb, reserved_mb = memory_forward(model, x, causal=True)
print(f"Allocated: {alloc_mb:.2f} MB | Reserved: {reserved_mb:.2f} MB")
