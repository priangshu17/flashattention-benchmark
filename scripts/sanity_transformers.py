import torch 

from src.model.transformer_block import TransformerBlock

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    
    B, T, E = 2, 8, 64
    H = 4
    
    x_fp32 = torch.randn(B, T, E, device = device, dtype = torch.float32)
    
    block_naive = TransformerBlock(E, H, "naive").to(device)
    block_sdpa = TransformerBlock(E, H, "sdpa").to(device)
    
    y_n = block_naive(x_fp32, causal = True)
    y_s = block_sdpa(x_fp32, causal = True)
    
    x_fp16 = torch.randn(B, T, E, device = device, dtype = torch.float16)
    
    block_flash = TransformerBlock(E, H, "flash").to(device).half()
    
    
    y_f = block_flash(x_fp16, causal = True)
    
    print(y_n.shape, y_s.shape, y_f.shape)

    
    print("Sanity Check Passed!")
    
    
if __name__ == "__main__":
    main()