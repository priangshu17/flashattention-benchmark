import torch 

from src.attention.naive_attention import NaiveAttention 
from src.attention.sdpa_attention import SDPAAttention
from src.attention.flash_attention import FlashAttention
from src.attention.attention_factory import UnifiedAttention
from src.model.transformer_block import TransformerBlock

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    B = 2  # Batch size
    H = 3  # Number of heads
    T = 5  # Sequence length
    D = 4  # Head dimension
    
    q = torch.randn(B, H, T, D, device = device)
    k = torch.randn(B, H, T, D, device = device)
    v = torch.randn(B, H, T, D, device = device)
    
    attention = NaiveAttention().to(device)
    
    sdpa = SDPAAttention().to(device)
    
    flash = FlashAttention().to(device)
    
    
    # Non-Causal Attention
    out_full = attention(q, k, v, causal = False)
    
    # Causal Attention
    out_causal = attention(q, k, v, causal = True)
    
    out_sdpa = sdpa(q, k, v, causal = False) 
    out_sdpa_causal = sdpa(q, k, v, causal = True)
    
    # FlashAttention requires fp16/bf16 on GPU
    q_f = q.half()
    k_f = k.half()
    v_f = v.half()
    
    out_flash = flash(q_f, k_f, v_f, causal = False)
    out_flash_causal = flash(q_f, k_f, v_f, causal = True)
    
    
    print("Input Shape:", q.shape)
    print("Output Shape (Non-Causal):", out_full.shape)
    print("Output Shape (Causal):", out_causal.shape)
    
    
    # Basic Sanity Assertions
    assert out_full.shape == (B, H, T, D)
    assert out_causal.shape == (B, H, T, D)
    
    
    # Check Numerical Sanity
    assert torch.isfinite(out_full).all()
    assert torch.isfinite(out_causal).all()
    
    
    # Causal Output should differ from non-causal
    difference = torch.mean(torch.abs(out_full - out_causal)).item()
    print(f"Mean absolute difference (causal vs non-causal): {difference:.6f}")
    
    # Numerical Comparison
    diff_full = torch.max(torch.abs(out_full - out_sdpa)).item()
    diff_causal = torch.max(torch.abs(out_causal - out_sdpa_causal)).item()
    print(f"Max abs diff (non-causal): {diff_full:.6e}")
    print(f"Max abs diff (causal): {diff_causal:.6e}")
    
    # Compare against naive (cast to same dtype)
    diff_flash = torch.max(torch.abs(out_full.half() - out_flash)).item()
    diff_flash_causal = torch.max(torch.abs(out_causal.half() - out_flash_causal)).item()
    
    print(f"Max abs diff Flash (non-causal): {diff_flash:.6e}")
    print(f"Max abs diff Flash (causal): {diff_flash_causal:.6e}")
    
    attn_naive = UnifiedAttention("naive").to(device)
    attn_sdpa = UnifiedAttention("sdpa").to(device)
    attn_flash = UnifiedAttention("flash").to(device)
    
    out_n = attn_naive(q, k, v, causal = True)
    out_s = attn_sdpa(q, k, v, causal = True)
    out_f = attn_flash(q, k, v, causal = True)
    
    print(out_n.shape, out_s.shape, out_f.shape)
    
    print("Sanity Check Passed!")
    
    
if __name__ == "__main__":
    main()