import torch 
from torch import nn 

from src.attention.naive_attention import NaiveAttention
from src.attention.sdpa_attention import SDPAAttention
from src.attention.flash_attention import FlashAttention


class UnifiedAttention(nn.Module):
    """
    Unified interface over multiple attention backends.
    
    Supported backends:
        - "naive"
        - "sdpa"
        - "flash"
    """
    
    def __init__(self, backend: str):
        super().__init__()
        
        backend = backend.lower()
        
        if backend == "naive":
            self.attn = NaiveAttention()
            self.requires_fp16 = False 
            
        elif backend == "sdpa":
            self.attn = SDPAAttention()
            self.requires_fp16 = False 
            
        elif backend == "flash":
            self.attn = FlashAttention()
            self.requires_fp16 = True 
            
        else:
            raise ValueError(
                f"Unknown attention backend '{backend}'."
                "Choose from ['naive', 'sdpa', 'flash']."
            )
            
        self.backend = backend 
        
    def forward(self, q, k, v, causal: bool = False):
        """
        Args:
            q, k, v: (B, H, T, D)
            causal: whether to apply causal masking
        """
        if self.requires_fp16:
            q = q.half()
            k = k.half()
            v = v.half()

            
        return self.attn(q, k, v, causal = causal)