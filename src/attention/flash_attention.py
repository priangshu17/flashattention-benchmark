import torch 
from torch.nn import Module

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError as e:
    raise ImportError(
        "FlashAttention is not installed or CUDA is unavailable."
    ) from e 
    
    
class FlashAttention(Module):
    """
    FlashAttention v2 wrapper with the same API as NaiveAttention and SDPAAttention.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, causal: bool = False):
        """
        Args:
            q, k, v: tensors of shape (B, H, T, D)
            causal: whether to apply causal masking
            
        Returns:
            output: tensor of shape (B, H, T, D)
        """
        B, H, T, D = q.shape 
        
        # FlashAttention expects (B, T, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # flash_attn_func computes attention without materializing T x T
        out = flash_attn_func(
            q, 
            k, 
            v,
            dropout_p = 0.0,
            causal = causal
        )
        
        # Back to (B, H, T, D)
        out = out.transpose(1, 2)
        
        return out 