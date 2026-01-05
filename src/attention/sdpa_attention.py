import torch 
import torch.nn.functional as F 

class SDPAAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention using PyTorch's fused SDPA.
    Serves as a strong baseline between naive and FlashAttention.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, causal: bool = False):
        """
        Args:
            q, k, v: Tensors of shape (B, H, T, D)
            causal: Whether to apply causal masking.
            
        Returns:
            output: Tensor of shape (B, H, T, D)
        """
        # Pytorch SDPA expects (B, H, T, D)
        output = F.scaled_dot_product_attention(
            q,
            k, 
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        
        return output 