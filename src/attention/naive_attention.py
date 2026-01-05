import torch 
import math 

class NaiveAttention(torch.nn.Module):
    """
    Reference implementation of scaled dot-product attention.
    
    This is intentionally slow and explicit.
    It serves as a correctness baseline for benchmarking.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, causal: bool = False):
        """
        Args:
            q, k, v: Tensors of shape (B, H, T, D)
            causal: Whether to apply causal (autoregressive) masking.
            
        Returns:
            output: tensor of shape (B, H, T, D)
        """
        B, H, T, D = q.shape 
        
        
        # 1. Compute raw attention scores: QK^T 
        # (B, H, T, D) x (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        
        # 2. Scale by sqrt(d_k)
        scores = scores / math.sqrt(D)
        
        # 3. Apply causal mask if needed
        if causal:
            # Create lower-triangular mask
            mask = torch.tril(
                torch.ones(T, T, device = scores.device, dtype = torch.bool)
            )
            scores = scores.masked_fill(~mask, float('-inf'))
            
        # 4. Softmax over last dimension (keys)
        attn_weights = torch.softmax(scores, dim = -1)
        
        # 5. Weighted sum of values
        # (B, H, T, T) x (B, H, T, D) -> (B, H, T, D)
        output = torch.matmul(attn_weights, v)
        
        return output 