import torch 
from torch import nn

from src.attention.attention_factory import UnifiedAttention


class TransformerBlock(nn.Module):
    """
    Minimal Transformer block focused on attention benchmarking.
    
    Structure:
        x -> LayerNorm -> QKV Projections
          -> Attention -> Output Projection
          -> Residual add
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_backend: str,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads
        
        # LayerNorm before attention (Pre-LN Style)
        self.ln = nn.LayerNorm(embed_dim)
        
        
        # QKV Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        
        # Attention backend (naive/sdpa/flash)
        self.attention = UnifiedAttention(attention_backend)
        
        # Output projections
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        
        
    def _shape(self, x: torch.Tensor):
        """
        Convert (B, T, E) -> (B, H, T, D)
        """
        B, T, E = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    
    def forward(self, x: torch.Tensor, causal: bool = False):
        """
        Args:
            x: input tensor of shape (B, T, E)
            causal: whether to apply causal masking
            
        Returns:
            output tensor of shape (B, T, E)
        """
        residual = x 
        
        # Pre-norm
        x = self.ln(x)
        
        # Project to Q, K, V
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        
        # Attention
        attn_out = self.attention(q, k, v, causal = causal)
        
        # Back to (B, T, E)
        B, H, T, D = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, H * D)
        
        # Output Projection
        out = self.out_proj(attn_out)
        
        # Residual connection
        return out + residual