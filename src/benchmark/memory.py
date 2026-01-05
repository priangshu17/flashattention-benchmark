import torch 

@torch.no_grad()
def memory_forward(
    model, 
    x,
    causal: bool,
    num_warmup: int = 10,
    num_iters: int = 1,
):
    """
    Measure peak GPU memory during forward pass.
    
    Returns:
        peak_allocated_mb: peak allocated memory in MB
        peak_reserved_mb: peak reserved memory in MB
    """
    assert x.is_cuda, "Input must be on CUDA"
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(x, causal = causal)
        
    torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(num_iters):
        _ = model(x, causal = causal)
        
    torch.cuda.synchronize()
    
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    
    return peak_allocated / 1024**2, peak_reserved / 1024**2


def memory_forward_backward(
    model,
    x, 
    causal: bool,
    num_warmup: int = 10,
    num_iters: int = 1,
):
    """
    Measure peak GPU memory during forward + backward pass.
    """
    assert x.is_cuda, "Input must be on CUDA"
    model.train()
    
    def loss_fn(y):
        return y.sum()
    
    # Warmup
    for _ in range(num_warmup):
        y = model(x, causal = causal)
        loss = loss_fn(y)
        loss.backward()
        model.zero_grad(set_to_none = True)
        
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(num_iters):
        y = model(x, causal = causal)
        loss = loss_fn(y)
        loss.backward()
        model.zero_grad(set_to_none = True)
        
    torch.cuda.synchronize()
    
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    
    return peak_allocated / 1024**2, peak_reserved / 1024**2
