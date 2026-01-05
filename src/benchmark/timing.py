import torch 

@torch.no_grad()
def time_forward(
    model,
    x,
    causal: bool,
    num_warmup: int = 10,
    num_iters: int = 50
):
    """
    Measure forward-pass latency using CUDA events.
    
    Returns:
        avg_time_ms: average forward time in milliseconds
    """
    assert x.is_cuda, "Input must be CUDA"
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(x, causal = causal)
        
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    
    start.record()
    for _ in range(num_iters):
        _ = model(x, causal = causal)
    end.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start.elapsed_time(end)
    avg_time_ms = total_time_ms / num_iters
    
    return avg_time_ms


def time_forward_backward(
    model,
    x,
    causal: bool,
    num_warmup: int = 10,
    num_iters: int = 50,
):
    """
    Measure forward + backward latency.
    """
    assert x.is_cuda, "Input must be CUDA"
    model.train()
    
    
    # Dummy loss(sum keeps graph simple)
    def loss_fn(y):
        return y.sum()
    
    # Warmup
    for _ in range(num_warmup):
        y = model(x, causal = causal)
        loss = loss_fn(y)
        loss.backward()
        model.zero_grad(set_to_none = True)
        
    torch.cuda.synchronize()
    
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    
    start.record()
    for _ in range(num_iters):
        y = model(x, causal = causal)
        loss = loss_fn(y)
        loss.backward()
        model.zero_grad(set_to_none = True)
    end.record()
    
    
    torch.cuda.synchronize()
    
    total_time_ms = start.elapsed_time(end)
    avg_time_ms = total_time_ms / num_iters 
    
    return avg_time_ms

def tokens_per_second(
    batch_size: int,
    seq_len: int,
    time_ms: float,
):
    """
    Computes throughput in tokens/sec.
    """
    tokens = batch_size * seq_len 
    return tokens / (time_ms / 1000.0)