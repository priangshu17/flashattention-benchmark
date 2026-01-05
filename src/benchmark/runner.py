import csv 
from typing import List 

import torch 

from src.model.transformer_block import TransformerBlock
from src.benchmark.timing import time_forward, tokens_per_second
from src.benchmark.memory import memory_forward


def run_seq_len_sweep(
    seq_lens: List[int],
    backends: List[str],
    batch_size: int,
    embed_dim: int,
    num_heads: int,
    causal: bool,
    device: str,
    output_csv: str,
):
    """
    Run a sequence-length sweep benchmark and write results to CSV.
    """
    
    results = []
    
    for backend in backends:
        print(f"\n=== Backend: {backend} ===")
        
        for T in seq_lens:
            print(f"Sequence Length: {T}")
            
            # Decide dtype
            if backend == "flash":
                dtype = torch.float16 
            else:
                dtype = torch.float32 
                
            # Create fresh input
            x = torch.randn(
                batch_size,
                T,
                embed_dim,
                device = device,
                dtype = dtype,
            )
            
            # Create fresh model
            model = TransformerBlock(
                embed_dim = embed_dim,
                num_heads = num_heads,
                attention_backend = backend,
            ).to(device)
            
            if dtype == torch.float16:
                model = model.half()
                
            # --- Timing --- 
            time_ms = time_forward(
                model = model,
                x = x,
                causal = causal,
            )
            
            tps = tokens_per_second(
                batch_size = batch_size,
                seq_len = T,
                time_ms = time_ms
            )
            
            # Memory
            mem_alloc_mb, mem_reserved_mb = memory_forward(
                model = model,
                x = x,
                causal = causal,
            )
            
            results.append({
                "backend": backend,
                "batch_size": batch_size,
                "seq_len": T,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "Causal": causal,
                "dtype": str(dtype),
                "time_ms": time_ms,
                "tokens_per_sec": tps,
                "mem_alloc_mb": mem_alloc_mb,
                "mem_reserved_mb": mem_reserved_mb,
            })
            
            # Free memory aggressively between configs
            del model, x 
            torch.cuda.empty_cache()
            
    # Write CSV
    fieldnames = list(results[0].keys())
    
    with open(output_csv, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults written to {output_csv}")