import torch 

from src.benchmark.runner import run_seq_len_sweep

def main():
    assert torch.cuda.is_available(), "CUDA is required"
    
    device = "cuda"
    
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    backends = ["naive", "sdpa", "flash"]
    
    run_seq_len_sweep(
        seq_lens=seq_lens,
        backends=backends,
        batch_size=2,
        embed_dim=512,
        num_heads=8,
        causal=True,
        device=device,
        output_csv="results/raw/seq_len_sweep.csv"
    )
    

if __name__ == "__main__":
    main()