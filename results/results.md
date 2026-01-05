### Results

We evaluate three attention implementations, Naive Attention, PyTorch SDPA, and FlashAttention using a controlled sequence-length sweep. All experiments are conducted on a single Transformer block with fixed batch size, embedding dimension, and number of heads, varying only the sequence length. We report forward-pass runtime, peak GPU memory allocation, and throughput (tokens/sec).

#### Memory Usage

![1767627366389](image/results/1767627366389.png)

*
    Figure: Attention Memory Usage vs Sequence Length*

The memory usage results show a stark contrast between the three implementations.
Naive attention exhibits quadratic growth in peak allocated memory as sequence length increases. This behaviour is expected, as naive attention explicitly materializes the full attention score matrix of size ***TxT***, along with its softmax output. As sequence length increases, memory consumption rapidly becomes the dominant bottleneck, reaching several gigabytes even for moderate sequence lengths.

SDPA significantly reduces memory usage compared to the naive implementation but still shows superlinear growth. While SDPA avoids some intermediate materializations through kernel fusion, it remains constrained by the need to conceptually operate over the full attention matrix, leading to steadily increasing memory pressure.

FlashAttention, in contrast, demonstrates near-linear memory growth with sequence length. Across all tested sequence lengths, its peak memory usage remains dramatically lower than both naive and SDPA attention. This confirms the core design goal of FlashAttention: avoiding explicit materialization of the attention matrix by computing attention in tiled blocks and streaming intermediate results through on-chip memory. These results empirically validate the theoretical ***O(T)*** memory complexity of FlashAttention versus the ***O(T^2)*** behaviour of standard attention.

#### Runtime Performance

![1767627795933](image/results/1767627795933.png)

*
    Figure: Attention Runtime vs Sequence Length*

The runtime results mirror the memory trends but also reveal important performance trade-offs.

Naive attention shows rapidly increasing forward-pass latency as sequence length grows, consistent with its quadratic computational and memory costs. At larger sequence lengths, runtime increases sharply, making naive attention impractical for long-context settings.

SDPA provides a substantial runtime improvement over naive attention across all sequence length. Kernel fusion and improved memory layouts reduce overhead, but runtime still increases noticeably with sequence length, reflecting residual memory-bound behavior.

FlashAttention achieves the lowest runtime at large sequence lengths, clearly outperforming both naive and SDPA attention. While FlashAttention does not always provide the lowest latency at very small sequence lengths, where kernel launch overhead and tiling setup costs dominate, it scales far more favorably as sequence length increases. Beyond moderate sequence lengths, FlashAttention consistently delivers superior performance, confirming that it effectively shifts attention computation from a memory-bound regime to a more compute-efficient one.

#### Throughput

![1767628102007](image/results/1767628102007.png)

*
    Figure: Attention Throughput vs Sequence Length*

Throughput results further highlight the scaling advantages of FlashAttention.

Naive attention throughput drops sharply as sequence length increases, reflecting both increased computation and sever memory traffic. SDPA maintains higher throughput tha naive attention but still exhibits a gradual decling with sequence length.

FlashAttention achieves the highest throughput across nearly all sequence lengths, with particularly strong advantages at medium to large sequence lengths. Although throughput may peak at intermediate sequence lengths, where GPU utilization is highest, it remains substantially higher than both naive and SDPA attention even as sequence length continues to grow. This demonstrates that FlashAttention enables sustained high token throughput in long-context regimes, which is critical for large language model inference and training.

#### Summary of Findings

Overall, these results demonstrates that:

* Naive attention is fundamentally limited by quadratic memory and runtime scaling, making it unsuitable for long sequences.
* SDPA offers meaningful improvements through kernel fusion but remains constrained by attention matrix scaling.
* FlashAttention consistently achieves dramatically lower memory usage, better runtime scaling, and higher throughput, especially at larger sequence lengths.

These empirical results confirm that the primary benefit of FlashAttention lies not merely in constant-factor speedups, but in a fundamental shift in the memory and compute characteristics of attention. This makes FlashAttention a key enabler for long-context Transformers and efficient large-scale inference.
