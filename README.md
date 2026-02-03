# Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

<div align=center>
<img width=90% src="./images/TokenSparseAttention.png"/>
</div>
</br>

This is the official repository of **"Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection"** (WIP).

## Usage
### 1. Installation
Installation with the requirements package.
```
conda create -n token_sparse python=3.10
conda activate token_sparse
cd token_sparse
./install.sh
```

### 2. Quick Start
Inference with Token Sparse Attention methods and evaluation and speedup benchmark.

```
# Run benchmark
./scripts/benchmark_attention.sh

# Run Evaluation
./scripts/eval.sh
```
