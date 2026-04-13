# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

# ATTN_METHOD: vanilla, minference, flexprefill
# --token_sparse: True, False

DEVICE=0
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
ATTN_METHOD=vanilla

CUDA_VISIBLE_DEVICES=$DEVICE python -m benchmark.attention \
--model_path $MODEL_PATH \
--attn_method $ATTN_METHOD \
--token_sparse \
--coverage 0.005 \
--num_warmups 1 \
--num_runs 5 \
--context_length 131072