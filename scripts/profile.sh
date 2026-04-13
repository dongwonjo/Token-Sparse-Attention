# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

DEVICE=0
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

CUDA_VISIBLE_DEVICES=$DEVICE python -m sparse_attn.token_sparse.profile.profile \
--model_path $MODEL_PATH \
--delta 0.5



