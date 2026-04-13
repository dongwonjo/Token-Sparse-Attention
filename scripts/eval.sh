# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

# ATTN_METHOD: vanilla, minference, flexprefill
# --token_sparse: True, False

DEVICE=0
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
ATTN_METHOD=vanilla

SAVE_DIR=outputs/test/$ATTN_METHOD

CUDA_VISIBLE_DEVICES=$DEVICE python -m main \
--model_path $MODEL_PATH \
--save_dir $SAVE_DIR \
--attn_method $ATTN_METHOD \
--token_sparse \
--coverage 0.005 \
--eval_longbench \
--eval_needle \
--eval_infinite_bench \
--eval_ruler



