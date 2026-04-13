from importlib.metadata import version
import warnings
import transformers

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama(model, args):
    transformers_version = check_version()
    version_list = ['4.46']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with Token-Sparse Attention. Token-Sparse Attention is tested with Transformers version {version_list}.")
        
    from sparse_attn.token_sparse.sparse_cluster import init_cluster, set_model
    from sparse_attn.llama_hijack_4_46 import sparse_attn_forward, llama_model_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward

    for layer in model.model.layers:
        layer.self_attn.forward = sparse_attn_forward.__get__(layer.self_attn)
        layer.self_attn.attn_method = args.attn_method
    
        if args.token_sparse:
            layer.self_attn.init_cluster = init_cluster.__get__(layer.self_attn)
            layer.self_attn.init_cluster()
        else:
            layer.self_attn.sparse_cluster = None
            
        if args.attn_method == "flexprefill":
            layer.self_attn.gamma = args.gamma
            layer.self_attn.tau = args.tau
            layer.self_attn.block_size = args.block_size
            layer.self_attn.min_budget = 1024
        if args.attn_method == "minference":
            layer.self_attn.adaptive_budget = args.adaptive_budget
        if args.attn_method == "xattention":
            layer.self_attn.block_size = 128
            layer.self_attn.xattention_stride = args.xattention_stride
            layer.self_attn.xattention_threshold = args.xattention_threshold
            layer.self_attn.xattention_use_triton = args.xattention_use_triton
            layer.self_attn.xattention_chunk_size = args.xattention_chunk_size
            
    if args.token_sparse:
        set_model(model, args)


def replace_mistral(model, args):
    transformers_version = check_version()
    version_list = ['4.46']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with Token-Sparse Attention. Token-Sparse Attention is tested with Transformers version {version_list}.")
        
    from sparse_attn.token_sparse.sparse_cluster import init_cluster, set_model
    from sparse_attn.mistral_hijack_4_46 import sparse_attn_forward, mistral_model_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward

    for layer in model.model.layers:
        layer.self_attn.forward = sparse_attn_forward.__get__(layer.self_attn)
        layer.self_attn.attn_method = args.attn_method
    
        if args.token_sparse:
            layer.self_attn.init_cluster = init_cluster.__get__(layer.self_attn)
            layer.self_attn.init_cluster()
        else:
            layer.self_attn.sparse_cluster = None
            
        if args.attn_method == "flexprefill":
            layer.self_attn.gamma = args.gamma
            layer.self_attn.tau = args.tau
            layer.self_attn.block_size = args.block_size
            layer.self_attn.min_budget = 1024
        if args.attn_method == "minference":
            layer.self_attn.adaptive_budget = args.adaptive_budget

    if args.token_sparse:
        set_model(model, args)
            
        