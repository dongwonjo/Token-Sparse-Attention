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
        
    from sparse_attn.token_sparse.profile.llama_hijack_4_46 import llama_model_forward, llama_decoder_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = llama_decoder_forward


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
        
    from sparse_attn.token_sparse.profile.mistral_hijack_4_46 import mistral_model_forward, mistral_decoder_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = mistral_decoder_forward
