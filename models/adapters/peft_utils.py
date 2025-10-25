"""
Attach LoRA adapters to a transformer for parameter-efficient fine-tuning.
Use when you later swap to a generative model for explanations.
"""
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel # Added import for type hinting

def add_lora_adapters(model: PreTrainedModel, r: int = 8, alpha: int = 16, dropout: float = 0.1, target_modules=None):
    """
    Wrap a HF transformer with LoRA adapters.
    
    The default target_modules are set for RoBERTa/BERT-style attention layers.
    """
    if target_modules is None:
        # These are the standard attention projection layers for RoBERTa/DistilRoBERTa
        # They map the input to Q, K, and V.
        target_modules = ["q_proj", "k_proj", "v_proj"]
        
        # Optional: You can also include the output linear layer for the self-attention
        # target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"] 
        
    cfg = LoraConfig(
        r=r, 
        lora_alpha=alpha, 
        target_modules=target_modules, 
        lora_dropout=dropout, 
        bias="none", 
        task_type="SEQ_CLS"
    )
    
    # NOTE: The get_peft_model function handles the complex model structure automatically.
    return get_peft_model(model, cfg)