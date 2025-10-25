"""
Attach LoRA adapters to a transformer for parameter-efficient fine-tuning.
Use when you later swap to a generative model for explanations.
"""
from peft import LoraConfig, get_peft_model

def add_lora_adapters(model, r: int = 8, alpha: int = 16, dropout: float = 0.1, target_modules=None):
    """
    Wrap a HF transformer with LoRA adapters.
    - r: rank of LoRA update matrices
    - alpha: scaling
    - dropout: LoRA dropout
    - target_modules: list[str] like ["q_proj", "v_proj"]
    """
    if target_modules is None:
        target_modules = ["q_lin", "v_lin", "k_lin", "o_lin"]  # common names; adjust per model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias="none", task_type="SEQ_CLS"
    )
    return get_peft_model(model, cfg)
