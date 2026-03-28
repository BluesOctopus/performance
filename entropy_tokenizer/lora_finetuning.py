import os
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

from config import BASE_TOKENIZER, DATA_DIR, HF_TOKEN
from hierarchical_tokenizer import HierarchicalTokenizer
from embedding_init import smart_initialize_embeddings, load_dynamic_operators

def get_embedding_layer_names(model) -> list[str]:
    """Helper to find the embedding and lm_head layer names for different architectures."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            names.append(name.split('.')[-1])
        elif isinstance(module, torch.nn.Linear) and "head" in name.lower():
            names.append(name.split('.')[-1])
    return list(set(names))

def run_finetuning(
    output_dir: str = "checkpoints/hierarchical_lora",
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 1000, # For quick testing
):
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    print("[LoRA] Loading base model and tokenizer...")
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER, token=HF_TOKEN, trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("[LoRA] Loading dynamic operators...")
    dynamic_ops = load_dynamic_operators()
    if not dynamic_ops:
        print("[LoRA] Error: No dynamic operators found. Run frequency_miner.py first.")
        return
        
    # Add new dynamic operators to tokenizer
    op_tokens = [op["token_str"] for op in dynamic_ops]
    base_tokenizer.add_tokens(op_tokens)

    # Wrap with our Hierarchical Tokenizer
    # Assuming hierarchical_tokenizer is also updated to use dynamic_ops
    tokenizer = HierarchicalTokenizer(base_tokenizer, dynamic_ops)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_TOKENIZER, # e.g., "Salesforce/codegen-350M-mono"
        token=HF_TOKEN,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("[LoRA] Initializing embeddings for new operators...")
    # This also resizes the token embeddings
    model = smart_initialize_embeddings(model, tokenizer.base_tokenizer, dynamic_ops)
    
    # Identify embedding layers to train
    modules_to_save = get_embedding_layer_names(model)
    print(f"[LoRA] Will train embedding/head layers: {modules_to_save}")
    
    # Set up LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], # Will auto-fallback or can be customized per model
        modules_to_save=modules_to_save # CRITICAL: we must train the new embeddings
    )
    
    # Some models need specific target modules
    if "codegen" in BASE_TOKENIZER.lower():
        peft_config.target_modules = ["qkv_proj", "out_proj"]
    elif "starcoder" in BASE_TOKENIZER.lower():
        peft_config.target_modules = ["c_attn", "c_proj"]
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("[LoRA] Loading and tokenizing dataset...")
    dataset = load_from_disk(str(DATA_DIR / "train"))
    
    def tokenize_function(examples):
        # We must use our custom encode method
        # Since it's a batch, we process one by one
        encoded_batch = [tokenizer.encode(text, truncation=True, max_length=512) for text in examples["content"]]
        return {"input_ids": encoded_batch}
        
    # Take a small subset for quick training
    small_dataset = dataset.select(range(min(5000, len(dataset))))
    tokenized_dataset = small_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.base_tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("[LoRA] Starting training...")
    trainer.train()
    
    print("[LoRA] Saving model...")
    trainer.save_model(output_dir)
    tokenizer.base_tokenizer.save_pretrained(output_dir)
    print(f"[LoRA] Done! Saved to {output_dir}")

if __name__ == "__main__":
    run_finetuning()
