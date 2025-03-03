from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the original model
base_model_path = "model/unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)

# Load LoRA adapter
lora_model_path = "outputs/Qwen2.5-7B-Instruct-unsloth-bnb-4bit-stock"
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge
merged_model = lora_model.merge_and_unload()

# Save the merged model
output_dir = "model/Qwen2.5-7B-Instruct-unsloth-bnb-4bit-stock-v1-4bit"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)