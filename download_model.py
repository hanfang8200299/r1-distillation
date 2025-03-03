import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
# model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit", local_dir_use_symlinks=False,local_dir='/home/zli/work/ethan/fine_tune/model/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit',resume_download=True)
snapshot_download(repo_id="unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", local_dir_use_symlinks=False,local_dir='/home/hxt/work/ethan/fine_tune/model/unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit',resume_download=True)