from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer
import torch
import pandas as pd
import gc


models = [
    'model/Qwen2.5-7B-Instruct-unsloth-bnb-4bit-stock-v1-4bit',
    'model/unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit',
    'model/unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit',
    ]

# Load prompts
data = pd.read_excel('data/LLM Finance Quiz Results.xlsx', engine='openpyxl')
questions = data['Question']
print(len(questions))

results = pd.DataFrame({'question':questions})

for model_path in models:
    # Load the model and tokenizer

    if model_path.startswith("model/unsloth"):
        # If the model is a pre-quantized model, addtional args needed
        model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
            quantization="bitsandbytes", 
            load_format="bitsandbytes"
        )
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=2,
        )
    tok = AutoTokenizer.from_pretrained(
        model_path
    )

    stop_token_ids = tok("<|im_end|>")["input_ids"]

    sampling_params = SamplingParams(
        max_tokens=5000,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )

    prompts = []
    for q in questions:
        prompt = "<|im_start|>system\nYou are a helpful assistant in financial services.<|im_end|>\n<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    # Inference
    outputs = model.generate(
        prompts,
        sampling_params=sampling_params
    )

    answers = []
    for output in outputs:
        generated_text = output.outputs[0].text
        answers.append(generated_text)
    print('Generated '+str(len(answers))+' answers.')


    # Consolidate the results to df
    results[model_path] = answers

    # Clear everything
    del prompts, model, tok, outputs
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

# Write the results to csv
results.to_csv('comparison_stock_questions.csv')

