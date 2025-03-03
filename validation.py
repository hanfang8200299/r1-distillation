import torch
from unsloth import FastLanguageModel
import json
import pandas as pd

# Load the models
model_name = "model/unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
max_seq_length = 2048

model_ori, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype = None,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
model_new = FastLanguageModel.get_peft_model(
    model_ori,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# add adapter
adapter_path = "outputs/Qwen2.5-7B-Instruct-unsloth-bnb-4bit/checkpoint-120"
model_new.load_adapter(
    model_id=adapter_path,
    adapter_name="my_adapter"
)

FastLanguageModel.for_inference(model_ori)
FastLanguageModel.for_inference(model_new)

# Prepare data
prompt_style = """
Your job is to answer the given question. Think step-by-step before you conclude.
###Question:
{}
You answer should follow the following structure:
###Think
<Your reasoning here>
###Answer:
<You final answer>
"""
with open('data/gemini_cot_2398_0220.json','r') as f:
    data = json.load(f)
data = data[10:15]

# question = "What are ways to establish credit history for international student?"
# question = '''
# 2025年2月10日，比亚迪在智能化战略发布会上推出“天神之眼”高阶智驾系统（DiPilot），宣布旗下所有车型均标配高阶智驾功能，正式开启“全民智驾”时代，并将高快领航的价格降低到了10万级车型以内。比亚迪董事长王传福表示：“整车智能，才是真智能。”这一理念贯穿于“天神之眼”的整个技术架构中，通过强大的硬件支持和自研的端到端算法架构，保障车辆在任何复杂场景中的感知和决策能力。 回顾2024年2月19日，比亚迪推出了秦PLUS荣耀版和驱逐舰05荣耀版，售价低至7.98万元起，并首次正式提出“电比油低”、“油电同价”的概念，标志着比亚迪开始“电动化”内卷之路。 2024年，比亚迪销量达到425万辆，稳居中国车企销量冠军，并成为全球新能源销量冠军。与此同时，比亚迪迎来品牌成立30周年，并创造了全球首个新能源乘用车累计销量突破1000万辆的纪录。然而，尽管比亚迪在电动化赛道取得巨大成功，其智能化发展却一直处于行业第二、第三梯队，长期被外界诟病智能座舱和辅助驾驶能力较为保守。 过去，比亚迪对自动驾驶技术持谨慎态度，董事长王传福曾公开质疑无人驾驶的可行性，强调技术尚不成熟且存在法律障碍。然而，2024年市场竞争加剧，比亚迪意识到智能驾驶正成为新能源车市场的核心竞争点，因此迅速调整战略，大幅加码智能驾驶技术研发，推出“天神之眼”高阶智驾系统，并计划实现全系普及化。那么这次比亚迪推动的“智驾平权”运动，是否又将开启新一轮的“智能化”内卷之战呢？ 比亚迪“天神之眼”高阶智驾系统采用了多种传感器的融合技术，包括激光雷达、毫米波雷达、超声波雷达和高清摄像头等，构建了360度全方位的感知视野。每种传感器都有其独特的优点和局限，通过多传感器的数据融合，系统可以综合各类传感器的优势，消除单一传感器的盲点，提供精准的环境感知能力。特别是在复杂的城市拥堵路况和高速公路环境下，“天神之眼”系统的感知能力表现尤为出色。 比亚迪在“天神之眼”系统中采用了自研的璇玑架构，配备高算力的计算平台。该平台能够实时处理大数据，精准监控车辆状态、环境信息及驾驶员状况，并根据实时数据进行快速决策。系统搭载的端到端大模型算法架构能够使得车辆在行驶过程中迅速做出反应，处理复杂的交通场景，保证行车安全。 “天神之眼”系统支持包括城市道路、高速公路、窄道通行、自动泊车等多种复杂场景下的智驾体验。例如，城市道路中的红绿灯识别、车道保持、避让大车、汇入路口等功能都能通过系统完美实现。此外，在停车场景中，“天神之眼”支持易四方泊车和易三方泊车等智能泊车功能，能够处理超过300种泊车场景，显著提升了智能泊车能力。 “天神之眼”C版本具备高快领航功能，可在高速公路上实现1000公里以上的自动驾驶，在时速100km/h下能够稳定完成自动刹车，代客泊车成功率高达99%。这些高阶功能不仅适用于比亚迪的高端车型，如仰望U8、腾势系列等，也逐步向中低端车型普及，拉低了高阶智驾系统的价格门槛。 “天神之眼”系统还具备无图城市领航功能（CNOA），即使在没有高清地图的情况下，系统也能通过实时感知和算法推理，在城市复杂的路况中进行导航。这一技术的推出，标志着比亚迪在智能驾驶领域技术上的一次跨越，能够在不依赖高清地图的情况下，精确控制车辆行驶路线。”以下是比亚迪发布会的内容。请告诉我有哪些投资机会，并告诉我原因
# '''
results=[]
for i in data:
    question = i['instruction']
    inputs = tokenizer([prompt_style.format(question)], return_tensors="pt").to("cuda")

    # Gen text
    outputs_ori = model_ori.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=5000, 
        use_cache=True
    )
    outputs_new = model_new.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=5000, 
        use_cache=True
    )
    
    
    # Decode the text
    generated_text_ori = tokenizer.decode(outputs_ori[0], skip_special_tokens=True)
    generated_text_new = tokenizer.decode(outputs_new[0], skip_special_tokens=True)
    
    res = {
        'question':question,
        'answer_ori':generated_text_ori,
        'answer_new':generated_text_new
    }
    results.append(res)
    
    # print(f"Original model:\n{model_name}\n{generated_text}")
    # with open('generated_text_original.txt', 'w') as f:
    #     f.write(generated_text)

df = pd.DataFrame(results)
df.to_csv('comparison.csv')