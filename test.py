import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = "./merged-ds-sft-baichuan-v1-gpu-a10012"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_path)
messages = []
messages.append({"role": "user", "content": "从医生的角度，糖尿病II型如何治疗？"})
response = model.chat(tokenizer, messages)
print(response)
