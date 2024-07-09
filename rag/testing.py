import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

quant_model_path = "/home/patchy/Model/models/mergered_chn_llama/"


tokenizer = AutoTokenizer.from_pretrained(quant_model_path, use_fast=True)
t_input = tokenizer("hello world", return_tensors="pt", padding=True)
t_input.to("cuda")

model = AutoModelForCausalLM.from_pretrained(quant_model_path, torch_dtype=torch.float16)
model.to("cuda")
model.eval()

with torch.no_grad():
    result = model(**t_input, output_hidden_states=True)


print(result)