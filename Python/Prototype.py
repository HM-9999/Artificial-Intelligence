from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# 軽量日本語モデル（rinnaのGPT2）
model_name = "rinna/japanese-gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_ai(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

iface = gr.Interface(fn=chat_with_ai, inputs="text", outputs="text", title="簡単対話AI")
iface.launch()
