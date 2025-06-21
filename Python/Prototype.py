from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

model_name = "rinna/japanese-gpt2-small"  # 使うモデル名をここで指定

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_ai(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_len = inputs['input_ids'].shape[1]
    outputs = model.generate(
        **inputs,
        max_length=input_len + 100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

iface = gr.Interface(fn=chat_with_ai, inputs="text", outputs="text", title="簡単対話AI")
iface.launch()
