import json
import gradio as gr

# JSONファイルを読み込む
with open("conversation_data.json", encoding="utf-8") as f:
    conversations = json.load(f)

# 会話応答関数
def chat(input_text):
    for pair in conversations:
        if pair["input"].strip() == input_text.strip():
            return pair["output"]
    return "すみません、うまく答えられません。"

# Gradio UI
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="HIDEKIの会話AI（JSON版）")
iface.launch()
