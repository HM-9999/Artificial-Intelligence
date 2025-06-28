import json
import gradio as gr

with open("conversation_data.json", encoding="utf-8") as f:
    conversations = json.load(f)

def chat(input_text, history=[]):
    for pair in conversations:
        if pair["input"].strip() == input_text.strip():
            response = pair["output"]
            break
    else:
        response = "すみません、わかりません。"

    history = history + [(input_text, response)]
    return "", history

css = """
.gradio-container {
    background-color: #f7f7f8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
        Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
}
.message.user {
    background: #0b93f6;
    color: white;
    border-radius: 12px 12px 0 12px;
    padding: 8px 12px;
    margin: 4px 0;
    max-width: 70%;
    align-self: flex-end;
}
.message.bot {
    background: #e5e5ea;
    color: black;
    border-radius: 12px 12px 12px 0;
    padding: 8px 12px;
    margin: 4px 0;
    max-width: 70%;
    align-self: flex-start;
}
"""

with gr.Blocks(css=css) as demo:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(show_label=False, placeholder="ここにメッセージを入力してください", lines=2)
    txt.submit(chat, [txt, chatbot], [txt, chatbot])
    txt.focus()

demo.launch()
