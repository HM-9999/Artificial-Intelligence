import openai

openai.api_key = "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"
openai.api_base = "https://openrouter.ai/openai/gpt-4o"

response = openai.ChatCompletion.create(
    model="mistralai/mistral-7b-instruct",  # 無料モデル
    messages=[
        {"role": "system", "content": "あなたは親切なAIです。"},
        {"role": "user", "content": "地球はなぜ青いの？"}
    ]
)

print(response["choices"][0]["message"]["content"])
