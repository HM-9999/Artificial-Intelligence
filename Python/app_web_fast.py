from flask import Flask, request, render_template, session, jsonify
import json

app = Flask(__name__)
app.secret_key = 'keio_qa_system_secret_key_2024'

# データを直接読み込み
with open('keio_qa_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    qa_data = data.get('qa_pairs', [])

def fast_search(query):
    """超高速直接検索"""
    if not query:
        return None
    
    query_lower = query.lower()
    
    # 直接文字列マッチング
    for qa in qa_data:
        if (query_lower in qa['question'].lower() or 
            query_lower in qa['answer'].lower()):
            return qa['answer']
    
    return None

@app.route("/", methods=["GET"])
def index():
    """軽量化されたメインページ"""
    return render_template('index.html', chat_history=[], error=None)

@app.route("/api/search", methods=["POST"])
def api_search():
    """高速API検索"""
    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"error": "質問を入力してください"})
        
        # 超高速直接検索
        answer = fast_search(question)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": f"エラー: {str(e)}"})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    """履歴クリアAPI"""
    return jsonify({"success": True})

if __name__ == "__main__":
    print("Web高速サーバー起動中...")
    print("http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) 