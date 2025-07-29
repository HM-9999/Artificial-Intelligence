from flask import Flask, request, render_template, session
from keio_qa_analyzer import KeioQAAnalyzer
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'keio_qa_system_secret_key_2024'  # セッション管理用の秘密鍵

# アプリケーション初期化
def initialize_app():
    """アプリケーションの初期化"""
    try:
        print("慶應義塾 Q&A システムを起動中...")
        
        # 必要なファイルの存在確認
        required_files = [
            'keio_qa_dataset.json',
            'templates/index.html',
            'static/style.css'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"必要なファイルが見つかりません: {missing_files}")
            return None
        
        # アナライザーの初期化
        analyzer = KeioQAAnalyzer()
        
        if not analyzer.qa_data:
            print("Q&Aデータの読み込みに失敗しました")
            return None
        
        print("アプリケーションの初期化が完了しました")
        return analyzer
        
    except Exception as e:
        print(f"アプリケーションの初期化に失敗しました: {e}")
        return None

# グローバル変数としてアナライザーを保持
analyzer = initialize_app()

def get_chat_history():
    """チャット履歴を取得"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def add_to_chat_history(question, answer, error=None):
    """チャット履歴に追加"""
    chat_history = get_chat_history()
    
    # 新しい会話を追加
    conversation = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'error': error
    }
    
    chat_history.append(conversation)
    
    # 履歴が多すぎる場合は古いものを削除（最新50件を保持）
    if len(chat_history) > 50:
        chat_history = chat_history[-50:]
    
    session['chat_history'] = chat_history

@app.route("/", methods=["GET", "POST"])
def index():
    """メインページ"""
    global analyzer
    
    # アナライザーが初期化されていない場合
    if analyzer is None:
        return render_template('index.html', 
                            error="システムの初期化に失敗しました。必要なファイルを確認してください。",
                            chat_history=[])
    
    chat_history = get_chat_history()
    error = None
    
    if request.method == "POST":
        current_question = request.form.get("question", "").strip()
        if current_question:
            try:
                # キーワード検索を優先し、結果が見つからない場合のみ類似質問検索
                search_results = analyzer.search_qa(current_question)
                if search_results:
                    # キーワード検索で結果が見つかった場合
                    current_answer = search_results[0]
                else:
                    # キーワード検索で結果が見つからない場合、類似質問検索
                    similar = analyzer.find_similar_questions(current_question)
                    if similar:
                        current_answer = similar[0][0]
                    else:
                        # どちらも見つからない場合
                        current_answer = None
                
                # チャット履歴に追加
                add_to_chat_history(current_question, current_answer, error)
                
            except Exception as e:
                error = f"検索中にエラーが発生しました: {str(e)}"
                add_to_chat_history(current_question, None, error)
        else:
            error = "質問を入力してください"
    
    return render_template('index.html', 
                         chat_history=chat_history,
                         error=error)

@app.route("/clear", methods=["POST"])
def clear_history():
    """チャット履歴をクリア"""
    session.pop('chat_history', None)
    return {'status': 'success'}

@app.route("/health")
def health_check():
    """ヘルスチェック用エンドポイント"""
    return {
        "status": "healthy" if analyzer is not None else "error",
        "data_count": len(analyzer.qa_data) if analyzer else 0
    }

if __name__ == "__main__":
    if analyzer is None:
        print("アプリケーションを起動できません")
        print("以下の点を確認してください:")
        print("1. keio_qa_dataset.json が存在するか")
        print("2. download_model.py を実行してモデルをダウンロードしたか")
        print("3. 必要な依存関係がインストールされているか")
    else:
        print("サーバーを起動中...")
        print("ブラウザで http://localhost:5000 にアクセスしてください")
        app.run(debug=True, host='0.0.0.0', port=5000)
