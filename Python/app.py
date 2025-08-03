from flask import Flask, request, render_template, session, jsonify, redirect, url_for
from keio_qa_analyzer import KeioQAAnalyzer
import os
import uuid
from datetime import datetime
import time
import threading
from functools import wraps

app = Flask(__name__)
app.secret_key = 'keio_qa_system_secret_key_2024'  # セッション管理用の秘密鍵

# ユーザーデータベース（簡易版）
USERS = {
    'admin': {
        'password': 'admin123',
        'name': '管理者',
        'role': 'admin'
    },
    'user1': {
        'password': 'user123',
        'name': '一般ユーザー1',
        'role': 'user'
    },
    'user2': {
        'password': 'user456',
        'name': '一般ユーザー2',
        'role': 'user'
    }
}

# ログイン必須デコレーター
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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

def search_with_timeout(question, timeout_seconds=10):
    """タイムアウト付きの検索処理"""
    result = {'answer': None, 'error': None}
    
    def search_task():
        try:
            # キーワード検索を優先し、結果が見つからない場合のみ類似質問検索
            search_results = analyzer.search_qa(question)
            if search_results:
                # キーワード検索で結果が見つかった場合
                result['answer'] = search_results[0]['answer']
            else:
                # キーワード検索で結果が見つからない場合、類似質問検索
                similar = analyzer.find_similar_questions(question, limit=3)
                if similar:
                    result['answer'] = similar[0][0]['answer']
                else:
                    # どちらも見つからない場合
                    result['answer'] = "申し訳ございませんが、該当する回答が見つかりませんでした。質問を言い換えてお試しください。"
        except Exception as e:
            result['error'] = f"検索中にエラーが発生しました: {str(e)}"
    
    # 検索処理を別スレッドで実行
    search_thread = threading.Thread(target=search_task)
    search_thread.start()
    search_thread.join(timeout=timeout_seconds)
    
    if search_thread.is_alive():
        result['error'] = "検索処理がタイムアウトしました。しばらく時間をおいて再度お試しください。"
    
    return result['answer'], result['error']

@app.route("/login", methods=["GET", "POST"])
def login():
    """ログインページ"""
    error = None
    
    if request.method == "POST":
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if username in USERS and USERS[username]['password'] == password:
            # ログイン成功
            session['user_id'] = username
            session['user_name'] = USERS[username]['name']
            session['user_role'] = USERS[username]['role']
            return redirect(url_for('index'))
        else:
            error = "ユーザー名またはパスワードが正しくありません"
    
    return render_template('login.html', error=error)

@app.route("/logout")
def logout():
    """ログアウト"""
    session.clear()
    return redirect(url_for('login'))

@app.route("/", methods=["GET", "POST"])
@login_required
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
                # タイムアウト付きの検索処理
                current_answer, search_error = search_with_timeout(current_question)
                
                if search_error:
                    error = search_error
                    add_to_chat_history(current_question, None, error)
                else:
                    add_to_chat_history(current_question, current_answer, None)
                
            except Exception as e:
                error = f"検索中にエラーが発生しました: {str(e)}"
                add_to_chat_history(current_question, None, error)
        else:
            error = "質問を入力してください"
    
    return render_template('index.html', 
                         chat_history=chat_history,
                         error=error,
                         user_name=session.get('user_name', ''))

@app.route("/search", methods=["POST"])
@login_required
def search_ajax():
    """AJAX用の検索エンドポイント"""
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'システムが初期化されていません'})
    
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': '質問を入力してください'})
    
    try:
        # タイムアウト付きの検索処理
        answer, error = search_with_timeout(question)
        
        if error:
            return jsonify({'error': error})
        else:
            return jsonify({'answer': answer})
            
    except Exception as e:
        return jsonify({'error': f'検索中にエラーが発生しました: {str(e)}'})

@app.route("/clear", methods=["POST"])
@login_required
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
