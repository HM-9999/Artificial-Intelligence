from flask import Flask, request, render_template, session, redirect, url_for, flash
from kyes_trivia_ai_analyzer import KyesTriviaAIAnalyzer
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'kyes_trivia_system_secret_key_2024'  # セッション管理用の秘密鍵

USER_FILE = 'users.json'

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_users(users):
    with open(USER_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

# アプリケーション初期化
def initialize_app():
    """アプリケーションの初期化"""
    try:
        print("KyesTrivia_AIを起動中...")
        
        # 必要なファイルの存在確認
        required_files = [
            'kyes_trivia_dataset.json',
            'templates/index.html',
            'templates/login.html',
            'templates/register.html',
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
        analyzer = KyesTriviaAIAnalyzer()
        
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

def add_to_chat_history(question, answer, vectors=None):
    """チャット履歴に追加"""
    chat_history = get_chat_history()
    
    # 新しい会話を追加
    conversation = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'vectors': vectors or {}
    }
    
    chat_history.append(conversation)
    
    # 履歴が多すぎる場合は古いものを削除（最新50件を保持）
    if len(chat_history) > 50:
        chat_history = chat_history[-50:]
    
    session['chat_history'] = chat_history

@app.route('/')
def root():
    if 'user_id' in session:
        return redirect('/chat')
    return redirect(url_for('login'))

@app.route('/chat', methods=["GET", "POST"])
def index():
    """メインページ（チャット）"""
    global analyzer
    
    if 'user_id' not in session:
        flash('チャットを利用するにはログインが必要です。')
        return redirect(url_for('login'))

    # アナライザーが初期化されていない場合
    if analyzer is None:
        return render_template('index.html', 
                            error="システムの初期化に失敗しました。必要なファイルを確認してください。",
                            chat_history=[])
    
    chat_history = get_chat_history()
    error = None
    
    if request.method == "POST":
        current_question = ""
        # JSONリクエストとフォームリクエストの両方に対応
        if request.is_json:
            data = request.get_json()
            current_question = data.get('message', '').strip()
        else:
            current_question = request.form.get("question", "").strip()

        if current_question:
            try:
                current_answer_text = "申し訳ありません、その質問にはお答えできません。"
                
                # キーワード検索
                search_results = analyzer.search_qa(current_question)
                if search_results:
                    current_answer_text = search_results[0]['answer']
                else:
                    # 類似質問検索
                    similar = analyzer.find_similar_questions(current_question)
                    if similar:
                        current_answer_text = similar[0][0]['answer']
                
                # 単語ベクトルを生成
                raw_vectors = analyzer.vectorize_words(current_question)
                # JSONシリアライズ可能な形式に変換
                vectors = {word: vec.tolist() for word, vec in raw_vectors.items()}

                # チャット履歴に追加
                add_to_chat_history(current_question, current_answer_text, vectors)
                
            except Exception as e:
                error_message = f"検索中にエラーが発生しました: {str(e)}"
                flash(error_message, 'error')
                # エラーが発生した場合も、ユーザーの質問を履歴に追加
                add_to_chat_history(current_question, "エラーが発生しました。詳細は管理者にご確認ください。")

    # セッションからチャット履歴を取得
    chat_history = session.get('chat_history', [])
    
    return render_template('index.html', 
                         chat_history=chat_history,
                         error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    """ログインページ"""
    if 'user_id' in session:
        return redirect('/chat')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()

        if username in users and check_password_hash(users[username]['password'], password):
            session['user_id'] = username
            session['username'] = username
            flash('ログインしました。')
            return redirect('/chat')
        else:
            flash('ユーザー名またはパスワードが正しくありません。', 'error')

    return render_template('login.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    """登録ページ"""
    if 'user_id' in session:
        return redirect('/chat')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()

        if not username or not password:
            flash('ユーザー名とパスワードを入力してください。', 'error')
        elif username in users:
            flash('このユーザー名は既に使用されています。', 'error')
        elif len(password) < 8:
            flash('パスワードは8文字以上で設定してください。', 'error')
        else:
            users[username] = {'password': generate_password_hash(password)}
            save_users(users)
            flash('登録が完了しました。ログインしてください。', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('ログアウトしました。')
    return redirect(url_for('login'))

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
        print("1. kyes_trivia_dataset.json が存在するか")
        print("2. download_model.py を実行してモデルをダウンロードしたか")
        print("3. 必要な依存関係がインストールされているか")
    else:
        print("サーバーを起動中...")
        print("ブラウザで http://localhost:5000 にアクセスしてください")
        app.run(debug=True, host='0.0.0.0', port=5000)
