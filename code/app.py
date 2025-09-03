from flask import Flask, request, render_template, session, redirect, url_for, flash, jsonify
from kyes_trivia_ai_analyzer import KyesTriviaAIAnalyzer
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
import json
from datetime import datetime
import logging
import logging

app = Flask(__name__)
app.secret_key = 'kyes_trivia_system_secret_key_2025'  # セッション管理用

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
            'kyes_trivia_ai_dataset.json',
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

        # # 未回答の質問に回答を生成 (現在この機能は無効化されています)
        # try:
        #     generated_count = analyzer.generate_answers_for_unanswered()
        #     if generated_count > 0:
        #         print(f"{generated_count}件の新しいQ&Aを自動生成しました")
        # except Exception as e:
        #     print(f"回答の自動生成中にエラーが発生しました: {e}")
        
        print("アプリケーションの初期化が完了しました")
        return analyzer
        
    except Exception as e:
        print(f"アプリケーションの初期化に失敗しました: {e}")
        return None

# 質問ロガーの設定
question_logger = logging.getLogger('user_questions')
question_logger.setLevel(logging.INFO)
log_file_path = os.path.join(os.path.dirname(__file__), 'user_questions.log')
# ハンドラが重複して追加されるのを防ぐ
if not question_logger.handlers:
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    question_logger.addHandler(file_handler)

# グローバル変数としてアナライザーを保持
analyzer = initialize_app()

chat_history = []

def get_chat_history():
    return chat_history

def add_to_chat_history(question, answer, learned=False):
    chat_history.append({'question': question, 'answer': answer, 'learned': learned})

def clear_chat_history():
    chat_history.clear()

@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/chat', methods=["GET", "POST"])
def index():
    """メインページ（チャット）"""
    global analyzer
    
    # JSONリクエスト（API経由）の場合はログインチェックをスキップ
    if not request.is_json and 'user_id' not in session:
        flash('チャットを利用するにはログインが必要です。')
        return redirect(url_for('login'))

    # アナライザーが初期化されていない場合
    if analyzer is None:
        return render_template('index.html', 
                            error="システムの初期化に失敗しました。必要なファイルを確認してください。",
                            chat_history=[])
    
    error = None
    
    if request.method == "POST":
        current_question = ""
        # JSONリクエストとフォームリクエストの両方に対応
        if request.is_json:
            data = request.get_json()
            current_question = data.get('question') if data else request.form.get('question')
        else:
            current_question = request.form.get("question", "").strip()

        # ユーザーの質問をログに記録
        if current_question:
            question_logger.info(f"User question: {current_question}")

        if current_question:
            try:
                # AIアナライザーで応答を生成
                response_data = analyzer.generate_response(current_question, get_chat_history())

                if 'error' in response_data:
                    return jsonify(response_data), 500

                # チャット履歴に追加
                add_to_chat_history(
                    response_data['question'], 
                    response_data['answer'], 
                    response_data.get('learned', False)
                )

                return jsonify(response_data)


            except Exception as e:
                error_message = f"検索中にエラーが発生しました: {str(e)}"
                print(f"!s!! エラー発生: {error_message}")
                import traceback
                traceback.print_exc()
                # エラーが発生した場合も、ユーザーの質問を履歴に追加
                add_to_chat_history(current_question, "エラーが発生しました。詳細は管理者にご確認ください。")
                return jsonify({'error': 'サーバーでエラーが発生しました。'}), 500

    # GETリクエストの場合、またはPOSTで質問がない場合はページを普通に表示
    initial_questions_by_category = {}
    if not chat_history:
        initial_questions_by_category = {
            "学校生活": [
                "忘れ物をした場合、どうすればいいですか？",
                "校章のデザインにはどんな意味がありますか？",
                "食堂のメニューについて教えてください。",
            ],
            "季節の行事": [
                "一番楽しいイベントは何ですか？",
                "文化祭はいつですか？",
                "体育祭の種目を教えてください。",
            ],
            "授業について": [
                "福澤諭吉先生の教えについて教えてください。",
                "履修登録はどのように行いますか？",
                "おすすめの選択科目はありますか？",
            ]
        }

    has_history = len(chat_history) > 0
    return render_template('index.html', 
                           chat_history=chat_history,
                           initial_questions_by_category=initial_questions_by_category,
                           has_history=has_history,
                           error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    """ログインページ"""
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()

        if username in users and check_password_hash(users[username]['password'], password):
            session['user_id'] = username
            session['username'] = username
            flash('ログインしました。')
            return redirect(url_for('index'))
        else:
            flash('ユーザー名またはパスワードが正しくありません。', 'error')

    return render_template('login.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    """登録ページ"""
    if 'user_id' in session:
        return redirect(url_for('index'))

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
    clear_chat_history()
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
