# KYES_Trivia_AI
こんにちはこのアプリについてです。

English | [日本語](README.ja.md)

---

A conversational AI chatbot designed to answer trivia questions based on a predefined knowledge base. This application uses a sentence-transformer model to understand natural language queries and find the most relevant answers.

## Key Features

*   **Natural Conversation:** Engage in a natural, chat-like conversation. The AI understands context and user sentiment to provide tailored responses.
*   **Secure Authentication:** Features a secure user registration and login system to protect user data and history.
*   **Smart Search:** Utilizes sentence embeddings (`sentence-transformers`) to perform semantic search, finding the most relevant answers even if the query doesn't use the exact keywords.
*   **Continuous Learning:** Unanswered questions are logged, allowing the system's knowledge base to be expanded over time, leading to more accurate responses.
*   **Question History:** Registered users can review their past questions at any time.
*   **Suggested Questions:** Provides users with sample questions grouped by category to help them get started.

## Tech Stack

*   **Backend:** Python, Flask
*   **Database:** SQLite with Flask-SQLAlchemy
*   **AI / ML:**
    *   `sentence-transformers` for creating text embeddings.
    *   `numpy` for vector calculations.
    *   `janome` for Japanese text tokenization.
*   **Frontend:** HTML, CSS, Vanilla JavaScript (with Fetch API)

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/HM-9999/Artificial-Intelligence.git>
    cd KYES_Trivia_AI
    ```

2.  **Install dependencies:**
    The `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the AI Model:**
    The application requires a pre-trained sentence-transformer model. Run the `download_model.py` script to fetch it.
    ```bash
    python code/download_model.py
    ```

4.  **Prepare the Knowledge Base:**
    Create a `kyes_trivia_ai_dataset.json` file in the root directory. This file contains the question-and-answer pairs that the AI will use. It should be a JSON array of objects, like this:
    ```json
    [
      {
        "id": 1,
        "question": "What are the school club activities?",
        "answer": "There are various sports and cultural clubs available.",
        "category": "School Life",
        "difficulty": "Easy",
        "tags": ["club", "activity"]
      }
    ]
    ```

5.  **Run the application:**
    ```bash
    python code/app.py
    ```
    The server will start, and you can access the application at `http://localhost:5000`.

## How to Use

1.  **Register:** Create a new account with a username and password.
2.  **Login:** Log in with your credentials.
3.  **Ask a Question:**
    *   Type your question into the chat input at the bottom.
    *   Or, select a category and click on a sample question to get started.
4.  **View Response:** The AI's answer will appear in the chat window.
5.  **Check History:** Navigate to the `/questions` page to see a history of all the questions you've asked.
6.  **Logout:** Securely log out of your account when you are finished.

## Project Structure

```
.
├── code/
│   ├── app.py                      # Main Flask application (routing, auth, etc.)
│   ├── kyes_trivia_ai_analyzer.py  # Core AI logic (search, embeddings, response generation)
│   ├── download_model.py           # Script to download the sentence-transformer model
│   ├── static/
│   │   ├── main.js                 # Frontend JavaScript for chat interface
│   │   └── style.css               # Styles for the application
│   └── templates/
│       ├── index.html              # Main chat page
│       ├── login.html              # Login page
│       ├── register.html           # Registration page
│       └── questions.html          # User question history page
│
├── kyes_trivia_ai_dataset.json     # Q&A knowledge base
├── question_embeddings.npy         # Cached embeddings for the questions
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

---

<a id="日本語"></a>

# KYES_Trivia_AI (日本語)

事前に定義された知識ベースに基づいてトリビアの質問に答えるように設計された、対話型のAIチャットボットです。このアプリケーションは、`sentence-transformer`モデルを使用して自然言語のクエリを理解し、最も関連性の高い回答を見つけ出します。

## 主な特徴

1. **自然な会話が可能**
   - 日常会話のような回答
   - 文脈を理解した回答を生成
   - ユーザーの感情の分析
*   **自然な会話:** 自然でチャットのような会話が可能です。AIは文脈やユーザーの感情を理解し、最適な応答を提供します。
*   **安全な認証:** ユーザーデータと履歴を保護するための、安全なユーザー登録・ログインシステムを搭載しています。
*   **スマート検索:** `sentence-transformers`による文章の埋め込みベクトルを利用してセマンティック検索を実行し、クエリが完全なキーワードを含んでいなくても最も関連性の高い回答を見つけます。
*   **継続的な学習:** 回答できなかった質問は記録され、時間とともにナレッジベースを拡張することで、より正確な応答が可能になります。
*   **質問履歴:** 登録ユーザーはいつでも過去の質問を確認できます。
*   **質問の提案:** ユーザーが始めやすいように、カテゴリ別に分類された質問のサンプルを提供します。

2. **セキュアな認証システム**
   - アカウントの登録
   - 個人情報の保護を重視
## 技術スタック

5. **継続的な学習**
   - ユーザーの質問から学習を進化
   - より正確な回答を提供
*   **バックエンド:** Python, Flask
*   **データベース:** SQLite with Flask-SQLAlchemy
*   **AI / 機械学習:**
    *   `sentence-transformers` (テキストの埋め込みベクトル作成)
    *   `numpy` (ベクトル計算)
    *   `janome` (日本語の形態素解析)
*   **フロントエンド:** HTML, CSS, Vanilla JavaScript (Fetch API)

## 使用方法
## はじめに

1. **アカウント登録**
   - 初めての方は新規登録からアカウントを作成
   - メールアドレスとパスワードを設定
以下の手順で、プロジェクトをローカル環境にセットアップして実行します。

2. **ログイン**
   - 登録済みのメールアドレスとパスワードでログイン
   - ゲストログインも可能（一部機能制限あり）
### 前提条件

3. **質問の入力**
   - チャット画面の入力欄に質問を入力
   - カテゴリボタンからサンプル質問を選択可能
*   Python 3.8以上

4. **回答の確認**
   - AIからの回答が即座に表示されます
   - 不明点はさらに質問を追加
### インストール

5. **チャット履歴の確認**
   - 過去の会話をいつでも確認可能
   - 必要に応じて履歴を削除
1.  **リポジトリをクローンします:**
    ```bash
    git clone <https://github.com/HM-9999/Artificial-Intelligence.git>
    cd KYES_Trivia_AI
    ```

6. **ログアウト**
   - 使用後は必ずログアウトを
   - 次回も同じアカウントで継続利用可能
2.  **依存関係をインストールします:**
    `requirements.txt` ファイルが提供されています。
    ```bash
    pip install -r requirements.txt
    ```

3.  **AIモデルをダウンロードします:**
    アプリケーションは、事前学習済みの `sentence-transformer` モデルを必要とします。`download_model.py` スクリプトを実行して取得してください。
    ```bash
    python code/download_model.py
    ```

4.  **ナレッジベースを準備します:**
    ルートディレクトリに `kyes_trivia_ai_dataset.json` ファイルを作成します。このファイルには、AIが使用するQ&Aペアが含まれます。以下のようなオブジェクトのJSON配列形式である必要があります。
    ```json
    [
      {
        "id": 1,
        "question": "学校のクラブ活動には何がありますか？",
        "answer": "様々なスポーツクラブや文化クラブがあります。",
        "category": "学校生活",
        "difficulty": "簡単",
        "tags": ["クラブ", "活動"]
      }
    ]
    ```

5.  **アプリケーションを実行します:**
    ```bash
    python code/app.py
    ```
    サーバーが起動し、 `http://localhost:5000` でアプリケーションにアクセスできます。

## 使い方

1.  **登録:** ユーザー名とパスワードで新しいアカウントを作成します。
2.  **ログイン:** 登録した情報でログインします。
3.  **質問する:**
    *   画面下部のチャット入力欄に質問を入力します。
    *   または、カテゴリを選択し、サンプル質問をクリックして開始します。
4.  **回答を見る:** AIからの回答がチャットウィンドウに表示されます。
5.  **履歴を確認する:** `/questions` ページに移動して、過去の質問履歴を確認できます。
6.  **ログアウト:** 使い終わったら、安全にアカウントからログアウトします。

## プロジェクト構成

```
.
├── code/
│   ├── app.py                      # メインFlaskアプリケーション (ルーティング, 認証など)
│   ├── kyes_trivia_ai_analyzer.py  # AIコアロジック (検索, ベクトル化, 応答生成)
│   ├── download_model.py           # sentence-transformerモデルのダウンロード用スクリプト
│   ├── static/
│   │   ├── main.js                 # フロントエンドJavaScript (チャットUI)
│   │   └── style.css               # アプリケーションのスタイルシート
│   └── templates/
│       ├── index.html              # メインチャットページ
│       ├── login.html              # ログインページ
│       ├── register.html           # 登録ページ
│       └── questions.html          # ユーザーの質問履歴ページ
│
├── kyes_trivia_ai_dataset.json     # Q&Aナレッジベース
├── question_embeddings.npy         # 質問のキャッシュ済み埋め込みベクトル
├── requirements.txt                # プロジェクトの依存関係
└── README.md                       # このファイル
```