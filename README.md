
### 2. New `README.ja.md` (Japanese)

This is a new file containing the cleaned-up Japanese documentation.

```diff
--- /dev/null
+++ b/c:\Users\hidek\all\Artificial-Intelligence\README.ja.md
@@ -0,0 +1,114 @@
+# KYES_Trivia_AI
+
+[English](README.md) | 日本語
+
+---
+
+事前に定義された知識ベースに基づいてトリビアの質問に答えるように設計された、対話型のAIチャットボットです。このアプリケーションは、`sentence-transformer`モデルを使用して自然言語のクエリを理解し、最も関連性の高い回答を見つけ出します。
+
+## 主な特徴
+
+*   **自然な会話:** 自然でチャットのような会話が可能です。AIは文脈やユーザーの感情を理解し、最適な応答を提供します。
+*   **安全な認証:** ユーザーデータと履歴を保護するための、安全なユーザー登録・ログインシステムを搭載しています。
+*   **スマート検索:** `sentence-transformers`による文章の埋め込みベクトルを利用してセマンティック検索を実行し、クエリが完全なキーワードを含んでいなくても最も関連性の高い回答を見つけます。
+*   **継続的な学習:** 回答できなかった質問は記録され、時間とともにナレッジベースを拡張することで、より正確な応答が可能になります。
+*   **質問履歴:** 登録ユーザーはいつでも過去の質問を確認できます。
+*   **質問の提案:** ユーザーが始めやすいように、カテゴリ別に分類された質問のサンプルを提供します。
+
+## 技術スタック
+
+*   **バックエンド:** Python, Flask
+*   **データベース:** SQLite with Flask-SQLAlchemy
+*   **AI / 機械学習:**
+    *   `sentence-transformers` (テキストの埋め込みベクトル作成)
+    *   `numpy` (ベクトル計算)
+    *   `janome` (日本語の形態素解析)
+*   **フロントエンド:** HTML, CSS, Vanilla JavaScript (Fetch API)
+
+## はじめに
+
+以下の手順で、プロジェクトをローカル環境にセットアップして実行します。
+
+### 前提条件
+
+*   Python 3.8以上
+
+### インストール
+
+1.  **リポジトリをクローンします:**
+    ```bash
+    git clone <your-repository-url>
+    cd KYES_Trivia_AI
+    ```
+
+2.  **依存関係をインストールします:**
+    `requirements.txt` ファイルが提供されています。
+    ```bash
+    pip install -r requirements.txt
+    ```
+
+3.  **AIモデルをダウンロードします:**
+    アプリケーションは、事前学習済みの `sentence-transformer` モデルを必要とします。`download_model.py` スクリプトを実行して取得してください。
+    ```bash
+    python code/download_model.py
+    ```
+
+4.  **ナレッジベースを準備します:**
+    ルートディレクトリに `kyes_trivia_ai_dataset.json` ファイルを作成します。このファイルには、AIが使用するQ&Aペアが含まれます。以下のようなオブジェクトのJSON配列形式である必要があります。
+    ```json
+    [
+      {
+        "id": 1,
+        "question": "学校のクラブ活動には何がありますか？",
+        "answer": "様々なスポーツクラブや文化クラブがあります。",
+        "category": "学校生活",
+        "difficulty": "簡単",
+        "tags": ["クラブ", "活動"]
+      }
+    ]
+    ```
+
+5.  **アプリケーションを実行します:**
+    ```bash
+    python code/app.py
+    ```
+    サーバーが起動し、 `http://localhost:5000` でアプリケーションにアクセスできます。
+
+## 使い方
+
+1.  **登録:** ユーザー名とパスワードで新しいアカウントを作成します。
+2.  **ログイン:** 登録した情報でログインします。
+3.  **質問する:**
+    *   画面下部のチャット入力欄に質問を入力します。
+    *   または、カテゴリを選択し、サンプル質問をクリックして開始します。
+4.  **回答を見る:** AIからの回答がチャットウィンドウに表示されます。
+5.  **履歴を確認する:** `/questions` ページに移動して、過去の質問履歴を確認できます。
+6.  **ログアウト:** 使い終わったら、安全にアカウントからログアウトします。
+
+## プロジェクト構成
+
+```
+.
+├── code/
+│   ├── app.py                      # メインFlaskアプリケーション (ルーティング, 認証など)
+│   ├── kyes_trivia_ai_analyzer.py  # AIコアロジック (検索, ベクトル化, 応答生成)
+│   ├── download_model.py           # sentence-transformerモデルのダウンロード用スクリプト
+│   ├── static/
+│   │   ├── main.js                 # フロントエンドJavaScript (チャットUI)
+│   │   └── style.css               # アプリケーションのスタイルシート
+│   └── templates/
+│       ├── index.html              # メインチャットページ
+│       ├── login.html              # ログインページ
+│       ├── register.html           # 登録ページ
+│       └── questions.html          # ユーザーの質問履歴ページ
+│
+├── kyes_trivia_ai_dataset.json     # Q&Aナレッジベース
+├── question_embeddings.npy         # 質問のキャッシュ済み埋め込みベクトル
+├── requirements.txt                # プロジェクトの依存関係
+├── README.md                       # このファイル (英語)
+└── README.ja.md                    # 日本語README
+```

This should make your project's documentation much cleaner and more accessible. Let me know if you have any other questions!

<!--
[PROMPT_SUGGESTION]Refactor the user authentication in `app.py` to remove the old JSON file system and rely solely on the SQLAlchemy database.[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]The global `chat_history` variable in `app.py` is shared between all users. Can you fix this so each user has their own separate chat history stored in the session?[/PROMPT_SUGGESTION]
-->
