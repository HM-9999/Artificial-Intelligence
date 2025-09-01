import requests
import json
import time

# FlaskアプリケーションのベースURL
BASE_URL = "http://localhost:5000"

def print_response(res_json, question):
    """レスポンスを整形して表示"""
    print(f"\n> あなた: {question}")
    time.sleep(1)
    
    if 'answer' in res_json:
        print(f"🤖 AI: {res_json['answer']}")
        if res_json.get('learned'):
            print("   (💡 AIは新しい質問を学習しました！)")
        
        if res_json.get('predicted_questions'):
            print("\n   🤔 おすすめの質問:")
            for pq in res_json['predicted_questions']:
                print(f"      - {pq}")

    elif 'error' in res_json:
        print(f"🤖 AI (エラー): {res_json['error']}")
    
    print("-" * 30)

def run_demo():
    """デモシナリオを実行"""
    
    # 0. 最初にチャット履歴をクリア
    try:
        requests.post(f"{BASE_URL}/clear")
        print("チャット履歴をクリアしました。")
    except requests.exceptions.ConnectionError as e:
        print(f"接続エラー: Flaskアプリケーションが起動しているか確認してください。 ({BASE_URL})")
        return

    # 1. 通常の質問
    question1 = "慶應義塾横浜初等部の特色は何ですか？"
    res1 = requests.post(f"{BASE_URL}/chat", json={'message': question1}).json()
    print_response(res1, question1)
    time.sleep(2)

    # 2. AIが知らない質問（学習をトリガー）
    question2 = "宇宙人はいますか？"
    res2 = requests.post(f"{BASE_URL}/chat", json={'message': question2}).json()
    print_response(res2, question2)
    time.sleep(2)

    # 3. 類似質問（重複回答防止をトリガー）
    question3 = "慶應の横浜初等部について教えて"
    res3 = requests.post(f"{BASE_URL}/chat", json={'message': question3}).json()
    print_response(res3, question3)
    time.sleep(2)

    # 4. サジェストされた質問から選択
    if res1.get('predicted_questions'):
        suggested_question = res1['predicted_questions'][0]
        print(f"--- サジェストされた質問「{suggested_question}」を選択 ---")
        res4 = requests.post(f"{BASE_URL}/chat", json={'message': suggested_question}).json()
        print_response(res4, suggested_question)
    
    print("\nデモが完了しました。")

if __name__ == "__main__":
    print("=== KYES Trivia AI デモスクリプト ===")
    run_demo()
