import os
from sentence_transformers import SentenceTransformer

def download_model():
    """
    モデルをダウンロードする関数
    既にローカルに存在する場合はスキップ
    """
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    local_path = 'paraphrase-multilingual-MiniLM-L12-v2-local'
    
    # ローカルにモデルが既に存在するかチェック
    if os.path.exists(local_path) and os.path.isdir(local_path):
        print(f"モデルは既にローカルに存在します: {local_path}")
        print("オフライン環境でも使用可能です。")
        return local_path
    
    try:
        print(f"モデルをダウンロード中: {model_name}")
        print("インターネット接続が必要です...")
        
        # モデルをダウンロード
        model = SentenceTransformer(model_name)
        
        # ローカルに保存
        print(f"モデルをローカルに保存中: {local_path}")
        model.save(local_path)
        
        print("モデルのダウンロードと保存が完了しました！")
        print("これでオフライン環境でも使用可能です。")
        return local_path
        
    except Exception as e:
        print(f"モデルのダウンロードに失敗しました: {e}")
        print("インターネット接続を確認してください。")
        return None

def load_model():
    """
    モデルを読み込む関数
    ローカルモデルを優先し、存在しない場合はダウンロードを試行
    """
    local_path = 'paraphrase-multilingual-MiniLM-L12-v2-local'
    
    # ローカルモデルが存在する場合
    if os.path.exists(local_path) and os.path.isdir(local_path):
        try:
            print(f"ローカルモデルを読み込み中: {local_path}")
            model = SentenceTransformer(local_path)
            print("ローカルモデルの読み込みが完了しました！")
            return model
        except Exception as e:
            print(f"ローカルモデルの読み込みに失敗しました: {e}")
            print("モデルの再ダウンロードを試行します...")
    
    # ローカルモデルが存在しない場合、ダウンロードを試行
    print("オンラインモデルを読み込み中...")
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("オンラインモデルの読み込みが完了しました！")
        return model
    except Exception as e:
        print(f"オンラインモデルの読み込みに失敗しました: {e}")
        print("インターネット接続を確認してください。")
        return None

if __name__ == "__main__":
    print("モデルダウンロードツール")
    print("=" * 40)
    
    # モデルをダウンロード
    result = download_model()
    
    if result:
        print(f"\nモデルパス: {result}")
        print("セットアップ完了！")
    else:
        print("\nセットアップに失敗しました。")
        print("インターネット接続を確認してから再実行してください。")