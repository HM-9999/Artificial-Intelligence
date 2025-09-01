import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np
from janome.tokenizer import Tokenizer
from download_model import load_model

# 類義語・関連語の辞書
SYNONYM_DICT = {
    "部活": ["クラブ", "サークル", "課外活動"],
    "慶應": ["慶應義塾", "慶應大学", "Keio"],
    "入試": ["入学試験", "受験", "入試制度"],
    "キャンパス": ["校舎", "施設", "キャンパス"],
    "授業": ["講義", "授業", "カリキュラム"],
    "学生": ["学生", "生徒", "慶應生"],
    "教授": ["教員", "教授", "先生"],
    "研究": ["研究", "研究室", "研究活動"],
    "留学": ["海外留学", "留学制度", "国際交流"],
    "就職": ["就職活動", "就職", "キャリア"],
    "奨学金": ["奨学金", "学費支援", "経済支援"],
    "図書館": ["図書館", "学習施設", "資料室"],
    "食堂": ["食堂", "カフェテリア", "学食"],
    "体育": ["体育", "スポーツ", "運動"],
    "文化祭": ["文化祭", "学園祭", "イベント"],
    "慶應義塾横浜初等部": ["慶應義塾横浜初等部", "初等部", "学校"],
}

def preprocess_text(text: str) -> str:
    """テキストの前処理"""
    # 特殊文字の正規化
    text = re.sub(r'[【】（）()（）]', '', text)
    text = re.sub(r'[、。，．]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def tokenize(text):
    t = Tokenizer()
    return [token.surface for token in t.tokenize(text)]

def expand_words(words, synonym_dict):
    expanded = set(words)
    for word in words:
        if word in synonym_dict:
            expanded.update(synonym_dict[word])
    return list(expanded)

class KyesTriviaAIAnalyzer:
    def __init__(self, data_file: str = "kyes_trivia_dataset.json"):
        self.data_file = data_file
        self.qa_data = []
        self.meta_info = {}
        self.load_data()
        print("モデルを読み込み中...")
        self.model = load_model()
        if self.model is None:
            raise RuntimeError("モデルの読み込みに失敗しました。download_model.pyを実行してください。")
        self.question_embeddings = self._embed_all_questions()
        self.unanswered_file = os.path.join(os.path.dirname(__file__), 'unanswered_questions.json')

    def load_data(self):
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSONデータはリスト形式なので、直接代入する
                self.qa_data = data
                self.meta_info = {}
            print(f"データ読み込み完了: {len(self.qa_data)}件のQ&A")
        except FileNotFoundError:
            print(f"エラー: {self.data_file} が見つかりません")
            self.qa_data = []
            self.meta_info = {}
        except json.JSONDecodeError as e:
            print(f"JSONファイル読み込みエラー: {e}")
            self.qa_data = []
            self.meta_info = {}

    def _embed_all_questions(self):
        if not self.qa_data:
            print("データがありません")
            return np.array([])
        
        print("質問文をベクトル化中...")
        # 前処理された質問文を作成
        processed_questions = []
        for qa in self.qa_data:
            # 質問文の前処理
            processed_q = preprocess_text(qa['question'])
            # 類義語拡張
            words = tokenize(processed_q)
            expanded_words = expand_words(words, SYNONYM_DICT)
            expanded_q = " ".join(expanded_words)
            processed_questions.append(expanded_q)
        
        embeddings = self.model.encode(processed_questions, convert_to_numpy=True)
        print(f"ベクトル化完了: {len(embeddings)}件")
        return embeddings

    def search_qa(self, query: str, category: str = None, 
                  difficulty: str = None, tags: List[str] = None) -> List[Dict]:
        results = []
        if query:
            # クエリの前処理と拡張
            processed_query = preprocess_text(query)
            words = tokenize(processed_query)
            expanded_words = expand_words(words, SYNONYM_DICT)
            expanded_query = " ".join(expanded_words)
            query_lower = expanded_query.lower()
        else:
            query_lower = ""
        
        for qa in self.qa_data:
            # キーワード検索（前処理されたテキストで比較）
            if query:
                processed_q = preprocess_text(qa['question'])
                processed_a = preprocess_text(qa['answer'])
                if (query_lower not in processed_q.lower() and 
                    query_lower not in processed_a.lower()):
                    continue
            
            # フィルタ適用
            if category and qa['category'] != category:
                continue
            if difficulty and qa['difficulty'] != difficulty:
                continue
            if tags:
                qa_tags = qa.get('tags', [])
                if not any(tag in qa_tags for tag in tags):
                    continue
            
            results.append(qa)
        
        return results
    
    def find_similar_questions(self, target_question: str, limit: int = 5) -> List[Tuple[Dict, float]]:
        if not self.qa_data or self.question_embeddings.shape[0] == 0:
            return []
        
        # 入力質問の前処理と拡張
        processed_target = preprocess_text(target_question)
        words = tokenize(processed_target)
        expanded_words = expand_words(words, SYNONYM_DICT)
        expanded_target = " ".join(expanded_words)
        
        # 入力質問をベクトル化
        target_emb = self.model.encode([expanded_target], convert_to_numpy=True)[0]
        
        # コサイン類似度計算（数値安定性を向上）
        similarities = np.dot(self.question_embeddings, target_emb) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(target_emb) + 1e-8)
        
        # 類似度の閾値フィルタリング（0.3以上のみ）
        threshold = 0.3
        valid_indices = similarities >= threshold
        
        if not np.any(valid_indices):
            return []
        
        # 上位N件を取得
        top_idx = similarities.argsort()[::-1][:limit]
        results = [(self.qa_data[i], float(similarities[i])) for i in top_idx if similarities[i] >= threshold]
        return results

    def vectorize_words(self, text: str) -> Dict[str, np.ndarray]:
        """
        文章を単語に分割し、各単語をベクトル化する。
        """
        if not self.model:
            print("モデルが読み込まれていません。")
            return {}
        
        # テキストの前処理と単語分割
        processed_text = preprocess_text(text)
        words = tokenize(processed_text)
        
        if not words:
            return {}
        
        # 重複を除いた単語リストでベクトル化
        unique_words = list(set(words))
        word_embeddings = self.model.encode(unique_words, convert_to_numpy=True)
        
        # 単語とベクトルのマッピングを作成
        vector_dict = {word: vec for word, vec in zip(unique_words, word_embeddings)}
        
        return vector_dict

    def get_basic_stats(self) -> Dict:
        if not self.qa_data:
            return {"error": "データがありません"}
        
        categories = Counter(qa['category'] for qa in self.qa_data)
        difficulties = Counter(qa['difficulty'] for qa in self.qa_data)
        
        all_tags = []
        for qa in self.qa_data:
            all_tags.extend(qa.get('tags', []))
        tag_counts = Counter(all_tags)
        
        question_lengths = [len(qa['question']) for qa in self.qa_data]
        answer_lengths = [len(qa['answer']) for qa in self.qa_data]
        
        stats = {
            "総Q&A数": len(self.qa_data),
            "カテゴリ別分布": dict(categories),
            "難易度別分布": dict(difficulties),
            "人気タグ": dict(tag_counts.most_common(10)),
            "質問文字数": {
                "平均": round(sum(question_lengths) / len(question_lengths), 1),
                "最大": max(question_lengths),
                "最小": min(question_lengths)
            },
            "回答文字数": {
                "平均": round(sum(answer_lengths) / len(answer_lengths), 1),
                "最大": max(answer_lengths),
                "最小": min(answer_lengths)
            }
        }
        
        return stats

    def get_qa_by_id(self, qa_id: int) -> Optional[Dict]:
        for qa in self.qa_data:
            if qa['id'] == qa_id:
                return qa
        return None

    def _load_unanswered_questions(self):
        if os.path.exists(self.unanswered_file):
            with open(self.unanswered_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_unanswered_questions(self, questions):
        with open(self.unanswered_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

    def learn_new_qa(self, question):
        unanswered = self._load_unanswered_questions()
        # 既に同じ質問がないかチェック
        if not any(item['question'] == question for item in unanswered):
            unanswered.append({"question": question, "answer": ""})
            self._save_unanswered_questions(unanswered)

    def get_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def predict_next_questions(self, question: str, limit: int = 3) -> List[str]:
        """現在の質問に関連する次の質問を予測する"""
        if not self.qa_data:
            return []

        # 現在の質問からキーワードを抽出
        processed_question = preprocess_text(question)
        words = tokenize(processed_question)
        
        if not words:
            return []

        # 関連タグを持つ質問を収集
        related_questions = []
        for qa in self.qa_data:
            qa_tags = qa.get('tags', [])
            # 質問文自体が類似していないか、かつタグが関連しているか
            if question != qa['question'] and any(word in qa_tags for word in words):
                related_questions.append(qa['question'])
        
        # 重複を除いて返す
        unique_questions = list(dict.fromkeys(related_questions))
        return unique_questions[:limit]

    def validate_data(self) -> Dict:
        issues = []
        
        for qa in self.qa_data:
            required_fields = ['id', 'question', 'answer', 'category']
            for field in required_fields:
                if not qa.get(field):
                    issues.append(f"ID {qa.get('id', 'Unknown')}: {field} が空です")
            
            if qa.get('question') and len(qa['question']) < 10:
                issues.append(f"ID {qa['id']}: 質問が短すぎます ({len(qa['question'])}文字)")
            
            if qa.get('answer') and len(qa['answer']) < 20:
                issues.append(f"ID {qa['id']}: 回答が短すぎます ({len(qa['answer'])}文字)")
        
        ids = [qa['id'] for qa in self.qa_data]
        duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
        if duplicate_ids:
            issues.append(f"重複するID: {duplicate_ids}")
        
        return {
            "total_issues": len(issues),
            "issues": issues,
            "quality_score": max(0, 100 - len(issues) * 5)
        }

def interactive_analyzer():
    analyzer = KyesTriviaAIAnalyzer()
    
    if not analyzer.qa_data:
        print("Q&Aデータが読み込めませんでした。kyes_trivia_dataset.json ファイルを確認してください。")
        return
    
    print("=== TYES Trivia AI 分析ツール ===")
    print("コマンド: stats, search, similar, vectorize, validate, quit")
    
    while True:
        command = input("\nコマンドを入力: ").strip().lower()
        
        if command == 'quit' or command == 'q':
            print("分析ツールを終了します")
            break
        
        elif command == 'stats':
            stats = analyzer.get_basic_stats()
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        
        elif command == 'search':
            query = input("検索キーワード: ")
            category = input("カテゴリ (空白でスキップ): ") or None
            difficulty = input("難易度 (空白でスキップ): ") or None
            
            results = analyzer.search_qa(query, category, difficulty)
            print(f"\n検索結果: {len(results)}件")
            for i, qa in enumerate(results[:5], 1):
                print(f"\n{i}. [{qa['category']}] {qa['question']}")
                print(f"   {qa['answer'][:100]}...")
        
        elif command == 'similar':
            target = input("類似質問を探したい質問: ")
            similar = analyzer.find_similar_questions(target)
            print(f"\n類似質問:")
            for qa, similarity in similar:
                print(f"類似度 {similarity:.3f}: {qa['question']}")
        
        elif command == 'vectorize':
            text = input("ベクトル化したい文章: ")
            vectors = analyzer.vectorize_words(text)
            if vectors:
                print("\n単語ベクトル (最初の5件):")
                for i, (word, vector) in enumerate(vectors.items()):
                    if i >= 5:
                        break
                    print(f"  - {word}: [ベクトル次元: {vector.shape}] - {vector[:4]}...")
            else:
                print("ベクトルを生成できませんでした。")

        elif command == 'validate':
            validation = analyzer.validate_data()
            print(f"\nデータ品質スコア: {validation['quality_score']}/100")
            print(f"問題数: {validation['total_issues']}")
            if validation['issues']:
                print("問題:")
                for issue in validation['issues'][:10]:
                    print(f"  - {issue}")
        
        else:
            print("無効なコマンドです")

if __name__ == "__main__":
    print("TYES Trivia AI 分析ツールを開始します...")
    interactive_analyzer()