import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple
# 追加: ベクトル化用
from sentence_transformers import SentenceTransformer
import numpy as np
# 追加: janomeによる形態素解析
from janome.tokenizer import Tokenizer

# 類義語・関連語の辞書
SYNONYM_DICT = {
    "部活": ["クラブ", "サークル", "課外活動"],
    "慶應": ["慶應義塾", "慶應大学", "Keio"],
    # 必要に応じて追加
}

def tokenize(text):
    t = Tokenizer()
    return [token.surface for token in t.tokenize(text)]

def expand_words(words, synonym_dict):
    expanded = set(words)
    for word in words:
        if word in synonym_dict:
            expanded.update(synonym_dict[word])
    return list(expanded)

class KeioQAAnalyzer:
    """慶應義塾Q&Aデータの分析・検索・実行ツール"""
    
    def __init__(self, data_file: str = "keio_qa_dataset.json"):
        self.data_file = data_file
        self.qa_data = []
        self.meta_info = {}
        self.load_data()
        # 追加: モデルとベクトル化（ローカル保存モデルを利用）
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2-local')  # ローカルパスに変更
        self.question_embeddings = self._embed_all_questions()
    
    def load_data(self):
        """Q&Aデータファイルを読み込む"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.qa_data = data.get('qa_pairs', [])
                self.meta_info = data.get('meta', {})
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
        """全質問文をベクトル化して保存"""
        if not self.qa_data:
            return np.array([])
        questions = [qa['question'] for qa in self.qa_data]
        embeddings = self.model.encode(questions, convert_to_numpy=True)
        return embeddings

    def get_basic_stats(self) -> Dict:
        """基本統計情報を取得"""
        if not self.qa_data:
            return {"error": "データがありません"}
        
        categories = Counter(qa['category'] for qa in self.qa_data)
        difficulties = Counter(qa['difficulty'] for qa in self.qa_data)
        
        # タグの統計
        all_tags = []
        for qa in self.qa_data:
            all_tags.extend(qa.get('tags', []))
        tag_counts = Counter(all_tags)
        
        # 質問文字数の統計
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
    
    def search_qa(self, query: str, category: str = None, 
                  difficulty: str = None, tags: List[str] = None) -> List[Dict]:
        """Q&Aを検索（単語分割＋類義語拡張対応）"""
        results = []
        # 追加: クエリを単語分割＋拡張
        if query:
            words = tokenize(query)
            expanded_words = expand_words(words, SYNONYM_DICT)
            expanded_query = " ".join(expanded_words)
            query_lower = expanded_query.lower()
        else:
            query_lower = ""
        
        for qa in self.qa_data:
            # キーワード検索
            if query and query_lower not in qa['question'].lower() and query_lower not in qa['answer'].lower():
                continue
            
            # カテゴリフィルタ
            if category and qa['category'] != category:
                continue
            
            # 難易度フィルタ
            if difficulty and qa['difficulty'] != difficulty:
                continue
            
            # タグフィルタ
            if tags:
                qa_tags = qa.get('tags', [])
                if not any(tag in qa_tags for tag in tags):
                    continue
            
            results.append(qa)
        
        return results
    
    def find_similar_questions(self, target_question: str, limit: int = 5) -> List[Tuple[Dict, float]]:
        """ベクトル化による意味ベースの類似質問検索（単語分割＋類義語拡張対応）"""
        if not self.qa_data or self.question_embeddings.shape[0] == 0:
            return []
        # 追加: クエリを単語分割＋拡張
        words = tokenize(target_question)
        expanded_words = expand_words(words, SYNONYM_DICT)
        expanded_question = " ".join(expanded_words)
        # 入力質問をベクトル化
        target_emb = self.model.encode([expanded_question], convert_to_numpy=True)[0]
        # コサイン類似度計算
        similarities = np.dot(self.question_embeddings, target_emb) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(target_emb) + 1e-8)
        # 上位N件を取得
        top_idx = similarities.argsort()[::-1][:limit]
        results = [(self.qa_data[i], float(similarities[i])) for i in top_idx]
        return results
    
    def get_category_analysis(self) -> Dict:
        """カテゴリ別の詳細分析"""
        analysis = defaultdict(lambda: {
            "count": 0,
            "difficulties": Counter(),
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "common_tags": Counter()
        })
        
        for qa in self.qa_data:
            cat = qa['category']
            analysis[cat]["count"] += 1
            analysis[cat]["difficulties"][qa['difficulty']] += 1
            analysis[cat]["avg_question_length"] += len(qa['question'])
            analysis[cat]["avg_answer_length"] += len(qa['answer'])
            
            for tag in qa.get('tags', []):
                analysis[cat]["common_tags"][tag] += 1
        
        # 平均値計算
        for cat_data in analysis.values():
            if cat_data["count"] > 0:
                cat_data["avg_question_length"] = round(cat_data["avg_question_length"] / cat_data["count"], 1)
                cat_data["avg_answer_length"] = round(cat_data["avg_answer_length"] / cat_data["count"], 1)
                cat_data["difficulties"] = dict(cat_data["difficulties"])
                cat_data["common_tags"] = dict(cat_data["common_tags"].most_common(5))
        
        return dict(analysis)
    
    def get_qa_by_id(self, qa_id: int) -> Optional[Dict]:
        """IDでQ&Aを取得"""
        for qa in self.qa_data:
            if qa['id'] == qa_id:
                return qa
        return None
    
    def export_filtered_data(self, filter_criteria: Dict, output_file: str):
        """フィルタ条件に基づいてデータをエクスポート"""
        filtered_data = self.qa_data
        
        # フィルタ適用
        if filter_criteria.get('category'):
            filtered_data = [qa for qa in filtered_data if qa['category'] == filter_criteria['category']]
        
        if filter_criteria.get('difficulty'):
            filtered_data = [qa for qa in filtered_data if qa['difficulty'] == filter_criteria['difficulty']]
        
        if filter_criteria.get('tags'):
            target_tags = filter_criteria['tags']
            filtered_data = [qa for qa in filtered_data 
                           if any(tag in qa.get('tags', []) for tag in target_tags)]
        
        # エクスポート
        export_data = {
            "meta": {
                "exported_at": datetime.now().isoformat(),
                "filter_criteria": filter_criteria,
                "total_count": len(filtered_data),
                "source_file": self.data_file
            },
            "qa_pairs": filtered_data
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"フィルタされたデータを {output_file} にエクスポートしました ({len(filtered_data)}件)")
            return True
        except Exception as e:
            print(f"エクスポートエラー: {e}")
            return False
    
    def create_training_data(self, output_file: str = "training_data.jsonl"):
        """OpenRouter用のトレーニングデータ形式で出力"""
        training_data = []
        
        for qa in self.qa_data:
            # システムプロンプト
            system_prompt = ("あなたは慶應義塾大学に関する質問に答える専門的なAIアシスタントです。"
                           "正確で詳細な情報を提供し、慶應義塾の歴史、理念、制度について深い知識を持っています。")
            
            # ユーザーの質問
            user_message = qa['question']
            
            # アシスタントの回答
            assistant_message = qa['answer']
            
            # JSONL形式のデータ
            training_item = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ],
                "metadata": {
                    "category": qa['category'],
                    "difficulty": qa['difficulty'],
                    "tags": qa.get('tags', []),
                    "id": qa['id']
                }
            }
            
            training_data.append(training_item)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"トレーニングデータを {output_file} に出力しました ({len(training_data)}件)")
            return True
        except Exception as e:
            print(f"トレーニングデータ出力エラー: {e}")
            return False
    
    def validate_data(self) -> Dict:
        """データの品質チェック"""
        issues = []
        
        for qa in self.qa_data:
            # 必須フィールドチェック
            required_fields = ['id', 'question', 'answer', 'category']
            for field in required_fields:
                if not qa.get(field):
                    issues.append(f"ID {qa.get('id', 'Unknown')}: {field} が空です")
            
            # 質問・回答の長さチェック
            if qa.get('question') and len(qa['question']) < 10:
                issues.append(f"ID {qa['id']}: 質問が短すぎます ({len(qa['question'])}文字)")
            
            if qa.get('answer') and len(qa['answer']) < 20:
                issues.append(f"ID {qa['id']}: 回答が短すぎます ({len(qa['answer'])}文字)")
            
            # 重複IDチェック
            ids = [qa['id'] for qa in self.qa_data]
            duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
            if duplicate_ids:
                issues.append(f"重複するID: {duplicate_ids}")
        
        return {
            "total_issues": len(issues),
            "issues": issues,
            "quality_score": max(0, 100 - len(issues) * 5)  # 簡単な品質スコア
        }

def interactive_analyzer():
    """対話式分析ツール"""
    analyzer = KeioQAAnalyzer()
    
    if not analyzer.qa_data:
        print("Q&Aデータが読み込めませんでした。keio_qa_data.json ファイルを確認してください。")
        return
    
    print("=== 慶應義塾 Q&A 分析ツール ===")
    print("コマンド: stats, search, similar, category, export, training, validate, quit")
    
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
                print(f"類似度 {similarity:.2f}: {qa['question']}")
        
        elif command == 'category':
            analysis = analyzer.get_category_analysis()
            for cat, data in analysis.items():
                print(f"\n=== {cat} ===")
                print(f"件数: {data['count']}")
                print(f"平均質問文字数: {data['avg_question_length']}")
                print(f"難易度分布: {data['difficulties']}")
        
        elif command == 'export':
            category = input("エクスポートするカテゴリ (空白で全て): ") or None
            filename = input("出力ファイル名: ") or "exported_qa.json"
            criteria = {}
            if category:
                criteria['category'] = category
            analyzer.export_filtered_data(criteria, filename)
        
        elif command == 'training':
            filename = input("トレーニングデータファイル名 (デフォルト: training_data.jsonl): ") or "training_data.jsonl"
            analyzer.create_training_data(filename)
        
        elif command == 'validate':
            validation = analyzer.validate_data()
            print(f"\nデータ品質スコア: {validation['quality_score']}/100")
            print(f"問題数: {validation['total_issues']}")
            if validation['issues']:
                print("問題:")
                for issue in validation['issues'][:10]:  # 最初の10件表示
                    print(f"  - {issue}")
        
        else:
            print("無効なコマンドです")

if __name__ == "__main__":
    print("慶應義塾Q&A分析ツールを開始します...")
    interactive_analyzer()