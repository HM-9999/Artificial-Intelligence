import json
import re
from typing import List, Dict, Set
from collections import defaultdict
import time

class KeioQAAnalyzerOptimized:
    def __init__(self, data_file: str = "keio_qa_dataset.json"):
        self.data_file = data_file
        self.qa_data = []
        self.search_index = {}  # 高速検索用インデックス
        self.word_to_qa = defaultdict(set)  # 単語 → QAインデックス
        self.qa_scores = {}  # QAスコアキャッシュ
        self.load_data()
        self.build_optimized_index()
    
    def load_data(self):
        """データの読み込み"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.qa_data = data.get('qa_pairs', [])
            print(f"データ読み込み完了: {len(self.qa_data)}件のQ&A")
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            self.qa_data = []

    def tokenize(self, text: str) -> Set[str]:
        """高速トークン化"""
        # 正規表現で高速分割
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if len(word) > 1}

    def build_optimized_index(self):
        """最適化されたインデックス構築"""
        print("最適化インデックスを構築中...")
        start_time = time.time()
        
        for i, qa in enumerate(self.qa_data):
            # 質問と回答をトークン化
            question_tokens = self.tokenize(qa['question'])
            answer_tokens = self.tokenize(qa['answer'])
            
            # 単語インデックスに追加
            for token in question_tokens | answer_tokens:
                self.word_to_qa[token].add(i)
            
            # 完全一致検索用インデックス
            self.search_index[qa['question'].lower()] = i
            self.search_index[qa['answer'].lower()] = i
        
        build_time = time.time() - start_time
        print(f"最適化インデックス構築完了: {len(self.word_to_qa)}個の単語, {build_time:.3f}秒")

    def fast_search(self, query: str) -> List[Dict]:
        """最適化された高速検索"""
        if not query or not self.qa_data:
            return []
        
        start_time = time.time()
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        # 候補QAの収集
        candidates = set()
        for token in query_tokens:
            if token in self.word_to_qa:
                candidates.update(self.word_to_qa[token])
        
        # スコア計算とソート
        results = []
        for idx in candidates:
            qa = self.qa_data[idx]
            score = self.calculate_score(query_tokens, qa)
            if score > 0:
                results.append((qa, score))
        
        # 上位3件を返す
        results.sort(key=lambda x: x[1], reverse=True)
        search_time = time.time() - start_time
        print(f"検索時間: {search_time:.4f}秒")
        
        return [qa for qa, score in results[:3]]

    def calculate_score(self, query_tokens: Set[str], qa: Dict) -> float:
        """高速スコア計算"""
        question_tokens = self.tokenize(qa['question'])
        answer_tokens = self.tokenize(qa['answer'])
        
        # 質問マッチング（重み2）
        question_score = len(query_tokens & question_tokens) * 2
        
        # 回答マッチング（重み1）
        answer_score = len(query_tokens & answer_tokens)
        
        return question_score + answer_score

    def find_similar_questions(self, target_question: str, limit: int = 3) -> List[tuple]:
        """最適化された類似質問検索"""
        if not target_question or not self.qa_data:
            return []
        
        start_time = time.time()
        target_tokens = self.tokenize(target_question)
        
        if not target_tokens:
            return []
        
        # 候補QAの収集
        candidates = set()
        for token in target_tokens:
            if token in self.word_to_qa:
                candidates.update(self.word_to_qa[token])
        
        similarities = []
        for idx in candidates:
            qa = self.qa_data[idx]
            question_tokens = self.tokenize(qa['question'])
            
            if target_tokens and question_tokens:
                overlap = len(target_tokens & question_tokens)
                similarity = overlap / len(target_tokens | question_tokens)
                
                if similarity > 0.1:
                    similarities.append((qa, similarity))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        search_time = time.time() - start_time
        print(f"類似検索時間: {search_time:.4f}秒")
        
        return similarities[:limit]

    def get_stats(self) -> Dict:
        """統計情報"""
        return {
            "総Q&A数": len(self.qa_data),
            "インデックス単語数": len(self.word_to_qa),
            "検索インデックス数": len(self.search_index)
        } 