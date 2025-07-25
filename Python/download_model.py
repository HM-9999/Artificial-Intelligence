from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model.save('paraphrase-multilingual-MiniLM-L12-v2-local')