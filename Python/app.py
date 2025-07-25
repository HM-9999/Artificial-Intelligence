from flask import Flask, request, render_template_string
from keio_qa_analyzer import KeioQAAnalyzer  # 先ほどのクラスを別ファイルに保存しておく

app = Flask(__name__)
analyzer = KeioQAAnalyzer()

# シンプルなHTMLテンプレート（フォーム + 結果表示）
HTML = """
<!doctype html>
<html>
<head>
  <title>慶應義塾についてお聞きください</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; }
    textarea { width: 100%; height: 80px; }
    .answer { margin-top: 1em; padding: 1em; background: #f0f0f0; border-radius: 5px; }
  </style>
</head>
<body>
  <h1>慶應義塾についてお聞きください</h1>
  <form method="post">
    <label for="question">質問を入力してください：</label><br>
    <textarea name="question" id="question">{{ question or "" }}</textarea><br>
    <button type="submit">質問する</button>
  </form>

  {% if answers %}
    <h2>回答</h2>
    {% for qa in answers %}
      <div class="answer">
        {{ qa['answer'] }}
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    question = None
    answers = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            # キーワード検索で最初の1件のみ
            search_results = analyzer.search_qa(question)
            if search_results:
                answers = [search_results[0]]
            else:
                # 類似質問の最上位1件のみ
                similar = analyzer.find_similar_questions(question)
                if similar:
                    answers = [similar[0][0]]
                else:
                    answers = []
    return render_template_string(HTML, question=question, answers=answers)

if __name__ == "__main__":
    app.run(debug=True)
