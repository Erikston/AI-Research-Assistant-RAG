from flask import Flask, render_template, request, jsonify
from rag_pipeline import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():

    question = request.json["question"]

    try:
        answer = ask_question(question)
    except Exception as e:
        answer = str(e)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)