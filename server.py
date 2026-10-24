from dotenv import load_dotenv

load_dotenv()  # must run before pipeline imports so ANTHROPIC_API_KEY is set

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from pipeline import run_pipeline  # noqa: E402

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
  return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
  data = request.get_json(silent=True) or {}
  question = data.get("question", "").strip()
  if not question:
    return jsonify({"error": "No question provided"}), 400
  try:
    result = run_pipeline(question)
    return jsonify({
      "answer_body": result["answer_body"],
      "papers": result["papers"],
      "stripped": result["stripped"],
    })
  except Exception as exc:
    return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
  print("PubMedAssist — http://localhost:5000")
  app.run(port=5000)
