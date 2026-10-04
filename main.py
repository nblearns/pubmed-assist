import sys

from dotenv import load_dotenv

load_dotenv()

from pipeline import run_pipeline  # noqa: E402 — load_dotenv must run first


def main():
  print("PubMedAssist — answers grounded in PubMed literature")
  print("Type your question, or 'quit' to exit.\n")

  while True:
    try:
      question = input("Question: ").strip()
    except (EOFError, KeyboardInterrupt):
      break

    if question.lower() in ("quit", "exit", "q", ""):
      break

    print("\nSearching PubMed...\n")
    try:
      result = run_pipeline(question)
      print(f"\n{result['answer']}")
      print(f"\n[{len(result['papers'])} paper(s) retrieved from PubMed]")
    except Exception as exc:
      print(f"Error: {exc}", file=sys.stderr)
    print()


if __name__ == "__main__":
  main()
