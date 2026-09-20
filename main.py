import sys

from dotenv import load_dotenv

load_dotenv()

from agents.researcher import run_researcher  # noqa: E402 — load_dotenv must run first

DISCLAIMER = (
  "\n---\n"
  "DISCLAIMER: This is a research-summary tool, not medical advice. "
  "It does not replace consultation with a qualified healthcare professional."
)


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
      answer, papers = run_researcher(question)
      print(f"\n{answer}")
      print(DISCLAIMER)
      print(f"\n[{len(papers)} paper(s) retrieved from PubMed]")
    except Exception as exc:
      print(f"Error: {exc}", file=sys.stderr)
    print()


if __name__ == "__main__":
  main()
