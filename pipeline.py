from agents.critic import review_answer
from agents.researcher import revise_answer, run_researcher
from config import MAX_REVISIONS

DISCLAIMER = (
  "\n---\n"
  "DISCLAIMER: This is a research-summary tool, not medical advice. "
  "It does not replace consultation with a qualified healthcare professional."
)


def _sources(papers: list[dict]) -> str:
  if not papers:
    return ""
  lines = [f"- {p['authors']} ({p['year']}). {p['title']} PMID {p['pmid']}" for p in papers]
  return "\n\nPapers retrieved:\n" + "\n".join(lines)


def run_pipeline(question: str) -> dict:
  """
  question -> researcher draft -> critic review -> revise (max MAX_REVISIONS) -> final.
  Returns {"answer", "papers", "reviews", "stripped"}; reviews is kept for evals.
  """
  answer, papers = run_researcher(question)
  reviews: list[dict] = []
  stripped = False

  # Nothing retrieved means nothing to verify against; the researcher must already say so.
  if papers:
    for attempt in range(MAX_REVISIONS + 1):
      review = review_answer(answer, papers)
      reviews.append(review)
      if review["overall"] == "pass":
        break
      if attempt == MAX_REVISIONS:
        # Revisions exhausted: delete what is still unsupported rather than ship it.
        answer = revise_answer(question, answer, papers, review["flagged"], remove_only=True)
        stripped = True
        break
      print(f"  [critic] {len(review['flagged'])} unsupported claim(s); revising ({attempt + 1}/{MAX_REVISIONS})")
      answer = revise_answer(question, answer, papers, review["flagged"])

  note = (
    "\n\nNote: some points could not be verified against the retrieved abstracts and were removed."
    if stripped
    else ""
  )
  return {
    "answer": f"{answer}{note}{_sources(papers)}{DISCLAIMER}",
    "papers": papers,
    "reviews": reviews,
    "stripped": stripped,
  }
