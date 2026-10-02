import json

from anthropic import Anthropic
from config import MAX_TOOL_ROUNDS, MODEL
from tools.pubmed import search_pubmed

client = Anthropic()

SYSTEM_PROMPT = """\
You are a biomedical research assistant. Answer the user's question using ONLY \
information from PubMed abstracts you retrieve with the search_pubmed tool.

Rules:
- Always call search_pubmed before answering. You may call it more than once \
  with different queries if the first results are insufficient.
- Answer strictly from the retrieved abstracts. Never fill gaps with your \
  training knowledge.
- If the abstracts do not address the question, say so explicitly — do not guess.
- Cite every factual claim inline: (LastName et al., Year) or (LastName, Year) \
  for single-author papers.
- Write clearly for an educated non-specialist reader.\
"""

TOOL_DEFINITION = {
  "name": "search_pubmed",
  "description": (
    "Search PubMed for peer-reviewed papers. "
    "Returns title, authors, journal, year, and abstract for each paper."
  ),
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": (
          "PubMed search query. Use relevant medical/scientific keywords "
          "and MeSH terms where appropriate."
        ),
      },
      "max_results": {
        "type": "integer",
        "description": "Number of papers to retrieve (1–10). Default: 5.",
        "default": 5,
      },
    },
    "required": ["query"],
  },
}


def _dedupe(papers: list[dict]) -> list[dict]:
  """Keep the first occurrence of each PMID (repeat searches overlap)."""
  seen: set[str] = set()
  unique = []
  for paper in papers:
    if paper["pmid"] not in seen:
      seen.add(paper["pmid"])
      unique.append(paper)
  return unique


def run_researcher(question: str) -> tuple[str, list[dict]]:
  """
  Run the researcher agent against the user's question.
  Returns (draft_answer, papers_used).
  """
  messages = [{"role": "user", "content": question}]
  papers_used: list[dict] = []

  for _ in range(MAX_TOOL_ROUNDS):
    response = client.messages.create(
      model=MODEL,
      max_tokens=2048,
      system=SYSTEM_PROMPT,
      tools=[TOOL_DEFINITION],
      messages=messages,
    )

    # Add assistant turn to history so the loop has full context
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "tool_use":
      tool_results = []
      for block in response.content:
        if block.type != "tool_use":
          continue
        if block.name == "search_pubmed":
          print(f"  [tool] search_pubmed({block.input})")
          papers = search_pubmed(**block.input)
          papers_used.extend(papers)
          content = json.dumps(papers)
          is_error = False
        else:
          # Every tool_use needs a matching tool_result or the API rejects the next call
          content = f"Unknown tool: {block.name}"
          is_error = True
        tool_results.append({
          "type": "tool_result",
          "tool_use_id": block.id,
          "content": content,
          "is_error": is_error,
        })
      messages.append({"role": "user", "content": tool_results})

    elif response.stop_reason == "end_turn":
      draft = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "",
      )
      return draft, _dedupe(papers_used)

    else:
      # max_tokens, refusal, etc. — fail loudly instead of looping forever
      raise RuntimeError(f"Researcher stopped unexpectedly: {response.stop_reason}")

  raise RuntimeError(f"Researcher exceeded {MAX_TOOL_ROUNDS} tool rounds without answering")


REVISE_PROMPT = """\
You are revising a draft answer that a reviewer found partly unsupported.

Rules:
- Use ONLY the papers provided below. Add no knowledge of your own and do not search.
- Fix or remove every flagged claim. Do not re-introduce them in other words.
- Keep claims that were not flagged, and keep inline citations: (LastName et al., Year).
- Return only the revised answer text.\
"""


def revise_answer(
  question: str,
  draft: str,
  papers: list[dict],
  flagged: list[dict],
  remove_only: bool = False,
) -> str:
  """Rewrite the draft so flagged claims are fixed (or, with remove_only, deleted)."""
  instruction = (
    "Delete each flagged claim entirely. Change nothing else."
    if remove_only
    else "Rewrite each flagged claim so the papers support it, or delete it if they cannot."
  )
  user = (
    f"<question>{question}</question>\n"
    f"<papers>{json.dumps(papers)}</papers>\n"
    f"<draft>{draft}</draft>\n"
    f"<flagged_claims>{json.dumps(flagged)}</flagged_claims>\n"
    f"{instruction}"
  )
  response = client.messages.create(
    model=MODEL,
    max_tokens=2048,
    system=REVISE_PROMPT,
    messages=[{"role": "user", "content": user}],
  )
  return next((b.text for b in response.content if hasattr(b, "text")), draft)
