import json

from anthropic import Anthropic
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


def run_researcher(question: str) -> tuple[str, list[dict]]:
  """
  Run the researcher agent against the user's question.
  Returns (draft_answer, papers_used).
  """
  messages = [{"role": "user", "content": question}]
  papers_used: list[dict] = []

  while True:
    response = client.messages.create(
      model="claude-sonnet-4-6",
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
        if block.type == "tool_use" and block.name == "search_pubmed":
          print(f"  [tool] search_pubmed({block.input})")
          papers = search_pubmed(**block.input)
          papers_used.extend(papers)
          tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(papers),
          })
      messages.append({"role": "user", "content": tool_results})

    elif response.stop_reason == "end_turn":
      draft = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "",
      )
      return draft, papers_used
