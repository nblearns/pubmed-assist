import json
import re

from anthropic import Anthropic
from config import CRITIC_MODEL

client = Anthropic()

CRITIC_SYSTEM_PROMPT = """\
You are a strict fact-checking reviewer. You receive a draft answer and the \
PubMed papers it was supposed to be based on. You have no other knowledge source.

For every factual claim in the draft:
- Find the paper (by PMID) the claim relies on.
- Decide a verdict:
  - supported: the abstract directly states or clearly entails the claim.
  - unsupported: no provided abstract supports it (even if it is true in general).
  - miscited: the claim is supported by a different paper than the one cited, \
or the cited paper says something different or weaker.
- For supported claims, copy a short VERBATIM quote from that abstract as evidence.

Rules:
- Judge only against the provided abstracts, never your own knowledge.
- Overstated certainty (e.g. "proves" when the abstract says "suggests") is unsupported.
- A statement that the abstracts do not answer the question is not a claim to check.
- Always respond by calling submit_review.\
"""

REVIEW_TOOL = {
  "name": "submit_review",
  "description": "Submit the claim-by-claim review of the draft answer.",
  "input_schema": {
    "type": "object",
    "properties": {
      "claims": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "claim": {"type": "string", "description": "The claim, as stated in the draft."},
            "cited_pmid": {"type": "string", "description": "PMID of the paper the claim relies on, or empty."},
            "verdict": {"type": "string", "enum": ["supported", "unsupported", "miscited"]},
            "evidence_quote": {"type": "string", "description": "Verbatim quote from that abstract. Empty if not supported."},
          },
          "required": ["claim", "cited_pmid", "verdict", "evidence_quote"],
        },
      },
    },
    "required": ["claims"],
  },
}


def _normalize(text: str) -> str:
  return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_abstract(quote: str, abstract: str) -> bool:
  """True if every fragment of the quote (split on ellipses) appears in the abstract."""
  haystack = _normalize(abstract)
  fragments = [_normalize(f).strip(" \"'") for f in re.split(r"\.\.\.|…", quote)]
  fragments = [f for f in fragments if f]
  return bool(fragments) and all(f in haystack for f in fragments)


def _verify(claims: list[dict], papers: list[dict]) -> list[dict]:
  """Don't trust the critic blindly: a 'supported' verdict must carry a quote that really exists."""
  abstracts = {p["pmid"]: p["abstract"] for p in papers}
  checked = []
  for claim in claims:
    claim = dict(claim)
    if claim["verdict"] == "supported":
      abstract = abstracts.get(claim["cited_pmid"])
      if abstract is None:
        claim["verdict"], claim["note"] = "unsupported", "cited PMID was not retrieved"
      elif not quote_in_abstract(claim["evidence_quote"], abstract):
        claim["verdict"], claim["note"] = "unsupported", "evidence quote not found in abstract"
    checked.append(claim)
  return checked


def review_answer(draft: str, papers: list[dict]) -> dict:
  """
  Review a draft against the retrieved papers.
  Returns {"overall": "pass" | "revise", "claims": [...], "flagged": [...]}.
  """
  user = f"<papers>{json.dumps(papers)}</papers>\n<draft>{draft}</draft>"
  response = client.messages.create(
    model=CRITIC_MODEL,
    max_tokens=4096,
    system=CRITIC_SYSTEM_PROMPT,
    tools=[REVIEW_TOOL],
    tool_choice={"type": "tool", "name": "submit_review"},  # forces structured output
    messages=[{"role": "user", "content": user}],
  )
  block = next((b for b in response.content if b.type == "tool_use"), None)
  if block is None:
    raise RuntimeError(f"Critic returned no review (stop_reason={response.stop_reason})")

  claims = _verify(block.input.get("claims", []), papers)
  flagged = [c for c in claims if c["verdict"] != "supported"]
  # Computed in code, not taken from the model.
  return {"overall": "revise" if flagged else "pass", "claims": claims, "flagged": flagged}
