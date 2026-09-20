# PubMedAssist

An agentic AI application that answers biomedical research questions by retrieving and citing real papers from PubMed — not from the model's training knowledge.

## How it works

The system uses a **researcher + critic multi-agent pattern**:

1. **Researcher agent** — takes your question, forms PubMed search terms, calls the PubMed API, reads the returned abstracts, and drafts an answer with inline citations (e.g. `Smith et al., 2023`).
2. **PubMed search tool** — calls NCBI's E-utilities API (`esearch` + `efetch`) to retrieve title, authors, journal, year, and abstract for matching papers.
3. **Critic agent** *(in progress)* — will review the researcher's draft against the retrieved abstracts and remove any claim not directly supported by them.

Every answer includes a medical disclaimer and a count of papers retrieved.

## Key agentic concepts demonstrated

| Concept | Where |
|---|---|
| Tool use / function calling | `agents/researcher.py` — agent decides when to call `search_pubmed` |
| Retrieval-augmented generation (RAG) | Answers grounded strictly in retrieved abstracts |
| Hallucination avoidance | System prompt enforces "retrieved abstracts only" rule |
| Multi-agent orchestration | Researcher → Critic pipeline |
| Agentic loop | `while True` loop handles multi-turn tool calls until `end_turn` |

## Project structure

```
pubmed-assist/
├── main.py                 # CLI entry point
├── agents/
│   ├── researcher.py       # Researcher agent with tool-calling loop
│   └── critic.py           # Critic agent (stub, in progress)
├── tools/
│   └── pubmed.py           # PubMed E-utilities API wrapper
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/nblearns/pubmed-assist.git
cd pubmed-assist
pip install -r requirements.txt
```

**2. Set your Anthropic API key**
```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=your_key_here
```

**3. Run**
```bash
python main.py
```

## Example

```
PubMedAssist — answers grounded in PubMed literature
Type your question, or 'quit' to exit.

Question: What are the effects of sleep deprivation on cognitive performance?

  [tool] search_pubmed({'query': 'sleep deprivation cognitive performance', 'max_results': 5})

Sleep deprivation consistently impairs attention, working memory, and executive
function. Lim & Dinges (2010) found that even moderate sleep restriction
accumulates cognitive deficits equivalent to total sleep deprivation...

---
DISCLAIMER: This is a research-summary tool, not medical advice. It does not
replace consultation with a qualified healthcare professional.

[5 paper(s) retrieved from PubMed]
```

## Tech stack

- [Claude API](https://docs.anthropic.com/) (Anthropic) — `claude-sonnet-4-6` as the agent model
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) — free public API, no key required
- Python 3.10+

## Disclaimer

This tool is for research summarisation only. It is not medical advice and does not replace consultation with a qualified healthcare professional.
