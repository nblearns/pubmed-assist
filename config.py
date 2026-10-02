import os

# Models are overridable via environment so evals can compare them.
MODEL = os.getenv("PUBMED_ASSIST_MODEL", "claude-sonnet-4-6")
CRITIC_MODEL = os.getenv("PUBMED_ASSIST_CRITIC_MODEL", MODEL)

# Safety caps: stop runaway loops (and runaway cost).
MAX_TOOL_ROUNDS = 5   # researcher tool-call round trips
MAX_REVISIONS = 2     # critic -> revise cycles before unsupported claims are stripped
