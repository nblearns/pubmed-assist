import os
from pathlib import Path


def _read_api_key() -> str:
  """Read ANTHROPIC_API_KEY from the .env next to this file, then fall back to os.environ."""
  env_file = Path(__file__).parent / ".env"
  try:
    for line in env_file.read_text(encoding="utf-8").splitlines():
      if line.startswith("ANTHROPIC_API_KEY="):
        return line.split("=", 1)[1].strip()
  except FileNotFoundError:
    pass
  return os.environ.get("ANTHROPIC_API_KEY", "")


ANTHROPIC_API_KEY = _read_api_key()

# Models are overridable via environment so evals can compare them.
MODEL = os.getenv("PUBMED_ASSIST_MODEL", "claude-haiku-4-5-20251001")
CRITIC_MODEL = os.getenv("PUBMED_ASSIST_CRITIC_MODEL", MODEL)

# Safety caps: stop runaway loops (and runaway cost).
MAX_TOOL_ROUNDS = 5   # researcher tool-call round trips
MAX_REVISIONS = 2     # critic -> revise cycles before unsupported claims are stripped
