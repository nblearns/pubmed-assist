"""Offline tests: no network, no API key. The Anthropic client is mocked."""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
  import anthropic  # noqa: F401
except ImportError:  # allow running without the SDK installed
  stub = types.ModuleType("anthropic")
  stub.Anthropic = MagicMock
  sys.modules["anthropic"] = stub

import pipeline  # noqa: E402
from agents import critic  # noqa: E402

PAPERS = [{
  "pmid": "1", "title": "Sleep and memory", "authors": "Lim J et al.", "journal": "Sleep",
  "year": "2010", "abstract": "Sleep restriction  impairs working memory\nin healthy adults.",
}]


def tool_use(claims):
  block = MagicMock(type="tool_use")
  block.input = {"claims": claims}
  return MagicMock(content=[block], stop_reason="tool_use")


def claim(verdict, pmid="1", quote="impairs working memory"):
  return {"claim": "c", "cited_pmid": pmid, "verdict": verdict, "evidence_quote": quote}


class QuoteTests(unittest.TestCase):
  def test_whitespace_and_case_insensitive(self):
    self.assertTrue(critic.quote_in_abstract("IMPAIRS working memory in healthy", PAPERS[0]["abstract"]))

  def test_ellipsis_fragments(self):
    self.assertTrue(critic.quote_in_abstract("Sleep restriction ... healthy adults", PAPERS[0]["abstract"]))

  def test_fabricated_quote_rejected(self):
    self.assertFalse(critic.quote_in_abstract("improves memory", PAPERS[0]["abstract"]))


class CriticTests(unittest.TestCase):
  def review(self, claims):
    with patch.object(critic, "client") as c:
      c.messages.create.return_value = tool_use(claims)
      return critic.review_answer("draft", PAPERS)

  def test_pass(self):
    self.assertEqual(self.review([claim("supported")])["overall"], "pass")

  def test_fake_quote_is_downgraded(self):
    r = self.review([claim("supported", quote="totally invented")])
    self.assertEqual(r["overall"], "revise")
    self.assertEqual(r["flagged"][0]["verdict"], "unsupported")

  def test_unretrieved_pmid_is_downgraded(self):
    self.assertEqual(self.review([claim("supported", pmid="999")])["overall"], "revise")

  def test_unsupported_flagged(self):
    self.assertEqual(len(self.review([claim("unsupported", quote="")])["flagged"]), 1)


class PipelineTests(unittest.TestCase):
  def run_pipeline(self, reviews, papers=PAPERS):
    with patch.object(pipeline, "run_researcher", return_value=("draft", papers)), \
         patch.object(pipeline, "review_answer", side_effect=reviews) as rv, \
         patch.object(pipeline, "revise_answer", return_value="revised") as rev:
      out = pipeline.run_pipeline("q")
    return out, rv, rev

  def test_pass_first_time(self):
    out, rv, rev = self.run_pipeline([{"overall": "pass", "claims": [], "flagged": []}])
    self.assertEqual(rev.call_count, 0)
    self.assertIn("DISCLAIMER", out["answer"])
    self.assertIn("PMID 1", out["answer"])

  def test_revise_then_pass(self):
    bad = {"overall": "revise", "claims": [], "flagged": [claim("unsupported")]}
    good = {"overall": "pass", "claims": [], "flagged": []}
    out, _, rev = self.run_pipeline([bad, good])
    self.assertEqual(rev.call_count, 1)
    self.assertFalse(out["stripped"])

  def test_strip_after_max_revisions(self):
    bad = {"overall": "revise", "claims": [], "flagged": [claim("unsupported")]}
    out, rv, rev = self.run_pipeline([bad, bad, bad])
    self.assertEqual(rv.call_count, 3)
    self.assertEqual(rev.call_count, 3)  # 2 rewrites + 1 remove-only
    self.assertTrue(rev.call_args.kwargs["remove_only"])
    self.assertTrue(out["stripped"])
    self.assertIn("removed", out["answer"])

  def test_no_papers_skips_critic(self):
    out, rv, _ = self.run_pipeline([], papers=[])
    self.assertEqual(rv.call_count, 0)
    self.assertIn("DISCLAIMER", out["answer"])


if __name__ == "__main__":
  unittest.main()
