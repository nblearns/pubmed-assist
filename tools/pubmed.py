import xml.etree.ElementTree as ET
from typing import Optional

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
  """Search PubMed and return structured paper metadata + abstracts."""
  pmids = _esearch(query, max_results)
  if not pmids:
    return []
  return _efetch(pmids)


def _esearch(query: str, max_results: int) -> list[str]:
  params = {
    "db": "pubmed",
    "term": query,
    "retmax": max_results,
    "retmode": "json",
  }
  resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=10)
  resp.raise_for_status()
  return resp.json()["esearchresult"]["idlist"]


def _efetch(pmids: list[str]) -> list[dict]:
  params = {
    "db": "pubmed",
    "id": ",".join(pmids),
    "rettype": "abstract",
    "retmode": "xml",
  }
  resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=15)
  resp.raise_for_status()
  return _parse_articles(resp.text)


def _parse_articles(xml_text: str) -> list[dict]:
  root = ET.fromstring(xml_text)
  return [_parse_article(a) for a in root.findall(".//PubmedArticle")]


def _parse_article(article: ET.Element) -> dict:
  medline = article.find("MedlineCitation")
  art = medline.find("Article")

  pmid = medline.findtext("PMID", "")
  title = art.findtext("ArticleTitle", "")
  journal_el = art.find("Journal")
  journal = (journal_el.findtext("ISOAbbreviation") or journal_el.findtext("Title", "")) if journal_el is not None else ""
  year = _parse_year(journal_el)
  authors = _parse_authors(art.find("AuthorList"))
  abstract = _parse_abstract(art)

  return {
    "pmid": pmid,
    "title": title,
    "authors": authors,
    "journal": journal,
    "year": year,
    "abstract": abstract,
  }


def _parse_authors(author_list: Optional[ET.Element]) -> str:
  if author_list is None:
    return ""
  names = []
  for author in author_list.findall("Author"):
    last = author.findtext("LastName", "")
    initials = author.findtext("Initials", "")
    if last:
      names.append(f"{last} {initials}".strip())
  if len(names) > 3:
    return f"{names[0]} et al."
  return ", ".join(names)


def _parse_year(journal_el: Optional[ET.Element]) -> str:
  if journal_el is None:
    return ""
  pub_date = journal_el.find("JournalIssue/PubDate")
  if pub_date is None:
    return ""
  year = pub_date.findtext("Year", "")
  if year:
    return year
  # MedlineDate format: "2023 Jan-Feb" — take first 4 chars
  medline_date = pub_date.findtext("MedlineDate", "")
  return medline_date[:4] if medline_date else ""


def _parse_abstract(art: ET.Element) -> str:
  abstract_el = art.find("Abstract")
  if abstract_el is None:
    return ""
  parts = []
  for text_el in abstract_el.findall("AbstractText"):
    label = text_el.get("Label")
    # itertext() handles inline markup like <b>, <i>, <sup>
    text = "".join(text_el.itertext())
    parts.append(f"{label}: {text}" if label else text)
  return " ".join(parts)
