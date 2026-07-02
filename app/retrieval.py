"""
Retrieval layer: BM25 over rich catalog fields + metadata filtering.
Now uses description, job_levels, languages, duration from the full catalog.
"""

import json
import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


# Map job-level synonyms → catalog job_levels values
JOB_LEVEL_MAP = {
    "entry": "Entry-Level",
    "entry-level": "Entry-Level",
    "junior": "Entry-Level",
    "fresher": "Entry-Level",
    "graduate": "Graduate",
    "grad": "Graduate",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "mid-professional": "Mid-Professional",
    "intermediate": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "professional": "Professional Individual Contributor",
    "manager": "Manager",
    "management": "Manager",
    "front line manager": "Front Line Manager",
    "frontline": "Front Line Manager",
    "supervisor": "Supervisor",
    "director": "Director",
    "executive": "Executive",
    "vp": "Executive",
    "c-suite": "Executive",
}

KEY_LABELS = {
    "A": "Ability Aptitude Cognitive Reasoning Numerical Verbal Inductive Deductive",
    "B": "Biodata Situational Judgement Judgment SJT Scenarios",
    "C": "Competencies UCF Universal",
    "D": "Development 360 Feedback Multi-Rater",
    "E": "Assessment Exercises Center Centre",
    "K": "Knowledge Skills Technical Programming Software",
    "M": "Motivational Motivation Driver",
    "P": "Personality Behaviour Behavior OPQ",
    "S": "Simulation Simulations",
}


class CatalogRetriever:
    def __init__(self, catalog_path: str = "catalog.json"):
        with open(catalog_path) as f:
            self.catalog = json.load(f)

        # Build BM25 corpus: name + description + keys labels + job_levels
        corpus = []
        for entry in self.catalog:
            parts = [entry["name"]]
            if entry.get("description"):
                # Use first 200 chars of description for BM25
                parts.append(entry["description"][:200])
            for code in entry.get("keys", []):
                parts.append(KEY_LABELS.get(code, ""))
            for jl in entry.get("job_levels", []):
                parts.append(jl)
            doc = " ".join(parts).lower()
            corpus.append(doc.split())

        self.bm25 = BM25Okapi(corpus)

        # Build URL set for validation
        self.valid_urls = {e["url"] for e in self.catalog}

    def _normalize_job_level(self, seniority: str) -> Optional[str]:
        """Map free-text seniority to a catalog job_level value."""
        if not seniority:
            return None
        sl = seniority.lower().strip()
        for key, val in JOB_LEVEL_MAP.items():
            if key in sl:
                return val
        return None

    def retrieve(
        self,
        query: str,
        test_types: Optional[List[str]] = None,
        job_level: Optional[str] = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
        language: Optional[str] = None,
        duration_max: Optional[int] = None,
        top_k: int = 20,
        boost_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        BM25 search + rich metadata filters.
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        catalog_job_level = self._normalize_job_level(job_level) if job_level else None

        candidates = []
        for idx, score in enumerate(scores):
            entry = self.catalog[idx]

            # --- Metadata filters ---
            if remote_only and not entry.get("remote"):
                continue
            if adaptive_only and not entry.get("adaptive"):
                continue
            if test_types and entry["test_type"] not in test_types:
                continue
            if catalog_job_level and entry.get("job_levels"):
                if catalog_job_level not in entry["job_levels"]:
                    continue
            if duration_max is not None:
                d = entry.get("duration_minutes")
                if d is not None and d > duration_max:
                    continue
            if language and entry.get("languages"):
                lang_lower = language.lower()
                entry_langs = " ".join(entry["languages"]).lower()
                if lang_lower not in entry_langs:
                    continue

            candidates.append((score, entry))

        # Sort by BM25 score
        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [e for _, e in candidates[:top_k]]

        # Boost additional types if requested (for REFINE)
        if boost_types:
            result_urls = {e["url"] for e in result}
            for btype in boost_types:
                type_cands = []
                for idx, score in enumerate(scores):
                    e = self.catalog[idx]
                    if e["test_type"] == btype and e["url"] not in result_urls:
                        type_cands.append((score, e))
                type_cands.sort(key=lambda x: x[0], reverse=True)
                for _, e in type_cands[:3]:
                    result.append(e)
                    result_urls.add(e["url"])

        return result[:top_k]

    def get_by_name(self, name: str) -> Optional[Dict]:
        """Fuzzy find by name (case-insensitive partial match)."""
        name_lower = name.lower()
        best, best_score = None, 0
        for e in self.catalog:
            n = e["name"].lower()
            if name_lower == n:
                return e
            # partial match score
            score = sum(1 for w in name_lower.split() if w in n)
            if score > best_score:
                best_score, best = score, e
        return best if best_score > 0 else None

    def compare(self, name1: str, name2: str) -> Dict:
        """Return side-by-side data for two assessments."""
        e1 = self.get_by_name(name1)
        e2 = self.get_by_name(name2)
        return {"assessment_1": e1, "assessment_2": e2}
