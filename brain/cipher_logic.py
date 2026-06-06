"""Cipher logic — decompose broad claims; cross-domain research when specifics exist.

Marie/Ciper facet drill-down (product name Ciper; module cipher_logic).

Aureon uses taxonomy + collected documents (not invented prose).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from brain.domains.generate_micros import topics_for
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY, lookup_names, taxonomy_catalog
from brain.simple_qa import to_simple_answer

# Broad terms → facet drill-down (Marie/Ciper style)
FACET_DECOMPOSITIONS: dict[str, list[str]] = {
    "blood": ["blood type (ABO/Rh)", "plasma / water fraction", "red blood cells", "iron / hemoglobin"],
    "energy": ["kinetic energy", "potential energy", "thermal heat", "chemical bond energy"],
    "learning": ["data collection", "labeling", "weight training", "evaluation / graduation"],
    "intelligence": ["pattern matching", "labeled classification", "measurable accuracy", "cross-domain links"],
    "consciousness": ["subjective experience", "neural correlates", "behavioral markers", "philosophical definitions"],
    "money": ["currency unit", "exchange value", "debt / credit", "commodity backing"],
    "code": ["syntax", "runtime behavior", "data structures", "test coverage"],
    "health": ["symptoms", "diagnosis", "treatment", "prevention"],
    "love": ["attachment", "chemistry / hormones", "behavior", "cultural meaning"],
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "who",
        "how",
        "why",
        "when",
        "where",
        "can",
        "could",
        "will",
        "would",
        "i",
        "you",
        "we",
        "they",
        "my",
        "your",
        "about",
        "does",
        "do",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "it",
        "this",
        "that",
        "move",
        "make",
        "have",
        "know",
        "tell",
        "explain",
        "describe",
    }
)

_CLAIM_RE = re.compile(
    r"\b(i|we|you|it|they)\s+(can|could|will|would|may|might)\s+[\w']+",
    re.IGNORECASE,
)


@dataclass
class TaxonomyHit:
    path: str
    domain: str
    subdomain: str
    micro: str
    match_kind: str
    match_label: str
    score: float


@dataclass
class CiperResult:
    mode: str
    reply: str
    subject: str
    facets: list[str] = field(default_factory=list)
    cross_domain_paths: list[str] = field(default_factory=list)
    domains_spanned: list[str] = field(default_factory=list)
    grounded: bool = False
    agi_traits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reply": self.reply,
            "subject": self.subject,
            "facets": self.facets,
            "cross_domain_paths": self.cross_domain_paths,
            "domains_spanned": self.domains_spanned,
            "grounded": self.grounded,
            "agi_traits": self.agi_traits,
        }


_taxonomy_index: list[TaxonomyHit] | None = None


def _build_taxonomy_index() -> list[TaxonomyHit]:
    global _taxonomy_index
    if _taxonomy_index is not None:
        return _taxonomy_index

    index: list[TaxonomyHit] = []
    catalog = taxonomy_catalog()
    for domain_slug, domain_entry in catalog.items():
        domain_name = domain_entry.get("name", domain_slug)
        for sub_slug, sub_entry in domain_entry.get("subdomains", {}).items():
            sub_name = sub_entry.get("name", sub_slug)
            for micro_slug, micro_entry in sub_entry.get("micro_subdomains", {}).items():
                micro_name = micro_entry.get("name", micro_slug)
                path = f"{domain_slug}.{sub_slug}.{micro_slug}"
                index.append(
                    TaxonomyHit(path, domain_slug, sub_slug, micro_slug, "micro", micro_name, 1.0)
                )
                for topic in micro_entry.get("topics", []):
                    index.append(
                        TaxonomyHit(path, domain_slug, sub_slug, micro_slug, "topic", topic, 1.2)
                    )
                index.append(
                    TaxonomyHit(path, domain_slug, sub_slug, micro_slug, "subdomain", sub_name, 0.8)
                )
        index.append(
            TaxonomyHit(f"{domain_slug}.*.*", domain_slug, "", "", "domain", domain_name, 0.5)
        )

    _taxonomy_index = index
    return index


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z']{3,}", (text or "").lower())
    out: list[str] = []
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if token not in out:
            out.append(token)
    return out


def _primary_subject(text: str, keywords: list[str]) -> str:
    if not keywords:
        return "that"
    cleaned = re.sub(r"\s+", " ", text.strip())
    lower = cleaned.lower()
    for prefix in ("what is ", "what are ", "what type of ", "tell me about ", "explain "):
        if lower.startswith(prefix):
            rest = cleaned[len(prefix) :].strip(" ?.")
            if rest:
                return rest.split(" and ")[0].strip()
    if keywords:
        return keywords[0]
    return "that"


def search_taxonomy(keyword: str, *, limit: int = 12) -> list[TaxonomyHit]:
    needle = keyword.lower()
    if len(needle) < 3:
        return []

    scored: list[tuple[float, TaxonomyHit]] = []
    for hit in _build_taxonomy_index():
        label = hit.match_label.lower()
        if needle == label:
            scored.append((4.0 + hit.score, hit))
        elif needle in label.split():
            scored.append((3.0 + hit.score, hit))
        elif label.startswith(needle) or needle in label:
            scored.append((2.0 + hit.score, hit))
        elif needle in label:
            scored.append((1.0 + hit.score, hit))

    scored.sort(key=lambda row: row[0], reverse=True)
    seen_paths: set[str] = set()
    out: list[TaxonomyHit] = []
    for _score, hit in scored:
        key = f"{hit.path}:{hit.match_label}"
        if key in seen_paths:
            continue
        seen_paths.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def cross_domain_hits(text: str, *, limit: int = 12) -> list[TaxonomyHit]:
    merged: list[TaxonomyHit] = []
    seen: set[str] = set()
    for keyword in _keywords(text)[:4]:
        for hit in search_taxonomy(keyword, limit=limit):
            key = f"{hit.path}:{hit.match_label}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:limit]


def facets_for_subject(subject: str, hits: list[TaxonomyHit]) -> list[str]:
    key = subject.lower().split()[0] if subject else ""
    if key in FACET_DECOMPOSITIONS:
        return list(FACET_DECOMPOSITIONS[key])

    facets: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        label = hit.match_label
        low = label.lower()
        if low in seen:
            continue
        seen.add(low)
        facets.append(label)
        if len(facets) >= 4:
            break

    if facets:
        return facets

    leaf = topics_for(hits[0].domain, hits[0].subdomain, hits[0].micro) if hits else []
    return leaf[:4] if leaf else []


def format_ciper_question(subject: str, facets: list[str]) -> str:
    if not facets:
        return f"What type of {subject}?"
    if len(facets) == 1:
        return f"What type of {subject}: {facets[0]}?"
    body = ", ".join(facets[:-1]) + f", the {facets[-1]}"
    return f"What type of {subject}, {body}?"


def _is_broad(text: str, subject: str, hits: list[TaxonomyHit]) -> bool:
    if _CLAIM_RE.search(text):
        return True
    key = subject.lower().split()[0]
    if key in FACET_DECOMPOSITIONS:
        return True
    domains = {h.domain for h in hits}
    if len(domains) >= 2 and len(hits) >= 3:
        return True
    if len(hits) >= 5:
        return True
    words = len(text.split())
    if words <= 6 and key in FACET_DECOMPOSITIONS:
        return True
    return len(domains) >= 3


def _fetch_docs_for_paths(paths: list[str], *, per_path: int = 1) -> list[dict[str, Any]]:
    from brain.self_inquiry import fetch_learning_context

    docs: list[dict[str, Any]] = []
    for path in paths:
        parts = path.split(".")
        if len(parts) != 3:
            continue
        domain, sub, micro = parts
        ctx = fetch_learning_context(domain, sub, micro, doc_limit=per_path)
        for doc in ctx.get("documents") or []:
            docs.append({**doc, "path": path})
        if len(docs) >= 4:
            break
    return docs


def _snippet(text: str, limit: int = 90) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return (cut or cleaned[:limit]).rstrip(".,;:") + "…"


def ciper_research(text: str) -> CiperResult | None:
    """Decompose broad input or answer with cross-domain evidence when specific enough."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned or cleaned.startswith("/"):
        return None

    keywords = _keywords(cleaned)
    if not keywords:
        return None

    subject = _primary_subject(cleaned, keywords)
    hits = cross_domain_hits(cleaned)
    domains = sorted({h.domain for h in hits})
    paths = []
    seen_paths: set[str] = set()
    for hit in hits:
        if hit.path.endswith(".*.*") or hit.path in seen_paths:
            continue
        seen_paths.add(hit.path)
        paths.append(hit.path)

    facets = facets_for_subject(subject, hits)
    agi_traits: list[str] = []
    if len(domains) >= 2:
        agi_traits.append("cross_domain_research")
    if facets:
        agi_traits.append("facet_decomposition")
    if hits:
        agi_traits.append("taxonomy_mapping")

    broad = _is_broad(cleaned, subject, hits)

    # Specific leaf topic with one dominant path → answer from corpus
    if not broad and hits:
        top = hits[0]
        docs = _fetch_docs_for_paths([top.path])
        if docs:
            names = lookup_names(top.domain, top.subdomain, top.micro)
            micro_name = names.get("micro_subdomain", top.micro)
            reply = _snippet(docs[0]["text"], 100)
            if len(domains) >= 2:
                other = next((d for d in domains if d != top.domain), None)
                if other:
                    reply = f"{reply} (also spans {other.replace('_', ' ')})."
                    agi_traits.append("cross_domain_link")
            return CiperResult(
                mode="answer",
                reply=to_simple_answer(reply, max_len=160),
                subject=subject,
                facets=facets,
                cross_domain_paths=paths[:4],
                domains_spanned=domains,
                grounded=True,
                agi_traits=agi_traits,
            )

        if top.match_kind == "topic":
            return CiperResult(
                mode="answer",
                reply=to_simple_answer(
                    f"{top.match_label} maps to {top.path.replace('_', ' ')}.",
                    max_len=160,
                ),
                subject=subject,
                facets=facets,
                cross_domain_paths=paths[:4],
                domains_spanned=domains,
                grounded=False,
                agi_traits=agi_traits,
            )

    # Broad claim or multi-domain topic → Ciper drill-down
    if broad and facets:
        question = format_ciper_question(subject, facets)
        return CiperResult(
            mode="decompose",
            reply=to_simple_answer(question, max_len=160),
            subject=subject,
            facets=facets,
            cross_domain_paths=paths[:4],
            domains_spanned=domains,
            grounded=False,
            agi_traits=agi_traits,
        )

    # Cross-domain map when hits exist but no single answer
    if hits and len(domains) >= 2:
        labels = [h.match_label for h in hits[:3]]
        reply = f"{subject} spans {len(domains)} domains: {', '.join(labels)}."
        return CiperResult(
            mode="cross_domain",
            reply=to_simple_answer(reply, max_len=160),
            subject=subject,
            facets=facets,
            cross_domain_paths=paths[:4],
            domains_spanned=domains,
            grounded=False,
            agi_traits=agi_traits,
        )

    if hits:
        top = hits[0]
        return CiperResult(
            mode="map",
            reply=to_simple_answer(f"{subject} → {top.match_label} ({top.path}).", max_len=160),
            subject=subject,
            facets=facets,
            cross_domain_paths=paths[:3],
            domains_spanned=domains,
            grounded=False,
            agi_traits=agi_traits,
        )

    return None


def ciper_follow_up_question(subject: str, hits: list[TaxonomyHit] | None = None) -> str | None:
    """Generate one Ciper-style clarifier for self-inquiry."""
    if hits is None:
        hits = cross_domain_hits(subject, limit=8)
    facets = facets_for_subject(subject, hits)
    if not facets:
        return None
    return format_ciper_question(subject, facets[:4])
