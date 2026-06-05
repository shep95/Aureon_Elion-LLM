"""Parse COMPLETE_HUMAN_DOMAIN_TAXONOMY.txt into Aureon knowledge_taxonomy.json."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DOMAIN_HEADER = re.compile(r"^\[DOMAIN\s+\d+\]\s+(.+)$")
SUBDOMAIN_LINE = re.compile(r"^(\d{2})\.(\d{2})\s+([A-Z0-9].*)$")
MICRO_LINE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2})\s+(.+)$")
TOPIC_LINE = re.compile(r"^-\s+(.+)$")


def slugify(text: str, *, max_len: int = 63) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[&/]", " and ", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text).strip("_")
    if not text:
        text = "topic"
    if not text[0].isalpha():
        text = f"x_{text}"
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def unique_slug(base: str, seen: set[str]) -> str:
    slug = base
    n = 2
    while slug in seen:
        suffix = f"_{n}"
        slug = (base[: max(1, 63 - len(suffix))] + suffix).rstrip("_")
        n += 1
    seen.add(slug)
    return slug


def parse_taxonomy(text: str) -> dict:
    domains: dict[str, dict] = {}
    current_domain: str | None = None
    current_sub: str | None = None
    domain_slugs: set[str] = set()
    sub_slugs: set[str] = set()
    micro_slugs: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("=") or line.startswith("━"):
            continue
        if line.startswith("Total ") or line.startswith("DOMAIN SUMMARY"):
            continue
        if line.startswith("END OF DOCUMENT") or line.startswith("COMPILED BY"):
            continue
        if re.match(r"^\d{2}\.\s", line):
            continue

        m_domain = DOMAIN_HEADER.match(line)
        if m_domain:
            name = m_domain.group(1).strip()
            current_domain = unique_slug(slugify(name), domain_slugs)
            domains[current_domain] = {
                "name": name,
                "subdomains": {},
            }
            current_sub = None
            sub_slugs = set()
            continue

        m_micro = MICRO_LINE.match(line)
        if m_micro and current_domain and current_sub:
            name = m_micro.group(4).strip()
            micro_slug = unique_slug(slugify(name), micro_slugs)
            domains[current_domain]["subdomains"][current_sub]["micro_subdomains"][micro_slug] = {
                "name": name,
                "topics": [],
            }
            continue

        m_sub = SUBDOMAIN_LINE.match(line)
        if m_sub and current_domain and not MICRO_LINE.match(line):
            name = m_sub.group(3).strip()
            current_sub = unique_slug(slugify(name), sub_slugs)
            micro_slugs = set()
            domains[current_domain]["subdomains"][current_sub] = {
                "name": name,
                "micro_subdomains": {},
            }
            continue

        m_topic = TOPIC_LINE.match(line)
        if m_topic and current_domain and current_sub:
            subs = domains[current_domain]["subdomains"][current_sub]["micro_subdomains"]
            if subs:
                last_micro = next(reversed(subs))
                subs[last_micro]["topics"].append(m_topic.group(1).strip())

    return domains


def to_training_tree(domains: dict) -> dict[str, dict[str, list[str]]]:
    """Flatten to domain → subdomain → [micro_slug, ...] for the brain pipeline."""
    tree: dict[str, dict[str, list[str]]] = {}
    for domain_slug, domain in domains.items():
        tree[domain_slug] = {}
        for sub_slug, sub in domain["subdomains"].items():
            tree[domain_slug][sub_slug] = list(sub["micro_subdomains"].keys())
    return tree


def stats(domains: dict) -> dict[str, int]:
    subs = sum(len(d["subdomains"]) for d in domains.values())
    micros = sum(
        len(sub["micro_subdomains"])
        for d in domains.values()
        for sub in d["subdomains"].values()
    )
    max_subs = max((len(d["subdomains"]) for d in domains.values()), default=0)
    max_micros = max(
        (
            len(sub["micro_subdomains"])
            for d in domains.values()
            for sub in d["subdomains"].values()
        ),
        default=0,
    )
    return {
        "domains": len(domains),
        "subdomains": subs,
        "micro_subdomains": micros,
        "max_subdomains_per_domain": max_subs,
        "max_micros_per_subdomain": max_micros,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "sources" / "COMPLETE_HUMAN_DOMAIN_TAXONOMY.txt"
    if not source.is_file():
        raise SystemExit(f"Missing source taxonomy: {source}")

    text = source.read_text(encoding="utf-8")
    domains = parse_taxonomy(text)
    payload = {
        "version": "1.0",
        "source": source.name,
        "compiled_by": "Zophiel Intelligence of the North",
        "stats": stats(domains),
        "domains": domains,
        "training_tree": to_training_tree(domains),
    }

    out = root / "knowledge_taxonomy.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["stats"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
