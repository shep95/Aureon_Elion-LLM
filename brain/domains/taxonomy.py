"""Human knowledge domain taxonomy — domains and subdomains."""

from __future__ import annotations

# Each top-level domain maps to subdomains the brain trains on independently.
KNOWLEDGE_TAXONOMY: dict[str, list[str]] = {
    "mathematics": [
        "algebra",
        "calculus",
        "geometry",
        "statistics",
        "number_theory",
        "topology",
        "logic",
    ],
    "physics": [
        "classical_mechanics",
        "thermodynamics",
        "electromagnetism",
        "quantum_mechanics",
        "relativity",
        "optics",
        "nuclear_physics",
    ],
    "chemistry": [
        "organic",
        "inorganic",
        "biochemistry",
        "physical_chemistry",
        "analytical",
        "materials",
    ],
    "biology": [
        "genetics",
        "ecology",
        "microbiology",
        "anatomy",
        "evolution",
        "molecular_biology",
        "neuroscience",
    ],
    "medicine": [
        "anatomy",
        "pharmacology",
        "pathology",
        "epidemiology",
        "surgery",
        "public_health",
    ],
    "computer_science": [
        "algorithms",
        "machine_learning",
        "systems",
        "networks",
        "databases",
        "security",
        "software_engineering",
    ],
    "engineering": [
        "mechanical",
        "electrical",
        "civil",
        "chemical",
        "aerospace",
        "control_systems",
        "materials",
    ],
    "astronomy": [
        "cosmology",
        "planetary_science",
        "astrophysics",
        "observational",
    ],
    "earth_science": [
        "geology",
        "oceanography",
        "meteorology",
        "climatology",
        "seismology",
    ],
    "history": [
        "ancient",
        "medieval",
        "modern",
        "political",
        "economic",
        "cultural",
        "military",
    ],
    "philosophy": [
        "ethics",
        "metaphysics",
        "epistemology",
        "logic",
        "political_philosophy",
        "philosophy_of_mind",
    ],
    "linguistics": [
        "syntax",
        "semantics",
        "phonology",
        "sociolinguistics",
        "computational",
        "sanskrit_studies",
    ],
    "psychology": [
        "cognitive",
        "developmental",
        "clinical",
        "social",
        "behavioral",
        "neuropsychology",
    ],
    "economics": [
        "microeconomics",
        "macroeconomics",
        "econometrics",
        "development",
        "finance",
        "behavioral",
    ],
    "law": [
        "constitutional",
        "criminal",
        "civil",
        "international",
        "corporate",
        "intellectual_property",
    ],
    "political_science": [
        "governance",
        "international_relations",
        "policy",
        "comparative_politics",
    ],
    "sociology": [
        "culture",
        "institutions",
        "stratification",
        "urban_studies",
        "demography",
    ],
    "anthropology": [
        "cultural",
        "archaeology",
        "physical",
        "linguistic",
    ],
    "education": [
        "pedagogy",
        "curriculum",
        "learning_science",
        "assessment",
    ],
    "arts": [
        "visual_arts",
        "sculpture",
        "design",
        "film",
        "photography",
    ],
    "literature": [
        "poetry",
        "fiction",
        "drama",
        "literary_criticism",
        "comparative_literature",
    ],
    "music": [
        "theory",
        "composition",
        "history",
        "ethnomusicology",
    ],
    "religion": [
        "theology",
        "comparative_religion",
        "mythology",
        "ethics",
    ],
    "geography": [
        "human",
        "physical",
        "cartography",
        "geopolitics",
    ],
    "agriculture": [
        "crop_science",
        "soil_science",
        "agronomy",
        "sustainable_farming",
    ],
    "architecture": [
        "design",
        "urban_planning",
        "structural",
        "history",
    ],
    "business": [
        "management",
        "marketing",
        "operations",
        "entrepreneurship",
        "accounting",
    ],
    "communication": [
        "journalism",
        "media_studies",
        "rhetoric",
        "public_relations",
    ],
    "vedic_sciences": [
        "astronomy_jyotisha",
        "grammar_vyakarana",
        "phonetics_shiksha",
        "ritual_kalpa",
        "philosophy_darshana",
    ],
}


def all_domain_slugs() -> list[str]:
    return list(KNOWLEDGE_TAXONOMY.keys())


def all_subdomain_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for domain, subdomains in KNOWLEDGE_TAXONOMY.items():
        for sub in subdomains:
            pairs.append((domain, sub))
    return pairs


def total_subdomains() -> int:
    return sum(len(subs) for subs in KNOWLEDGE_TAXONOMY.values())
