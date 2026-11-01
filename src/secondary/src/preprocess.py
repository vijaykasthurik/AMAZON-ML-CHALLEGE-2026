"""
Text Preprocessing and Normalization Module.
Handles Unicode normalization, legal suffix standardization, address tokenization, and number extraction.
Language-agnostic and country-aware to support US, India, and France.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional

# Compiled regex patterns for speed
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")

# Common business stopwords / legal forms across US, India, and France
LEGAL_TERMS = {
    # English / US / Global
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", 
    "co", "company", "enterprises", "enterprise", "group", "holdings", "services",
    "lp", "llp", "pc", "pllc", "assoc", "associates", "international", "intl",
    # India
    "pvt", "private", "m/s", "ms",
    # France
    "sarl", "sasu", "sas", "sa", "eurl", "snc", "gie", "sci", "ste", "societe",
}

# Common address stopwords / designations
ADDR_STOPWORDS = {
    # English / US
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd", 
    "drive", "dr", "lane", "ln", "court", "ct", "circle", "cir", "way", 
    "terrace", "ter", "highway", "hwy", "suite", "ste", "apt", "apartment",
    "floor", "fl", "building", "bldg", "near", "opp", "opposite", "behind",
    "po", "box", "post", "office", "township", "twp",
    # France
    "rue", "bd", "bld", "chemin", "ch", "impasse", "imp", "allee", "place",
    "pl", "cours", "crs", "zone", "za", "zi", "zac", "batiment", "bat", 
    "etage", "porte", "bis", "ter",
}


def normalize_text(text: Optional[str]) -> str:
    """
    Standardize text: Unicode NFKD normalization, lowercasing, stripping special characters.
    """
    if not text:
        return ""
    # Normalize unicode accents (e.g., É -> E) while retaining characters
    norm = unicodedata.normalize("NFKD", str(text))
    # Remove combining accent marks
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.lower()
    # Replace punctuation with space
    norm = RE_PUNCT.sub(" ", norm)
    # Collapse multiple whitespaces
    norm = RE_SPACES.sub(" ", norm).strip()
    return norm


def extract_tokens(text: Optional[str], remove_stopwords: bool = False, min_len: int = 2) -> List[str]:
    """
    Extract normalized word tokens.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [t for t in tokens if len(t) >= min_len and t not in LEGAL_TERMS and t not in ADDR_STOPWORDS]
    else:
        tokens = [t for t in tokens if len(t) >= min_len]
    return tokens


def extract_numbers(text: Optional[str]) -> Set[str]:
    """
    Extract numeric tokens (e.g. house number, street number, pincode) from address or name.
    Leading zeros are normalized unless it is a 5-6 digit postal code.
    """
    if not text:
        return set()
    matches = RE_NUMBERS.findall(str(text))
    result = set()
    for m in matches:
        # Avoid single digit 0 or 1 if meaningless, keep all relevant numbers
        if len(m) >= 2 or (len(m) == 1 and m != "0"):
            result.add(m)
            # also add stripped leading zero variant (e.g. '0684' -> '684')
            stripped = m.lstrip("0")
            if stripped:
                result.add(stripped)
    return result


def extract_name_blocking_keys(name: Optional[str]) -> List[str]:
    """
    Generate candidate blocking keys from business name:
    1. First 2 significant words
    2. Longest word (>= 4 chars)
    3. First word (>= 3 chars)
    """
    tokens = extract_tokens(name, remove_stopwords=True)
    keys = []
    if not tokens:
        # Fallback to raw tokens if all were removed
        tokens = extract_tokens(name, remove_stopwords=False)
        if not tokens:
            return []

    # Key 1: First significant word
    if len(tokens[0]) >= 3:
        keys.append("w1:" + tokens[0])

    # Key 2: Pair of first two significant words
    if len(tokens) >= 2:
        keys.append("w2:" + tokens[0] + "_" + tokens[1])

    # Key 3: Longest significant token if distinct from first word
    longest = max(tokens, key=len)
    if len(longest) >= 5 and longest != tokens[0]:
        keys.append("long:" + longest)

    return keys


if __name__ == "__main__":
    # Test cases
    sample_name = "Maure Williams Colombier Inc."
    sample_addr = "85 Wayne Avenue, Ticonderoga, NY 12883"
    print(f"Original Name: {sample_name}")
    print(f"Normalized  : {normalize_text(sample_name)}")
    print(f"Name Tokens : {extract_tokens(sample_name, remove_stopwords=True)}")
    print(f"Blocking Keys: {extract_name_blocking_keys(sample_name)}")
    print(f"Numbers     : {extract_numbers(sample_addr)}")
