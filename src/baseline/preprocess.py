"""Normalisation of business names and addresses.

Every raw record is turned into a handful of cleaned string fields that the
blocking and matching stages work on:

  nn     full normalised name (legal forms mapped to one canonical token)
  nc     "core" name: nn without legal forms and filler words
  nk     compact core name (no spaces) - lets "westernvaneck.com" meet "Western Vaneck"
  nl     canonical legal-form tokens found in the name (e.g. "pvt ltd")
  na     alias found after "dba", "t/a", "aka"... (empty when absent)
  an     full normalised address (street types, states, ordinals canonicalised)
  ac     core address: an without state / region tokens and filler words
  anum   numbers found in the address, leading zeros removed, space separated

Non-Latin (Indic) text is first looked up in a word lexicon that is learned
from the training pairs (``learn_lexicon``); unseen words fall back to the rule
based romaniser in translit.py. Accents are stripped with the standard library
(NFKD), so no third-party transliteration package is needed.

Run ``python preprocess.py`` to learn the lexicon and cache normalised parquet
files for train and test.
"""

from __future__ import annotations

import _env  # noqa: F401  (must come before numpy)

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from translit import has_indic, is_indic, romanize  # noqa: E402

LEXICON_PATH = os.path.join(config.CACHE_DIR, "lexicon.json")

# --------------------------------------------------------------------------
# Static vocabularies (general language knowledge, not entity data)
# --------------------------------------------------------------------------

# Legal / company-form words -> canonical token.
LEGAL = {
    "private": "pvt", "pvt": "pvt", "prv": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "limted": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "corpn": "corp", "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "ltdco": "ltd co", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "ets": "ets", "etablissements": "ets", "etablissement": "ets",
}
LEGAL_CANON = set(" ".join(LEGAL.values()).split())

# Words that carry no identity on their own.
NAME_FILLER = {"the", "and", "of", "mr", "mrs", "ms", "m s", "de", "des", "du",
               "la", "le", "les", "l", "d", "a", "an", "et", "und"}

# Alias markers: "<name> dba <other name>".
ALIAS_RE = re.compile(
    r"\s(?:d/b/a|dba|t/a|a/k/a|aka|f/k/a|fka|trading as|doing business as|formerly)\s")
URL_TAIL_RE = re.compile(r"\s*\|\s*(?:https?://)?(?:www\.)?\S+$")
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|io|biz|info|us)$")
DIGIT_AS_LETTER = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                                 "6": "g", "7": "t", "8": "b"})
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
                 "eleventh": "11", "twelfth": "12", "premier": "1", "1er": "1", "1ere": "1"}

# Street-type and address word abbreviations -> one canonical form.
ADDR_ABBR = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "ste": "st", "sainte": "st",
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln", "court": "ct",
    "ct": "ct", "crt": "ct", "circle": "cir", "cir": "cir", "boulevard": "blvd",
    "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "place": "pl", "pl": "pl",
    "plc": "pl", "terrace": "ter", "terr": "ter", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "square": "sq", "sq": "sq", "trail": "trl",
    "trl": "trl", "way": "way", "wy": "way", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "heights": "hts", "hts": "hts", "point": "pt", "pt": "pt", "center": "ctr",
    "centre": "ctr", "ctr": "ctr", "expressway": "expy", "freeway": "fwy",
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "route": "rte",
    "rte": "rte", "impasse": "imp", "imp": "imp", "chemin": "chem", "ch": "chem",
    "chem": "chem", "quai": "quai", "cours": "crs", "crs": "crs", "passage": "pass",
    "sente": "sente", "residence": "res", "res": "res", "rpt": "rpt", "rond": "rpt",
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "marg", "colony": "colony",
    "clny": "colony", "sector": "sec", "sec": "sec", "extn": "ext", "extension": "ext",
    "ext": "ext", "cross": "cross", "main": "main", "phase": "ph", "ph": "ph",
    "layout": "layout", "complex": "complex", "cmplx": "complex", "bldg": "bldg",
    "building": "bldg", "apartment": "apt", "apartments": "apt", "apt": "apt",
    "appartement": "apt", "suite": "ste", "floor": "fl", "fl": "fl", "flr": "fl",
    "etage": "fl", "near": "near", "nr": "near", "opp": "near", "opposite": "near",
    "behind": "near", "beside": "near", "vlg": "village", "vill": "village",
    "township": "twp", "twp": "twp",
}
# Tokens that are dropped from the *core* address (noise / filler / admin level).
ADDR_FILLER = {
    "unit", "apt", "no", "number", "num", "door", "flat", "flt", "plot", "house",
    "hno", "h", "fl", "ste", "near", "po", "box", "pob", "dist", "district", "tal",
    "taluka", "tehsil", "via", "post", "ps", "city", "of", "cdp", "borough", "twp",
    "village", "town", "municipality", "the", "de", "du", "des", "la", "le", "les",
    "l", "d", "bis", "and", "c", "o", "s", "null", "none", "na", "nan", "region",
    "shop", "office", "room", "block", "blk", "bldg", "gf", "ground",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu & kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "ch", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld", "andaman and nicobar islands": "an",
}
# Region / department names seen in the French test addresses; treated like states.
FR_ADMIN = ["hauts de france", "nouvelle aquitaine", "pays de la loire", "gironde",
            "nord", "loire atlantique", "pas de calais", "ile de france",
            "auvergne rhone alpes", "occitanie", "bretagne", "normandie", "grand est",
            "provence alpes cote d azur", "bourgogne franche comte", "centre val de loire"]

# Per-country map: address component text -> canonical admin token
# ("kansas" and "ks" -> "adm_ks"). Unknown countries use every known name.
ADMIN_BY_COUNTRY = {"US": {}, "India": {}}
for _country, _m in (("US", US_STATES), ("India", IN_STATES)):
    for _name, _code in _m.items():
        ADMIN_BY_COUNTRY[_country][_name] = "adm_" + _code
        ADMIN_BY_COUNTRY[_country][_code] = "adm_" + _code
ADMIN_ANY = {}
for _name in FR_ADMIN:
    ADMIN_ANY[_name] = "adm_" + _name.replace(" ", "_")
for _country in ("US", "India"):
    for _k, _v in ADMIN_BY_COUNTRY[_country].items():
        if len(_k) > 2:  # full names only; two-letter codes are too ambiguous
            ADMIN_ANY.setdefault(_k, _v)

_PUNCT_STRIP = ".,;:!?()[]{}\"'|/\\-_#*<>~`^%$@+=«»“”‘’"
_NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_SPACES_RE = re.compile(r"\s+")
_ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")
_SPLIT_DL_RE = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_POBOX_RE = re.compile(r"\b(?:p\s*o\s*box|post\s*box|po\s*box|pob|b\s*p)\s*\d+")
_NULL_RE = re.compile(r"<null>|\bn/a\b|\bnull\b|\bnone\b|\bnan\b")

LEXICON: dict = {}


# --------------------------------------------------------------------------
# Character level cleaning
# --------------------------------------------------------------------------

def _native_token_key(tok: str) -> str:
    """Lexicon key of a raw token: NFKC, no zero-width characters, no edge punctuation."""
    tok = unicodedata.normalize("NFKC", tok)
    tok = "".join(ch for ch in tok if unicodedata.category(ch) != "Cf")
    return tok.strip(_PUNCT_STRIP)


def romanise_text(text: str) -> str:
    """Replace every Indic word by its lexicon translation or romanisation."""
    if not has_indic(text):
        return text
    out = []
    for tok in text.split():
        if has_indic(tok):
            key = _native_token_key(tok)
            if key in LEXICON:
                out.append(LEXICON[key])
            else:
                out.append(romanize(tok))
        else:
            out.append(tok)
    return " ".join(out)


def ascii_fold(text: str) -> str:
    """Lower-case, drop zero-width/format characters and strip accents."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = romanise_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ß", "ss").replace("æ", "ae").replace("ø", "o").replace("œ", "oe")
    return text.lower()


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

def _clean_name_tokens(text: str) -> list:
    """Tokenise a folded name string and canonicalise each token."""
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", text)  # l.c.s.w -> lcsw
    text = text.replace(".", "").replace("'", "")
    text = _NONALNUM_RE.sub(" ", text)
    toks = []
    for t in text.split():
        if not t.isalpha() and not t.isdigit() and not _ORD_RE.fullmatch(t):
            n_dig = sum(c.isdigit() for c in t)
            if n_dig <= 2 and len(t) - n_dig >= 1:
                t = t.translate(DIGIT_AS_LETTER)  # 6ordon -> gordon, c0 -> co
        canon = LEGAL.get(t)
        for c in (canon.split() if canon else [t]):
            if not toks or toks[-1] != c:  # "vidyalaya vidyalaya" -> "vidyalaya"
                toks.append(c)
    return toks


def normalize_name(raw) -> tuple:
    """Return (nn, nc, nk, nl, na) for one raw business name."""
    if raw is None:
        return "", "", "", "", ""
    text = ascii_fold(str(raw))
    text = URL_TAIL_RE.sub("", text) if "|" in text else text
    alias = ""
    m = ALIAS_RE.search(" " + text + " ")
    if m:
        s = " " + text + " "
        left, right = s[:m.start()].strip(), s[m.end():].strip()
        text, alias = (left, right) if left else (right, "")
    stripped = text.strip().strip(_PUNCT_STRIP).strip()
    dm = DOMAIN_RE.match(stripped)
    if dm:
        text = dm.group(1).replace("-", " ")
    toks = _clean_name_tokens(text)
    nn = " ".join(toks)
    legal = [t for t in toks if t in LEGAL_CANON]
    core = [t for t in toks if t not in LEGAL_CANON and t not in NAME_FILLER]
    if not core:
        core = toks
    nc = " ".join(core)
    nk = "".join(core)
    na = " ".join(t for t in _clean_name_tokens(alias)
                  if t not in LEGAL_CANON and t not in NAME_FILLER) if alias else ""
    return nn, nc, nk, " ".join(sorted(set(legal))), na


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

def normalize_address(raw, country: str) -> tuple:
    """Return (an, ac, anum) for one raw address.

    The address is processed comma component by comma component: a component
    that is exactly a state / region name or code (e.g. "Kansas", "KS",
    "Maharashtra", "MH", "Gironde") becomes an admin token, which is kept in
    ``an`` in canonical form but excluded from the core address ``ac``.
    """
    if raw is None:
        return "", "", ""
    text = ascii_fold(str(raw))
    text = _NULL_RE.sub(" ", text)
    text = _POBOX_RE.sub(" ", text)
    text = text.replace("&", " and ").replace("'", "")
    admin = ADMIN_BY_COUNTRY.get(country, ADMIN_ANY)
    toks, core = [], []
    for comp in text.split(","):
        comp = _SPACES_RE.sub(" ", _NONALNUM_RE.sub(" ", comp)).strip()
        if not comp:
            continue
        code = admin.get(comp)
        if code is not None:
            toks.append(code)
            continue
        comp = _ORD_RE.sub(r"\g<1>", comp)
        comp = _SPLIT_DL_RE.sub(" ", comp)
        for t in comp.split():
            t = ORDINAL_WORDS.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            else:
                t = ADDR_ABBR.get(t, t)
            toks.append(t)
            if t not in ADDR_FILLER:
                core.append(t)
    an = " ".join(toks)
    ac = " ".join(core)
    anum = " ".join(t for t in toks if t.isdigit())
    return an, ac, anum


# --------------------------------------------------------------------------
# Lexicon learned from training pairs
# --------------------------------------------------------------------------

def learn_lexicon(min_count: int = 3, min_share: float = 0.5, max_pairs: int = 3_000_000):
    """Learn native-script word -> Latin word translations from training matches.

    For every matched (S1, S2/S3) pair whose S2/S3 side contains Indic script:
      * names: if both names have the same number of tokens they are aligned
        position by position (transliteration keeps word order);
      * addresses: each native-script comma component is counted against every
        Latin component of the S1 address; the state name, present in every S1
        address of that state, wins the vote.
    A translation is kept when it is seen ``min_count`` times and wins at least
    ``min_share`` of the votes for that native word.
    """
    t0 = time.time()
    s1 = pl.read_csv(config.source_path("train", 1), separator="\t", quote_char=None,
                     infer_schema=False)
    oth = pl.concat([pl.read_csv(config.source_path("train", s), separator="\t",
                                 quote_char=None, infer_schema=False) for s in (2, 3)])
    gt = pl.read_csv(config.gt_path(), separator="\t", quote_char=None, infer_schema=False)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").drop_nulls()
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "eid"}))
    oth = oth.filter(pl.col("country") == "India")
    # keep only records with non-ASCII text to save time
    oth = oth.filter(pl.col("business_name").str.contains(r"[^\x00-\x7F]")
                     | pl.col("business_address").fill_null("").str.contains(r"[^\x00-\x7F]"))
    df = (pairs.join(oth.rename({"entity_id": "eid", "business_name": "n2",
                                 "business_address": "a2"}).drop("country"), on="eid")
          .join(s1.rename({"entity_id": "s1", "business_name": "n1",
                           "business_address": "a1"}).drop("country"), on="s1"))
    if df.height > max_pairs:
        df = df.sample(max_pairs, seed=config.SEED)
    print(f"lexicon: {df.height} native-script pairs ({time.time() - t0:.0f}s)")

    votes = defaultdict(Counter)
    for n1, n2, a1, a2 in df.select("n1", "n2", "a1", "a2").iter_rows():
        if n2 and has_indic(n2):
            t2 = [_native_token_key(t) for t in n2.split()]
            t1 = [_native_token_key(t).lower() for t in n1.split()]
            t2 = [t for t in t2 if t]
            t1 = [t for t in t1 if t]
            if len(t1) == len(t2):
                for a, b in zip(t2, t1):
                    if has_indic(a) and b:
                        votes[a][b] += 1
        if a2 and a1 and has_indic(a2):
            comps1 = {c.strip().lower() for c in a1.split(",") if c.strip()}
            for comp in a2.split(","):
                comp = comp.strip()
                if comp and has_indic(comp):
                    for c1 in comps1:
                        votes[comp][c1] += 1
                    votes[comp]["__n__"] += 1
    lex = {}
    for native, cnt in votes.items():
        total = cnt.pop("__n__", None) or sum(cnt.values())
        best, c = cnt.most_common(1)[0]
        if c >= min_count and c / total >= min_share and not has_indic(best):
            lex[native] = best
    # multi-word native components (state names) are also stored per word so
    # romanise_text can look them up token by token
    for native, latin in list(lex.items()):
        parts = native.split()
        if len(parts) > 1:
            lat = latin.split()
            if len(lat) == len(parts):
                for p, l in zip(parts, lat):
                    lex.setdefault(_native_token_key(p), l)
            else:
                lex.setdefault(_native_token_key(parts[0]), latin)
                for p in parts[1:]:
                    lex.setdefault(_native_token_key(p), "")
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False)
    print(f"lexicon: {len(lex)} entries learned in {time.time() - t0:.0f}s")
    return lex


def load_lexicon():
    """Load the learned lexicon into the module global used by ascii_fold."""
    global LEXICON
    if os.path.exists(LEXICON_PATH):
        with open(LEXICON_PATH, encoding="utf-8") as f:
            LEXICON = json.load(f)
    return LEXICON


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def _init_worker():
    load_lexicon()


def _normalize_chunk(rows):
    """Worker: normalise a list of (name, address, country) tuples."""
    out = []
    for name, addr, country in rows:
        out.append(normalize_name(name) + normalize_address(addr, country))
    return out


NORM_COLS = ["nn", "nc", "nk", "nl", "na", "an", "ac", "anum"]


def normalize_frame(df: pl.DataFrame, n_jobs: int = config.N_JOBS,
                    chunk: int = 20000) -> pl.DataFrame:
    """Add the normalised columns to a raw source frame (parallel)."""
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(n_jobs, initializer=_init_worker) as pool:
        res = pool.map(_normalize_chunk, chunks, chunksize=1)
    flat = [r for part in res for r in part]
    cols = list(zip(*flat)) if flat else [[] for _ in NORM_COLS]
    return df.with_columns([pl.Series(c, list(v), dtype=pl.Utf8)
                            for c, v in zip(NORM_COLS, cols)])


def read_source(split: str, source: int) -> pl.DataFrame:
    """Read one raw TSV with every column as string (no quote handling)."""
    return pl.read_csv(config.source_path(split, source), separator="\t",
                       quote_char=None, infer_schema=False)


def norm_path(split: str, source: int) -> str:
    """Parquet cache path of a normalised source."""
    return os.path.join(config.CACHE_DIR, f"norm_{split}_s{source}.parquet")


def main():
    """Learn the lexicon (if missing) and cache normalised train/test sources."""
    if not os.path.exists(LEXICON_PATH):
        learn_lexicon()
    load_lexicon()
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        for source in (1, 2, 3):
            t0 = time.time()
            df = read_source(split, source)
            df = normalize_frame(df)
            df.write_parquet(norm_path(split, source))
            print(f"{split} s{source}: {df.height} rows normalised in {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
