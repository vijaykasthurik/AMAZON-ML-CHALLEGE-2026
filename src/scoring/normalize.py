"""Text normalization for business names and addresses.

Everything here is hand-written domain knowledge (abbreviations, legal forms,
state codes) plus a native-script -> Latin token dictionary that is *learned
from the training ground truth* (see learn_dict.py). No external data.
"""
import re
import unicodedata
from unidecode import unidecode

# ---------------------------------------------------------------- constants
NON_ASCII = re.compile(r"[^\x00-\x7f]")
INDIC = re.compile(r"[ऀ-෿]")
PUNCT = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")
ORDINAL = re.compile(r"^\d+(st|nd|rd|th|e|er|eme|ere|bis|ter)?$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})

# legal forms -> canonical token; canonical tokens are dropped from the "core" name
LEGAL = {
    "private": "pvt", "pvt": "pvt", "prvt": "pvt", "pvte": "pvt", "priv": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "limitada": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp", "pc": "pc", "pllc": "pllc",
    "pa": "pa", "ltee": "ltd", "opc": "opc", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "scp": "scp", "selarl": "selarl", "gie": "gie", "ets": "ets",
    "etablissements": "ets", "etablissement": "ets", "group": "group", "groupe": "group",
}
LEGAL_CANON = set(LEGAL.values())
NAME_STOP = {"the", "and", "et", "of", "de", "du", "des", "la", "le", "les", "l", "d", "a", "k",
             "aka", "dba", "fka", "m", "s", "ms", "mr", "mrs", "smt", "shri", "sri", "dr", "www", "com",
             "net", "org", "in", "fr", "en"}

# address abbreviations; country-specific overrides applied first
ADDR_COMMON = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard",
    "cir": "circle", "ct": "court", "crt": "court", "hwy": "highway", "pkwy": "parkway", "pky": "parkway",
    "pl": "place", "sq": "square", "trl": "trail", "tr": "trail", "ter": "terrace", "terr": "terrace",
    "cres": "crescent", "pt": "point", "mt": "mount", "ft": "fort", "hts": "heights", "jn": "junction",
    "jct": "junction", "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "aly": "alley",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite", "ngr": "nagar",
    "mkt": "market", "bldg": "building", "apts": "apartments", "extn": "extension", "ext": "extension",
    "sec": "sector", "secto": "sector", "colony": "colony", "clny": "colony", "rly": "railway",
    "stn": "station", "chowk": "chowk", "marg": "marg", "twp": "township", "vlg": "village",
    "vill": "village", "po": "post", "dist": "district", "tq": "taluk", "tal": "taluk",
    "raod": "road", "stree": "street", "cdp": "", "ctr": "center", "centre": "center",
}
ADDR_FR = {
    "r": "rue", "st": "saint", "ste": "sainte", "bd": "boulevard", "blvd": "boulevard",
    "av": "avenue", "ave": "avenue", "all": "allee", "pl": "place", "imp": "impasse",
    "ch": "chemin", "chem": "chemin", "rte": "route", "sq": "square", "fg": "faubourg",
    "fbg": "faubourg", "crs": "cours", "pass": "passage", "qu": "quai", "res": "residence",
    "lot": "lotissement", "prof": "professeur", "gen": "general", "mal": "marechal",
}
# filler tokens that carry no identity in addresses (their numbers are kept separately)
ADDR_FILLER = {"no", "nos", "number", "door", "h", "hno", "house", "plot", "flat", "shop", "office",
               "off", "unit", "apt", "apartment", "suite", "ste", "pmb", "box", "po", "post", "floor",
               "fl", "flr", "gf", "ff", "sf", "null", "none", "nan", "na", "c", "o", "s", "f", "g", "b",
               "d", "a", "k", "survey", "sy", "khasra", "kh", "ward", "property", "premises", "wing",
               "block", "blk", "room", "rm", "near", "opposite", "behind", "beside", "n", "de", "du",
               "des", "la", "le", "les", "l", "d", "et", "and", "the", "of", "bis", "ter", "cedex", "bp"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "ladakh": "la", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "andaman and nicobar islands": "an", "lakshadweep": "ld", "or": "od", "tg": "ts", "ct": "cg",
    "ut": "uk", "ori": "od",
}
FR_REGIONS = {
    "nouvelle aquitaine": "naq", "hauts de france": "hdf", "pays de la loire": "pdl",
    "ile de france": "idf", "bretagne": "bre", "normandie": "nor", "occitanie": "occ",
    "grand est": "ges", "auvergne rhone alpes": "ara", "provence alpes cote d azur": "pac",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
}


def _state_regex(d):
    """Compile a longest-first regex that matches any full state name in `d` as whole words."""
    keys = sorted((k for k in d if " " in k or len(k) > 2), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


STATE_MAPS = {"US": (US_STATES, _state_regex(US_STATES)),
              "India": (IN_STATES, _state_regex(IN_STATES)),
              "France": (FR_REGIONS, _state_regex(FR_REGIONS))}
STATE_CODES = {c: set(d.values()) for c, (d, _) in STATE_MAPS.items()}

# phonetic key: digraph folding, voiced->unvoiced, drop vowels after 1st char, squeeze repeats
_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
             ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k"), ("wh", "w"), ("x", "ks")]
_PH_TABLE = str.maketrans({"q": "k", "c": "k", "z": "s", "w": "v", "d": "t", "g": "k", "b": "p", "j": "c"})
_VOWELS = re.compile(r"[aeiouyh]")
_REPEAT = re.compile(r"(.)\1+")


def phon(tok: str) -> str:
    """Phonetic skeleton of a Latin token; robust to transliteration vowels and doubling."""
    if not tok or tok.isdigit():
        return tok
    for a, b in _DIGRAPHS:
        if a in tok:
            tok = tok.replace(a, b)
    tok = tok.translate(_PH_TABLE)
    head, tail = tok[0], _VOWELS.sub("", tok[1:])
    return _REPEAT.sub(r"\1", head + tail)


# ---------------------------------------------------------------- latinization
TRANSLIT = {}  # native token -> latin token, filled by set_dict()


def set_dict(d):
    """Install the learned native->Latin token dictionary (called in each worker)."""
    global TRANSLIT
    TRANSLIT = d or {}


def latinize(s: str) -> str:
    """Convert any script to lowercase ASCII: learned dictionary for Indic tokens, unidecode otherwise."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if INDIC.search(s):
        out = []
        for t in s.split():
            if INDIC.search(t):
                core = t.strip(".,;:()[]-")
                m = TRANSLIT.get(core)
                out.append(m if m is not None else unidecode(t))
            else:
                out.append(t)
        s = " ".join(out)
    if NON_ASCII.search(s):
        s = re.sub(r"[Nn]\s?[º°]", " no ", s).replace("º", " ").replace("°", " ")
        s = unidecode(s)
    return s.lower()


# ---------------------------------------------------------------- names
def _deleet(tok: str) -> str:
    """Undo digit-for-letter noise (sta1der, c0mmercial, 5arl) but keep real numbers/ordinals."""
    if tok.isdigit() or ORDINAL.match(tok) or not any(ch.isdigit() for ch in tok):
        return tok
    letters = sum(ch.isalpha() for ch in tok)
    if letters >= 2 and letters >= len(tok) - 2:
        return tok.translate(LEET)
    return tok


def norm_name(raw: str):
    """Return (full, core, is_web) for a business name.

    full: canonical tokens in original order (legal forms canonicalized, repeats squeezed)
    core: full minus legal forms and stop words (identity-bearing tokens)
    """
    s = latinize(raw)
    is_web = int(bool(re.search(r"(www\.|\.com\b|\.in\b|\.net\b|\.org\b|\.fr\b|^@)", s)))
    s = re.sub(r"\b(a/k/a|d/b/a|f/k/a|m/s|c/o)\b", " ", s)
    s = s.replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"\.(com|net|org|in|co|fr|info|biz|us)\b", " ", s)
    s = " ".join(_deleet(t) for t in s.split())
    s = PUNCT.sub(" ", s)
    toks = s.split()
    if len(toks) == 1 and len(toks[0]) > 7 and toks[0].endswith("com"):
        toks, is_web = [toks[0][:-3]], 1  # glued website name: "sarkdesigncom"
    # join runs of single letters: "l l c" -> "llc", "p t" -> "pt"
    joined, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            joined.append("".join(buf)); buf = []
        joined.append(t)
    if buf:
        joined.append("".join(buf))
    full = []
    for t in joined:
        t = LEGAL.get(t, t)
        if full and full[-1] == t:
            continue  # "engineering engineering" -> one token
        full.append(t)
    core = [t for t in full if t not in LEGAL_CANON and t not in NAME_STOP]
    if not core:
        core = [t for t in full if t not in NAME_STOP] or full
    # remove repeated tokens anywhere, keep first occurrence
    seen, core_u = set(), []
    for t in core:
        if t not in seen:
            seen.add(t); core_u.append(t)
    return " ".join(full), " ".join(core_u), is_web


# ---------------------------------------------------------------- addresses
def norm_addr(raw: str, country: str):
    """Return (tokens, numbers, state) for an address.

    tokens: canonical word tokens (abbreviations expanded, state names -> codes, filler dropped)
    numbers: sorted unique digit groups with leading zeros stripped, as a space-joined string
    state: detected state / region code or ''
    """
    if raw is None:
        return "", "", ""
    s = latinize(raw)
    s = s.replace("<null>", " ")
    nums = sorted({n.lstrip("0") or "0" for n in DIGITS.findall(s)})
    s = PUNCT.sub(" ", DIGITS.sub(" ", s))
    s = SPACES.sub(" ", s).strip()
    state = ""
    smap = STATE_MAPS.get(country)
    if smap:
        d, rx = smap
        m = rx.search(s)
        if m:
            state = d[m.group(1)]
            s = rx.sub(" ", s)
    abbr = ADDR_FR if country == "France" else None
    toks = []
    codes = STATE_CODES.get(country, ())
    for t in s.split():
        if abbr is not None and t in abbr:
            t = abbr[t]
        elif t in ADDR_COMMON:
            t = ADDR_COMMON[t]
        if not t:
            continue
        if smap and len(t) == 2 and t in codes:
            state = state or t
            continue
        if smap and t in smap[0] and len(t) <= 3:  # abbreviations like 'ori', 'tg'
            state = state or smap[0][t]
            continue
        if t in ADDR_FILLER:
            continue
        toks.append(t)
    # drop consecutive duplicates (city repeated as district)
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return " ".join(out), " ".join(nums), state
