"""Rule based romanisation of the Indic scripts found in the data.

Source 2 and Source 3 often write Indian business names and state names in a
native script (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Gurmukhi,
Oriya, Malayalam) while Source 1 is always in Latin script. To compare them we
map every Indic character to an approximate Latin spelling.

All nine scripts are laid out in parallel 128-codepoint Unicode blocks (the
ISCII heritage), so a single offset table covers them. Consonants carry an
inherent "a" that is replaced by a following vowel sign or removed by a virama;
the word-final inherent "a" is dropped (schwa deletion), which is what English
loan words written in these scripts need ("लिमिटेड" -> "limited").

This is only the fallback: preprocess.py first looks words up in a lexicon
learned from the training pairs, and uses this module for words it has not seen.
No external library or data is used.
"""

from __future__ import annotations

# Unicode blocks of the Brahmic scripts used in the data.
_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

# Independent vowels (offset -> latin). Long vowels are collapsed to short ones
# because the English spelling rarely doubles them.
_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
# Consonants (offset -> latin, without the inherent vowel).
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh",
    0x5E: "f", 0x5F: "y",
}
# Dependent vowel signs (matras).
_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li",
}
_VIRAMA = 0x4D
_NUKTA = 0x3C
# Nasal / aspiration marks that follow a syllable.
_MARKS = {0x01: "n", 0x02: "n", 0x03: "h"}
# Malayalam chillu letters and Bengali khanda ta: consonants with no vowel.
_DEAD = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l",
         0x0D7F: "k", 0x09CE: "t", 0x0A70: "n"}


def _classify(cp: int):
    """Return (kind, latin) for an Indic code point, or (None, None)."""
    if cp in _DEAD:
        return "dead", _DEAD[cp]
    for base in _BLOCKS:
        if base <= cp < base + 0x80:
            off = cp - base
            if off in _CONSONANTS:
                return "cons", _CONSONANTS[off]
            if off in _MATRAS:
                return "matra", _MATRAS[off]
            if off in _VOWELS:
                return "vowel", _VOWELS[off]
            if off == _VIRAMA:
                return "virama", ""
            if off == _NUKTA:
                return "nukta", ""
            if off in _MARKS:
                return "mark", _MARKS[off]
            if 0x66 <= off <= 0x6F:
                return "digit", str(off - 0x66)
            if off in (0x64, 0x65):
                return "space", " "
            return "skip", ""
    return None, None


def is_indic(ch: str) -> bool:
    """True if ``ch`` belongs to one of the supported Indic blocks."""
    cp = ord(ch)
    return any(b <= cp < b + 0x80 for b in _BLOCKS) or cp in _DEAD


def has_indic(text: str) -> bool:
    """True if ``text`` contains at least one Indic character."""
    return any(is_indic(c) for c in text)


def romanize(text: str) -> str:
    """Romanise every Indic run inside ``text``; other characters pass through.

    >>> romanize("लिमिटेड")
    'limited'
    """
    # Malayalam writes the English "t" sound as a doubled / nasal-cluster RRA.
    text = text.replace("റ്റ", "ട്ട").replace(
        "ന്റ", "ന്ട")
    out = []
    pending_a = False  # a consonant was emitted and still owns its inherent "a"
    for ch in text:
        kind, lat = _classify(ord(ch))
        if kind is None:
            if pending_a:
                pending_a = False  # word-final schwa deletion
            out.append(ch)
            continue
        if kind == "cons":
            if pending_a:
                out.append("a")
            out.append(lat)
            pending_a = True
        elif kind == "matra":
            out.append(lat)
            pending_a = False
        elif kind == "virama":
            pending_a = False
        elif kind == "nukta" or kind == "skip":
            continue
        elif kind == "mark":
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(lat)
        else:  # vowel, digit, space, dead consonant
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(lat)
    return "".join(out)


if __name__ == "__main__":
    for w in ["राम मार्केटिंग प्राइवेट लिमिटेड", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
              "ಕರ್ನಾಟಕ", "గెలాక్సీ సాఫ్ట్‌వేర్ ప్రైవేట్ లిమిటెడ్", "ଶ୍ୟାମ ସିଷ୍ଟମସ୍",
              "ഈസ്റ്റ് മാനേജ്മെന്റ്", "தமிழ்நாடு", "ગુજરાત", "महाराष्ट्र"]:
        print(w, "->", romanize(w))
