"""Stage 1a — text canonicalisation shared by all three sources, both splits.

    raw ──NFKD──► romanise Indic scripts ──► strip Latin accents ──► lowercase, & → and
        ──► punctuation → space ──► join single-letter runs (l l c → llc) ──► collapse spaces

EDA findings behind each step (notebooks/01_eda.ipynb):
* 11–19% of satellite names are in one of 9 Indic scripts. All Indic Unicode blocks share the same
  internal layout (inherited from ISCII), so ONE table keyed by (codepoint & 0x7F) romanises
  Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam.
* Spurious accents (ÉLECTRONICS, Àmicale) → strip combining marks U+0300–U+036F only. Stripping
  all combining marks would also delete Indic vowel signs.
* Domain-style names, brackets, junk prefixes (`--`, `***`) and `L.L.C.` all vanish under the
  punctuation rule + the "squash" form (alphanumerics only).
"""
import re
import unicodedata

# ---- Indic romanisation (deterministic in-code table: no external data or services) ----------
_CONSONANTS = {
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'n', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j',
    0x1D: 'jh', 0x1E: 'n', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n', 0x24: 't',
    0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n', 0x29: 'n', 0x2A: 'p', 0x2B: 'f', 0x2C: 'b',
    0x2D: 'bh', 0x2E: 'm', 0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l',
    0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
}
_OTHER = {
    0x01: 'n', 0x02: 'n', 0x03: 'h',                                          # candrabindu, anusvara, visarga
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u', 0x0B: 'ri', 0x0C: 'li',
    0x0D: 'e', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au',  # vowels
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri', 0x44: 'ri', 0x45: 'e',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',            # vowel signs
    0x4E: 't', 0x70: 'n',                                                     # Bengali khanda-ta, Gurmukhi tippi
    0x7A: 'n', 0x7B: 'n', 0x7C: 'r', 0x7D: 'l', 0x7E: 'l', 0x7F: 'k',         # Malayalam chillus
    **{0x66 + d: str(d) for d in range(10)},                                  # native digits
}
_SCHWA_BEFORE = set(_CONSONANTS) | {0x01, 0x02, 0x03}   # a consonant followed by these keeps its inherent 'a'


def _is_indic(ch):
    return 0x0900 <= ord(ch) < 0x0D80


def romanise(s):
    """Approximate ISO-15919-style romanisation of any Indic script, e.g.
    'राम मार्केटिंग प्राइवेट लिमिटेड' -> 'ram marketing praivet limited'.
    Inherent 'a' is added only between consonants (word-final schwa is dropped, as in Hindi)."""
    if s.isascii():
        return s
    out = []
    for i, ch in enumerate(s):
        if not _is_indic(ch):
            out.append(ch)
            continue
        off = ord(ch) & 0x7F
        if off in _CONSONANTS:
            out.append(_CONSONANTS[off])
            nxt = s[i + 1] if i + 1 < len(s) else ''
            if nxt and _is_indic(nxt) and (ord(nxt) & 0x7F) in _SCHWA_BEFORE:
                out.append('a')
        else:
            out.append(_OTHER.get(off, ''))    # virama, nukta, length marks -> ''
    return ''.join(out)


# ---- canonical form -------------------------------------------------------------------------
_LATIN_MARKS = re.compile('[̀-ͯ]')
_PUNCT = re.compile(r'[^0-9a-z\s]')
_INITIALS = re.compile(r'(?<=\b\w) (?=\w\b)')      # 'l l c' -> 'llc', 'p c' -> 'pc'
_DOMAIN = re.compile(r'^\s*(www\.)?[\w-]+\.(com|net|org|in|co|fr|biz|info|us|io)\b', re.I)
TLDS = {'www', 'com', 'net', 'org', 'in', 'co', 'fr', 'biz', 'info', 'us', 'io'}


def canon(s):
    """Canonical lowercase ASCII form used for every comparison (see module docstring)."""
    s = romanise(unicodedata.normalize('NFKD', s))
    s = _LATIN_MARKS.sub('', s).lower().replace('&', ' and ')
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()   # drop any leftover non-ASCII
    s = _PUNCT.sub(' ', s)
    return _INITIALS.sub('', ' '.join(s.split()))


def is_domain(raw):
    """True for names like 'seguraclassicarmour.com' / 'Hmgreen.Com' (~3% of satellite names)."""
    return bool(_DOMAIN.match(raw))


def has_indic(raw):
    """True if the raw string contains any Indic-script character."""
    return any(_is_indic(ch) for ch in raw)


if __name__ == '__main__':
    # smallest runnable self-check of the tricky cases seen in EDA
    assert romanise('राम मार्केटिंग प्राइवेट लिमिटेड') == 'ram marketing praivet limited', romanise('राम मार्केटिंग प्राइवेट लिमिटेड')
    assert canon('CRESCENT ÉLECTRONICS-PRIVATE LIMITED') == 'crescent electronics private limited'
    assert canon('Probst & Duran Newhold L.L.C.') == 'probst and duran newhold llc'
    assert canon('-- Holloway Peak Inc [Seafood]') == 'holloway peak inc seafood'
    assert canon('#10816 Weston Dr.') == '10816 weston dr'
    assert is_domain('seguraclassicarmour.com') and not is_domain('Segura Classic Armour Corp')
    print('s1_normalise self-check OK')
