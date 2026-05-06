# -*- coding: utf-8 -*-
"""
Taiwanese (Tâi-gí / Hō-ló) POJ → phoneme g2p for GPT-SoVITS.

Input:  Pe̍h-ōe-jī (POJ) romanized Taiwanese, e.g. "Sin-tsng-á"
Output: phoneme sequence using `tw_` prefixed tokens (registered in symbols2_tw.py — the 1033-symbol tw vocab; symbols2.py is the zh-only 732-vocab and does NOT contain tw_* tokens, by design)

Token scheme (separate tones):
    tw_<initial>     for syllable onset (17 initials)
    tw_<final>       for nucleus + coda (277 possible)
    tw_T<tone>       for tone (1-9)

Each syllable yields up to 3 tokens: [initial?, final, tone].
Syllables with zero initial (e.g. "á") emit only [final, tone].
Syllabic m/ng (e.g. "m̄") emit [final, tone] (final = "m" or "ng").

This module is separate from `chinese.py` / `chinese2.py`; it does NOT
modify any existing phoneme handling. New tokens are appended at the
end of `symbols2.symbols` so that pretrained checkpoints remain
backward-compatible (their existing embedding rows keep the same IDs).
"""

import os
import re
import unicodedata

# ---------------------------------------------------------------------------
# POJ inventory
# ---------------------------------------------------------------------------

# Initials — order matters: longest-prefix-first when matching.
INITIALS = ['p', 'ph', 'b', 'm', 't', 'th', 'n', 'l',
            'k', 'kh', 'g', 'ng', 'h', 'ts', 'tsh', 's', 'j']

_INITIALS_SORTED = sorted(INITIALS, key=lambda x: -len(x))

# Old POJ digraphs we transparently map to modern spelling
_OLD_INITIAL_MAP = {
    'chh': 'tsh',
    'ch':  'ts',
}

# Nuclei (vowel cores), longest-prefix-first.
NUCLEI = ['iau', 'oai', 'uai',
          'ai', 'au', 'ia', 'io', 'iu', 'oa', 'oe', 'ua', 'ue', 'ui',
          'er', 'ir', 'oo',
          'a', 'e', 'i', 'o', 'u']

# Codas (suffixes after the nucleus). Empty string = open syllable.
CODAS = ['', 'm', 'n', 'ng', 'p', 't', 'k', 'h',
         'nn', 'mh', 'nh', 'ngh', 'nnh']

# Build the closed set of legal finals.
_VALID_FINALS = set()
for _nuc in NUCLEI:
    for _coda in CODAS:
        _VALID_FINALS.add(_nuc + _coda)
# Syllabic nasals
_VALID_FINALS.update({'m', 'ng', 'mh', 'ngh'})

FINALS = sorted(_VALID_FINALS)

# Tone diacritic (combining mark) → tone number.
_TONE_DIACRITIC = {
    '́': 2,  # ́  acute
    '̀': 3,  # ̀  grave
    '̂': 5,  # ̂  circumflex
    '̄': 7,  # ̄  macron
    '̍': 8,  # ̍  vertical line above
    '̋': 9,  # ̋  double acute (rare, varies by region)
}

# Punctuation we want to keep as phoneme separators (mapped to standard punctuation tokens).
_PUNCT_MAP = {
    '。': '.', '，': ',', '？': '?', '！': '!', '；': ',', '：': ',',
    '、': ',', '·': ',', '…': '…', '"': ',', '"': ',', "'": ',', "'": ',',
    '(': ',', ')': ',', '（': ',', '）': ',', '—': '-',
}

# Source punctuation we recognise verbatim
_KEEP_PUNCT = set('.,?!…-')


# ---------------------------------------------------------------------------
# Token names
# ---------------------------------------------------------------------------

def initial_token(ini):
    return 'tw_' + ini

def final_token(fin):
    return 'tw_' + fin

def tone_token(tone):
    return 'tw_T' + str(tone)


def get_taiwanese_symbols():
    """Return the list of phoneme tokens this module emits.

    Used by `symbols2.py` to extend the global vocabulary. Always returns
    the same deterministic ordering so token IDs are stable.
    """
    syms = []
    for ini in INITIALS:
        syms.append(initial_token(ini))
    for fin in FINALS:
        syms.append(final_token(fin))
    for t in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        syms.append(tone_token(t))
    return syms


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _normalize_syllable(syl):
    """NFD-decompose, extract tone diacritic, return (base_letters, tone)."""
    nfd = unicodedata.normalize('NFD', syl.lower())
    tone = 1
    out = []
    for ch in nfd:
        if ch in _TONE_DIACRITIC:
            tone = _TONE_DIACRITIC[ch]
        elif unicodedata.combining(ch):
            # Some other combining mark — discard.
            continue
        else:
            out.append(ch)
    base = ''.join(out)
    # Superscript ⁿ (U+207F) → -nn (alternate POJ nasalisation marker).
    base = base.replace('ⁿ', 'nn')
    # Strip apostrophes (syllable-boundary markers).
    base = base.replace("'", '').replace('’', '')
    # NFD also drops the combining-dot-above (U+0358) used in `o͘` to write
    # the POJ vowel `oo`. Restore: any `o` that originally had U+0358 → `oo`.
    # We re-scan the original string to detect it.
    if '͘' in unicodedata.normalize('NFD', syl):
        # Replace each o-with-dot pattern in NFD order.
        nfd2 = unicodedata.normalize('NFD', syl.lower())
        rebuilt = []
        i = 0
        while i < len(nfd2):
            ch = nfd2[i]
            if ch == 'o' and i + 1 < len(nfd2) and nfd2[i + 1] == '͘':
                rebuilt.append('oo')
                i += 2
            elif ch in _TONE_DIACRITIC or unicodedata.combining(ch):
                i += 1
            else:
                rebuilt.append(ch)
                i += 1
        base = ''.join(rebuilt).replace('ⁿ', 'nn').replace("'", '').replace('’', '')
    return base, tone


def _split_initial(base):
    """Greedy longest-prefix match. Returns (initial, rest)."""
    # Treat syllabic m / ng / mh / ngh as no-initial.
    if base in ('m', 'ng', 'mh', 'ngh'):
        return '', base
    # Map old-POJ digraphs first.
    for old, new in _OLD_INITIAL_MAP.items():
        if base.startswith(old):
            return new, base[len(old):]
    for ini in _INITIALS_SORTED:
        if base.startswith(ini) and len(base) > len(ini):
            return ini, base[len(ini):]
    return '', base


def _adjust_entering_tone(final, tone):
    """Tone 1 + stop coda (-p/-t/-k/-h) → tone 4 (lower entering)."""
    if final and final[-1] in 'ptkh' and tone == 1:
        return 4
    return tone


def parse_syllable(syl):
    """Parse one POJ syllable → (initial, final, tone). May return ('', '', None)."""
    base, tone = _normalize_syllable(syl)
    if not base:
        return '', '', None
    ini, fin = _split_initial(base)
    tone = _adjust_entering_tone(fin, tone)
    return ini, fin, tone


# ---------------------------------------------------------------------------
# Sentence-level g2p
# ---------------------------------------------------------------------------

# A token in the input text is either a POJ syllable or a piece of punctuation.
_TOKEN_RE = re.compile(
    r"([A-Za-zÀ-ɏ̀-ͯⁿ'’]+)"  # POJ word (letters + diacritics + ⁿ + apostrophes)
    r"|([\.,\?\!…\-:;])"
)


def text_normalize(text):
    """Light normalisation: map smart quotes / Chinese punctuation to ASCII."""
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    # Collapse the POJ enclitic marker `--` to a single hyphen.
    text = re.sub(r'-{2,}', '-', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def g2p(text):
    """Return list[str] of phoneme tokens for the input POJ text.

    Unknown finals are emitted as the literal `UNK` symbol (already in the
    base vocabulary) so the downstream pipeline does not crash.
    """
    text = text_normalize(text)
    out = []
    for m in _TOKEN_RE.finditer(text):
        word, punct = m.group(1), m.group(2)
        if punct:
            # Map ASCII-equivalent punctuation that exists in pu_symbols.
            if punct == '…':
                out.append('…')
            elif punct in '.,?!-':
                out.append(punct)
            continue
        # Words may contain hyphens (after smart-punct mapping above only
        # alphabetic + diacritics survive here, but defend anyway).
        for syl in re.split(r'[\-]+', word):
            if not syl:
                continue
            ini, fin, tone = parse_syllable(syl)
            if tone is None:
                continue
            if ini:
                tok = initial_token(ini)
                out.append(tok if _is_known(tok) else 'UNK')
            if fin:
                tok = final_token(fin)
                out.append(tok if _is_known(tok) else 'UNK')
            out.append(tone_token(tone))
    if not out:
        out = [',']
    return out


# Resolve symbol membership lazily so this module can be imported even
# before `symbols2_tw.symbols` finishes constructing.  We always validate
# against the tw 1033-vocab — when this module is in use, the tw vocab is
# the right reference.  The zh-only `symbols2.py` does NOT contain tw_*
# tokens, so referring to it here would mark every tw_* token as UNK.
_SYMBOL_SET = None
def _is_known(tok):
    global _SYMBOL_SET
    if _SYMBOL_SET is None:
        try:
            from text import symbols2_tw as _s2tw
            _SYMBOL_SET = set(_s2tw.symbols)
        except Exception:
            return True  # be permissive if symbols not yet importable
    return tok in _SYMBOL_SET


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    samples = [
        "Sin-tsng-á",
        "tsián-á",
        "bē-suah",
        "Báng-kah Tāi-tō",
        "Guá thīn--lí",
        "Lí hó-bô?",
        "Tâi-oân-uē",
        "Goá ài lí",
        "tha̍k-tsheh",
        "ho͘",
    ]
    print(f"Vocab: {len(get_taiwanese_symbols())} new tw_ tokens")
    for s in samples:
        print(f"  {s!r:35s} -> {g2p(s)}")
