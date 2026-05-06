import os
# if os.environ.get("version","v1")=="v1":
#   from text.symbols import symbols
# else:
#   from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2          # zh — 732, never grows
from text import symbols2_tw as symbols_v2_tw    # tw — 1033 (zh + tw_*)

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}
_symbol_to_id_v2_tw = {s: i for i, s in enumerate(symbols_v2_tw.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      version: "v1" → zh 322, "v2" → zh 732 (default), "v2tw" → tw 1033.
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        phones = [_symbol_to_id_v1[symbol] for symbol in cleaned_text]
    elif version == "v2tw":
        phones = [_symbol_to_id_v2_tw[symbol] for symbol in cleaned_text]
    else:
        phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

    return phones
