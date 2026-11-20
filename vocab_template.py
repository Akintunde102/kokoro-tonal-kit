"""
178-Slot Atomic Vocabulary Builder for Tonal & Low-Resource Languages on Kokoro-82M.

Kokoro-82M has a fixed 178-slot embedding table (IDs 0..177):
  - Slot 0 is reserved (PAD / BOS / EOS).
  - 114 slots are occupied by BASE_KOKORO_VOCAB (IPA symbols & punctuation).
  - 63 remaining slots in 1..177 are vacant and safe for atomic (sound + tone) tokens.
"""

from typing import Dict, Iterable, List, Tuple

BASE_KOKORO_VOCAB: Dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, '"': 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19,
    "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23, "A": 24, "I": 25, "O": 31, "Q": 33,
    "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45,
    "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54,
    "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62,
    "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70,
    "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81,
    "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99,
    "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114,
    "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126,
    "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136,
    "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156,
    "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173,
    "ᵻ": 177,
}

FREE_SLOTS: List[int] = [i for i in range(1, 178) if i not in set(BASE_KOKORO_VOCAB.values())]


def build_atomic_vocab(
    atomic_symbols: Iterable[str],
    preassigned: Dict[str, int] | None = None,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Pack custom (sound + tone) symbols into Kokoro's 63 vacant slots (1..177)
    without resizing the 178-slot embedding table.

    Returns (full_vocab_dict, sorted_greedy_keys).
    """
    vocab = dict(BASE_KOKORO_VOCAB)
    used = set(vocab.values())
    if preassigned:
        for sym, slot in preassigned.items():
            if not (1 <= slot <= 177):
                raise ValueError(f"Slot {slot} for {sym!r} out of range 1..177")
            vocab[sym] = slot
            used.add(slot)

    free = iter([i for i in range(1, 178) if i not in used])
    for sym in atomic_symbols:
        if sym in vocab:
            continue
        try:
            vocab[sym] = next(free)
        except StopIteration as exc:
            raise ValueError("Exceeded Kokoro 178-slot table (63 custom slots available).") from exc

    greedy_keys = sorted([k for k in vocab if len(k) > 1], key=len, reverse=True)
    return vocab, greedy_keys


def tokenize_atomic(ipa_text: str, vocab: Dict[str, int], greedy_keys: List[str]) -> List[int]:
    """Greedy longest-match tokenizer ensuring multi-char (vowel+tone) units map to 1 token."""
    tokens: List[int] = []
    i = 0
    n = len(ipa_text)
    while i < n:
        for k in greedy_keys:
            if ipa_text.startswith(k, i):
                tokens.append(vocab[k])
                i += len(k)
                break
        else:
            ch = ipa_text[i]
            if ch in vocab:
                tokens.append(vocab[ch])
            i += 1
    return tokens


if __name__ == "__main__":
    assert len(BASE_KOKORO_VOCAB) == 114
    assert len(FREE_SLOTS) == 63
    v, keys = build_atomic_vocab(["aˊ", "aˋ", "ɔˊ", "ɔˋ", "kp", "ɡb"], preassigned={"aˉ": 43, "ɔˉ": 76})
    ids = tokenize_atomic("ɔˉkɔˋ", v, keys)
    assert len(ids) == 3, ids
    print(f"✅ vocab_template self-test passed! {len(FREE_SLOTS)} free slots available, sample ids={ids}")
