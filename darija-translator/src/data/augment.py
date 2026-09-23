from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import normalize_english_for_darija

SUBJECT_ORDER = ["i", "you", "he", "we", "they"]

QUESTION_WORDS = {
    "what": "chno",
    "where": "fin",
    "why": "3lash",
    "when": "fou9ach",
    "how": "kifach",
}

VERB_STEMS = {
    "go": "mshi",
    "come": "ji",
    "eat": "akul",
    "drink": "shrb",
    "learn": "t3lm",
    "study": "qra",
    "read": "qra",
    "write": "ktb",
    "buy": "shri",
    "sell": "bi3",
    "help": "3awn",
    "understand": "fhem",
    "wait": "tsenna",
    "start": "bda",
    "love": "bghi",
    "want": "bghi",
}

WANT_FORMS = {
    "i": "bghit",
    "you": "bghiti",
    "he": "bgha",
    "we": "bghina",
    "they": "bghaw",
}

NOUNS = {
    "banana": "banana",
    "camera": "camera",
    "pizza": "pizza",
    "city": "mdina",
    "school": "madrasa",
    "house": "dar",
    "book": "ktab",
    "phone": "telefon",
}

PRON_SUFFIXES = {
    "my": "i",
    "your": "ek",
    "his": "u",
    "our": "na",
    "their": "hum",
}


def _prefixed(stem: str, prefix: str) -> str:
    if stem.startswith(prefix):
        return stem
    return f"{prefix}{stem}"


def conjugate_subject_verb(subject: str, stem: str) -> str:
    if subject == "i":
        return _prefixed(stem, "kan")
    if subject == "you":
        return _prefixed(stem, "kat")
    if subject == "he":
        return _prefixed(stem, "kay")
    if subject == "we":
        return f"{_prefixed(stem, 'kan')}w"
    if subject == "they":
        return f"{_prefixed(stem, 'kay')}w"
    return stem


def _pluralize_en(noun: str) -> str:
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return f"{noun[:-1]}ies"
    if noun.endswith(("s", "x", "z", "sh", "ch")):
        return f"{noun}es"
    return f"{noun}s"


def _pluralize_darija(noun: str) -> str:
    if noun.endswith("a"):
        return f"{noun[:-1]}at"
    return f"{noun}at"


def _build_augmented_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for verb_en, stem in VERB_STEMS.items():
        for subject in SUBJECT_ORDER:
            pairs.append((f"{subject} {verb_en}", conjugate_subject_verb(subject, stem)))

        pairs.append((f"i want to {verb_en}", _prefixed(stem, "n")))
        pairs.append((f"i would like to {verb_en}", _prefixed(stem, "n")))
        pairs.append((f"i want you to {verb_en}", f"bghitek {_prefixed(stem, 't')}"))
        pairs.append((f"i would like you to {verb_en}", f"bghitek {_prefixed(stem, 't')}"))
        pairs.append((f"i'd like you to {verb_en}", f"bghitek {_prefixed(stem, 't')}"))

        for q_en, q_dar in QUESTION_WORDS.items():
            pairs.append((f"{q_en} do i {verb_en}", f"{q_dar} {conjugate_subject_verb('i', stem)}"))
            pairs.append((f"{q_en} do you {verb_en}", f"{q_dar} {conjugate_subject_verb('you', stem)}"))

    for subject, want_form in WANT_FORMS.items():
        for q_en, q_dar in QUESTION_WORDS.items():
            pairs.append((f"{q_en} do {subject} want", f"{q_dar} {want_form}"))

    for en_noun, darija_noun in NOUNS.items():
        en_plural = _pluralize_en(en_noun)
        darija_plural = _pluralize_darija(darija_noun)

        pairs.append((en_noun, darija_noun))
        pairs.append((en_plural, darija_plural))
        pairs.append((f"the {en_plural}", darija_plural))

        for en_pron, suffix in PRON_SUFFIXES.items():
            pairs.append((f"{en_pron} {en_noun}", f"{darija_noun}{suffix}"))

    pairs.append(("what do i want", "chno bghit"))
    pairs.append(("what do you want", "chno bghiti"))
    pairs.append(("where do i go", "fin kanmshi"))
    pairs.append(("why do we wait", "3lash kantsennaw"))
    pairs.append(("how do they learn", "kifach kayt3lmw"))

    return pairs


def augment_parallel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    synthetic_pairs = _build_augmented_pairs()
    synthetic_df = pd.DataFrame(synthetic_pairs, columns=["en", "darija"])

    synthetic_df["en"] = synthetic_df["en"].map(normalize_english_for_darija)
    synthetic_df["darija"] = synthetic_df["darija"].str.strip()

    combined = pd.concat([df[["en", "darija"]], synthetic_df], ignore_index=True)
    combined["en"] = combined["en"].astype(str).map(normalize_english_for_darija)
    combined["darija"] = combined["darija"].astype(str).str.strip()
    combined = combined[(combined["en"].str.len() > 0) & (combined["darija"].str.len() > 0)]

    return combined.drop_duplicates(subset=["en", "darija"]).reset_index(drop=True)


def save_augmented_preview(df: pd.DataFrame, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
