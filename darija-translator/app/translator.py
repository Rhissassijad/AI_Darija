import csv
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.data.preprocess import normalize_english_for_darija
except Exception:
    def normalize_english_for_darija(text: str) -> str:
        return " ".join(text.strip().split()).lower()


TASK_PREFIX = "translate English to Moroccan Darija: "
MODEL_PATH = PROJECT_ROOT / "models" / "final_model"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"

# rule | hybrid | model
TRANSLATOR_MODE = os.getenv("TRANSLATOR_MODE", "hybrid").strip().lower()

MOCK_DICTIONARY = {
    "hello": "Salam",
    "how are you": "Labas 3lik?",
    "thank you": "Shukran",
    "good morning": "Sbah lkhir",
    "good night": "Tsbah 3la khir",
}

WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
TOKEN_PARTS_PATTERN = re.compile(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)?([^A-Za-z0-9']*)$")
TRAILING_PUNCT_PATTERN = re.compile(r"([?.!]+)$")

QUESTION_WORDS = {
    "what": "chno",
    "where": "fin",
    "why": "3lash",
    "when": "fou9ach",
    "how": "kifach",
    "who": "chkun",
    "whom": "chkun",
    "whose": "dyal chkun",
    "which": "ashmen",
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
    "do": "dir",
    "make": "dir",
    "work": "khdm",
    "sleep": "n3s",
    "sit": "gls",
    "stand": "wqf",
    "speak": "hdr",
    "talk": "hdr",
    "listen": "sma3",
    "hear": "sma3",
    "see": "shuf",
    "watch": "shuf",
    "say": "gul",
    "tell": "gul",
    "open": "7l",
    "close": "sd",
    "play": "l3b",
    "live": "skn",
    "can": "qder",
    "able": "qder",
}

VERB_FORMS = {
    "go": {"np": "mshi", "np_pl": "mshiw", "past": "msha", "past_stem": "mshi", "imp": "sir"},
    "come": {"np": "ji", "np_pl": "jiw", "past": "ja", "past_stem": "ji", "imp": "aji"},
    "eat": {"np": "akul", "np_pl": "aklu", "past": "kla", "past_stem": "kli", "imp": "kul"},
    "drink": {"np": "shrb", "np_pl": "shrbu", "past": "shreb", "past_stem": "shreb", "imp": "shrb"},
    "learn": {"np": "t3lm", "np_pl": "t3lmu", "past": "t3lm", "past_stem": "t3lm", "imp": "t3lm"},
    "study": {"np": "qra", "np_pl": "qraw", "past": "qra", "past_stem": "qri", "imp": "qra"},
    "read": {"np": "qra", "np_pl": "qraw", "past": "qra", "past_stem": "qri", "imp": "qra"},
    "write": {"np": "ktb", "np_pl": "ktbu", "past": "kteb", "past_stem": "kteb", "imp": "kteb"},
    "buy": {"np": "shri", "np_pl": "shriw", "past": "shra", "past_stem": "shri", "imp": "shri"},
    "sell": {"np": "bi3", "np_pl": "bi3u", "past": "ba3", "past_stem": "bi3", "imp": "bi3"},
    "help": {"np": "3awn", "np_pl": "3awnu", "past": "3awn", "past_stem": "3awn", "imp": "3awn"},
    "understand": {"np": "fhem", "np_pl": "fhemu", "past": "fhem", "past_stem": "fhem", "imp": "fhem"},
    "wait": {"np": "tsenna", "np_pl": "tsennaw", "past": "tsenna", "past_stem": "tsenni", "imp": "tsenna"},
    "start": {"np": "bda", "np_pl": "bdaw", "past": "bda", "past_stem": "bdi", "imp": "bda"},
    "love": {"np": "bghi", "np_pl": "bghiw", "past": "bgha", "past_stem": "bghi", "imp": "bghi"},
    "want": {"np": "bghi", "np_pl": "bghiw", "past": "bgha", "past_stem": "bghi", "imp": "bghi"},
    "do": {"np": "dir", "np_pl": "diru", "past": "dar", "past_stem": "dr", "imp": "dir"},
    "make": {"np": "dir", "np_pl": "diru", "past": "dar", "past_stem": "dr", "imp": "dir"},
    "work": {"np": "khdm", "np_pl": "khdmu", "past": "khdm", "past_stem": "khdm", "imp": "khdm"},
    "sleep": {"np": "n3s", "np_pl": "n3su", "past": "n3s", "past_stem": "n3s", "imp": "n3s"},
    "sit": {"np": "gls", "np_pl": "glsu", "past": "gls", "past_stem": "gls", "imp": "gls"},
    "stand": {"np": "wqf", "np_pl": "wqfu", "past": "wqf", "past_stem": "wqf", "imp": "wqf"},
    "speak": {"np": "hdr", "np_pl": "hdru", "past": "hdr", "past_stem": "hdr", "imp": "hdr"},
    "talk": {"np": "hdr", "np_pl": "hdru", "past": "hdr", "past_stem": "hdr", "imp": "hdr"},
    "listen": {"np": "sma3", "np_pl": "sma3u", "past": "sma3", "past_stem": "sma3", "imp": "sma3"},
    "hear": {"np": "sma3", "np_pl": "sma3u", "past": "sma3", "past_stem": "sma3", "imp": "sma3"},
    "see": {"np": "shuf", "np_pl": "shufu", "past": "shaf", "past_stem": "shf", "imp": "shuf"},
    "watch": {"np": "shuf", "np_pl": "shufu", "past": "shaf", "past_stem": "shf", "imp": "shuf"},
    "say": {"np": "gul", "np_pl": "gulu", "past": "gal", "past_stem": "gl", "imp": "gul"},
    "tell": {"np": "gul", "np_pl": "gulu", "past": "gal", "past_stem": "gl", "imp": "gul"},
    "open": {"np": "7l", "np_pl": "7lu", "past": "7l", "past_stem": "7l", "imp": "7l"},
    "close": {"np": "sd", "np_pl": "sdu", "past": "sd", "past_stem": "sd", "imp": "sd"},
    "play": {"np": "l3b", "np_pl": "l3bu", "past": "l3b", "past_stem": "l3b", "imp": "l3b"},
    "live": {"np": "skn", "np_pl": "sknu", "past": "skn", "past_stem": "skn", "imp": "skn"},
    "can": {"np": "qder", "np_pl": "qdru", "past": "qder", "past_stem": "qder", "imp": "qder"},
    "able": {"np": "qder", "np_pl": "qdru", "past": "qder", "past_stem": "qder", "imp": "qder"},
}

WANT_FORMS = {
    "i": "bghit",
    "you": "bghiti",
    "he": "bgha",
    "she": "bghat",
    "we": "bghina",
    "you_plural": "bghitu",
    "they": "bghaw",
}

SUBJECT_ALIASES = {
    "i": "i",
    "me": "i",
    "you": "you",
    "he": "he",
    "him": "he",
    "she": "she",
    "her": "she",
    "it": "he",
    "we": "we",
    "us": "we",
    "they": "they",
    "them": "they",
}

SUBJECT_PRONOUNS = {
    "i": "ana",
    "you": "nta",
    "he": "huwa",
    "she": "hiya",
    "we": "7na",
    "you_plural": "ntuma",
    "they": "huma",
}

POSSESSIVE_FORMS = {
    "my": "dyali",
    "your": "dyalk",
    "his": "dyalu",
    "her": "dyalha",
    "our": "dyalna",
    "their": "dyalhum",
}

HAVE_FORMS = {
    "i": "3endi",
    "you": "3endk",
    "he": "3endu",
    "she": "3endha",
    "we": "3endna",
    "you_plural": "3endkum",
    "they": "3endhum",
}

NEED_FORMS = {
    "i": "khassni",
    "you": "khassk",
    "he": "khassu",
    "she": "khassha",
    "we": "khassna",
    "you_plural": "khasskum",
    "they": "khasshum",
}

BE_PAST_FORMS = {
    "i": "kunt",
    "you": "kunti",
    "he": "kan",
    "she": "kanet",
    "we": "kunna",
    "you_plural": "kuntu",
    "they": "kanu",
}

ENGLISH_VERB_ALIASES = {
    "am": "be",
    "are": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "been": "be",
    "has": "have",
    "had": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "made": "make",
    "went": "go",
    "gone": "go",
    "came": "come",
    "ate": "eat",
    "eaten": "eat",
    "drank": "drink",
    "drunk": "drink",
    "learnt": "learn",
    "learned": "learn",
    "studied": "study",
    "wrote": "write",
    "written": "write",
    "bought": "buy",
    "sold": "sell",
    "helped": "help",
    "understood": "understand",
    "waited": "wait",
    "started": "start",
    "loved": "love",
    "wanted": "want",
    "worked": "work",
    "slept": "sleep",
    "sat": "sit",
    "stood": "stand",
    "spoke": "speak",
    "spoken": "speak",
    "talked": "talk",
    "listened": "listen",
    "heard": "hear",
    "saw": "see",
    "seen": "see",
    "said": "say",
    "told": "tell",
    "opened": "open",
    "closed": "close",
    "played": "play",
    "lived": "live",
}

PAST_MARKERS = set(ENGLISH_VERB_ALIASES) - {"am", "are", "is", "has", "does"}

WORD_TRANSLATIONS = {
    "a": "",
    "an": "",
    "the": "l",
    "to": "l",
    "in": "f",
    "from": "mn",
    "with": "m3a",
    "and": "w",
    "or": "wlla",
    "but": "walakin",
    "now": "daba",
    "today": "lyum",
    "tomorrow": "ghedda",
    "yesterday": "lbare7",
    "here": "hna",
    "there": "tema",
    "home": "dar",
    "house": "dar",
    "school": "lmadrasa",
    "market": "souq",
    "city": "mdina",
    "morocco": "lmeghrib",
    "marrakech": "Marrakech",
    "casablanca": "Casablanca",
    "rabat": "Rabat",
    "darija": "darija",
    "arabic": "l3rbiya",
    "english": "lngliziya",
    "french": "fransawiya",
    "coffee": "qahwa",
    "tea": "atay",
    "water": "lma",
    "bread": "khobz",
    "food": "makla",
    "couscous": "seksu",
    "money": "flous",
    "time": "lweqt",
    "book": "ktab",
    "letter": "brriya",
    "car": "tomobil",
    "bus": "tobis",
    "train": "tran",
    "problem": "mushkil",
    "name": "smiya",
    "friend": "sahb",
    "teacher": "ustad",
    "student": "talib",
    "work": "lkhdma",
    "good": "mezyan",
    "bad": "khayb",
    "big": "kbir",
    "small": "sghir",
    "beautiful": "zwin",
    "nice": "zwin",
    "sick": "mrid",
    "hungry": "ji3an",
    "thirsty": "3tshan",
    "tired": "3yan",
    "happy": "far7an",
    "hot": "skhun",
    "cold": "bard",
    "ready": "wajed",
    "quickly": "bzerba",
    "slowly": "bshwiya",
    "please": "afak",
    "yes": "iyeh",
    "no": "la",
    "nothing": "walu",
    "nobody": "7ta wa7ed",
}

FEMININE_ADJECTIVES = {
    "mezyan": "mezyana",
    "khayb": "khayba",
    "kbir": "kbira",
    "sghir": "sghira",
    "zwin": "zwina",
    "mrid": "mrida",
    "ji3an": "ji3ana",
    "3tshan": "3tshana",
    "3yan": "3yana",
    "far7an": "far7ana",
    "skhun": "skhuna",
    "bard": "barda",
    "wajed": "wajda",
}


def _model_ready(path: Path) -> bool:
    required = ["config.json", "tokenizer_config.json"]
    return all((path / f).exists() for f in required)


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def _normalize_phrase(text: str) -> str:
    normalized = normalize_english_for_darija(text)
    return " ".join(WORD_PATTERN.findall(normalized.lower()))


def _split_token_parts(token: str) -> dict:
    match = TOKEN_PARTS_PATTERN.match(token)
    if not match:
        return {"token": token, "prefix": "", "core": "", "suffix": "", "norm": ""}

    prefix, core, suffix = match.groups()
    core = core or ""
    return {
        "token": token,
        "prefix": prefix or "",
        "core": core,
        "suffix": suffix or "",
        "norm": core.lower(),
    }


def _prefixed(stem: str, prefix: str) -> str:
    if stem.startswith(prefix):
        return stem
    return f"{prefix}{stem}"


def _verb_forms(verb: str) -> dict[str, str] | None:
    lemma = _english_verb_lemma(verb)
    if lemma in VERB_FORMS:
        return VERB_FORMS[lemma]

    stem = VERB_STEMS.get(lemma)
    if not stem:
        return None

    return {
        "np": stem,
        "np_pl": f"{stem}u",
        "past": stem,
        "past_stem": stem,
        "imp": stem,
    }


def _nonpast_stem(verb: str, plural: bool = False) -> str | None:
    forms = _verb_forms(verb)
    if not forms:
        return None
    return forms["np_pl"] if plural else forms["np"]


def _conjugate_subject_verb(subject: str, stem: str) -> str:
    if subject == "i":
        return _prefixed(stem, "kan")
    if subject == "you":
        return _prefixed(stem, "kat")
    if subject == "he":
        return _prefixed(stem, "kay")
    if subject == "she":
        return _prefixed(stem, "kat")
    if subject == "we":
        return f"{_prefixed(stem, 'kan')}w"
    if subject == "you_plural":
        return f"{_prefixed(stem, 'kat')}w"
    if subject == "they":
        return f"{_prefixed(stem, 'kay')}w"
    return stem


def _conjugate_present(subject: str, verb: str, habitual: bool = True) -> str | None:
    plural = subject in {"we", "you_plural", "they"}
    stem = _nonpast_stem(verb, plural=plural)
    if not stem:
        return None

    if subject == "i":
        prefix = "kan" if habitual else "n"
    elif subject in {"you", "she"}:
        prefix = "kat" if habitual else "t"
    elif subject == "he":
        prefix = "kay" if habitual else "y"
    elif subject == "we":
        prefix = "kan" if habitual else "n"
    elif subject == "you_plural":
        prefix = "kat" if habitual else "t"
    elif subject == "they":
        prefix = "kay" if habitual else "y"
    else:
        prefix = "ka"

    return _prefixed(stem, prefix)


def _conjugate_subjunctive(subject: str, verb: str) -> str | None:
    return _conjugate_present(subject, verb, habitual=False)


def _conjugate_past(subject: str, verb: str) -> str | None:
    forms = _verb_forms(verb)
    if not forms:
        return None

    past = forms["past"]
    stem = forms["past_stem"]
    if subject == "i":
        return f"{stem}t"
    if subject == "you":
        return f"{stem}ti"
    if subject == "he":
        return past
    if subject == "she":
        return f"{past}t" if past.endswith(("a", "i")) else f"{past}at"
    if subject == "we":
        return f"{stem}na"
    if subject == "you_plural":
        return f"{stem}tu"
    if subject == "they":
        return f"{past}w" if past.endswith(("a", "i")) else f"{past}u"
    return past


def _negate_verb(conjugated: str) -> str:
    if not conjugated:
        return conjugated
    return f"ma-{conjugated}sh"


def _english_verb_lemma(word: str) -> str:
    w = word.lower()
    if w in ENGLISH_VERB_ALIASES:
        return ENGLISH_VERB_ALIASES[w]
    if w in VERB_STEMS:
        return w

    if w.endswith("ies") and len(w) > 3:
        candidate = f"{w[:-3]}y"
        if candidate in VERB_STEMS:
            return candidate

    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            candidate = w[: -len(suffix)]
            if candidate in VERB_STEMS:
                return candidate

    return w


def _stem_for_word(word: str) -> str | None:
    lemma = _english_verb_lemma(word)
    return VERB_STEMS.get(lemma)


def _build_grammar_phrase_pairs() -> dict[str, str]:
    pairs: dict[str, str] = {}
    subjects = ["i", "you", "he", "she", "we", "they"]

    for en_verb, stem in VERB_STEMS.items():
        for subject in subjects:
            conjugated = _conjugate_present(subject, en_verb)
            if conjugated:
                pairs[f"{subject} {en_verb}"] = conjugated

        pairs[f"i want to {en_verb}"] = f"bghit {_prefixed(stem, 'n')}"
        pairs[f"i would like to {en_verb}"] = f"bghit {_prefixed(stem, 'n')}"
        pairs[f"i want you to {en_verb}"] = f"bghitek {_prefixed(stem, 't')}"
        pairs[f"i would like you to {en_verb}"] = f"bghitek {_prefixed(stem, 't')}"
        pairs[f"i'd like you to {en_verb}"] = f"bghitek {_prefixed(stem, 't')}"

        for q_en, q_dar in QUESTION_WORDS.items():
            pairs[f"{q_en} do i {en_verb}"] = f"{q_dar} {_conjugate_subject_verb('i', stem)}"
            pairs[f"{q_en} do you {en_verb}"] = f"{q_dar} {_conjugate_subject_verb('you', stem)}"

    for subject, want_form in WANT_FORMS.items():
        for q_en, q_dar in QUESTION_WORDS.items():
            pairs[f"{q_en} do {subject} want"] = f"{q_dar} {want_form}"

    pairs["what do i want"] = "chno bghit"
    return pairs


@lru_cache(maxsize=1)
def _load_phrase_dictionary() -> tuple[dict, int]:
    phrase_map = {}

    for en, darija in MOCK_DICTIONARY.items():
        key = _normalize_phrase(en)
        if key:
            phrase_map[key] = darija

    for en, darija in _build_grammar_phrase_pairs().items():
        key = _normalize_phrase(en)
        if key:
            phrase_map[key] = darija

    if DATASET_PATH.exists():
        with DATASET_PATH.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                en = (row.get("en") or "").strip()
                darija = (row.get("darija") or "").strip()
                key = _normalize_phrase(en)
                if key and darija:
                    phrase_map[key] = darija

    max_len = max((len(k.split()) for k in phrase_map), default=1)
    return phrase_map, max_len


@lru_cache(maxsize=1)
def _load_dataset_phrase_map() -> dict[str, str]:
    dataset_map: dict[str, str] = {}
    if not DATASET_PATH.exists():
        return dataset_map

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            en = (row.get("en") or "").strip()
            darija = (row.get("darija") or "").strip()
            key = _normalize_phrase(en)
            if key and darija:
                # Latest row wins, so app corrections can override old lines.
                dataset_map[key] = darija
    return dataset_map


def get_dataset_translation(text: str) -> str | None:
    key = _normalize_phrase(text)
    if not key:
        return None
    return _load_dataset_phrase_map().get(key)


def save_translation_pair(english_text: str, darija_text: str) -> bool:
    en = english_text.strip()
    darija = darija_text.strip()
    if not en or not darija:
        return False

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DATASET_PATH.exists()

    with DATASET_PATH.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["en", "darija"])
        writer.writerow([en, darija])

    _load_dataset_phrase_map.cache_clear()
    _load_phrase_dictionary.cache_clear()
    return True


def _decorate_translation(translation: str, prefix: str, suffix: str) -> str:
    output = translation
    if prefix and not output.startswith(prefix):
        output = f"{prefix}{output}"
    if suffix and not output.endswith(suffix):
        output = f"{output}{suffix}"
    return output


def _span_is_phrase_compatible(token_infos: list, start: int, size: int) -> bool:
    end = start + size
    span = token_infos[start:end]
    if any(not token["norm"] for token in span):
        return False

    if size == 1:
        return True

    if span[0]["suffix"]:
        return False
    if span[-1]["prefix"]:
        return False

    for token in span[1:-1]:
        if token["prefix"] or token["suffix"]:
            return False

    return True


def _append_trailing_punct(source_text: str, translated: str) -> str:
    if not translated:
        return translated

    match = TRAILING_PUNCT_PATTERN.search(source_text.strip())
    if not match:
        return translated

    punct = match.group(1)
    if translated.endswith(punct):
        return translated
    return f"{translated}{punct}"


def _normalized_words(text: str) -> list[str]:
    words = _normalize_phrase(text).split()
    contractions = {
        "i'm": ["i", "am"],
        "you're": ["you", "are"],
        "he's": ["he", "is"],
        "she's": ["she", "is"],
        "it's": ["it", "is"],
        "we're": ["we", "are"],
        "they're": ["they", "are"],
        "i'll": ["i", "will"],
        "you'll": ["you", "will"],
        "he'll": ["he", "will"],
        "she'll": ["she", "will"],
        "we'll": ["we", "will"],
        "they'll": ["they", "will"],
        "i'd": ["i", "would"],
        "you'd": ["you", "would"],
        "he'd": ["he", "would"],
        "she'd": ["she", "would"],
        "we'd": ["we", "would"],
        "they'd": ["they", "would"],
        "don't": ["do", "not"],
        "doesn't": ["does", "not"],
        "didn't": ["did", "not"],
        "can't": ["can", "not"],
        "cannot": ["can", "not"],
        "won't": ["will", "not"],
        "wouldn't": ["would", "not"],
        "shouldn't": ["should", "not"],
        "mustn't": ["must", "not"],
        "isn't": ["is", "not"],
        "aren't": ["are", "not"],
        "wasn't": ["was", "not"],
        "weren't": ["were", "not"],
        "gonna": ["going", "to"],
    }

    expanded: list[str] = []
    for word in words:
        expanded.extend(contractions.get(word, [word]))
    return expanded


def _subject_for_word(word: str) -> str | None:
    return SUBJECT_ALIASES.get(word)


def _is_past_verb_word(word: str) -> bool:
    return word in PAST_MARKERS or word.endswith("ed")


def _translate_tail(words: list[str]) -> str:
    translated = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in POSSESSIVE_FORMS and i + 1 < len(words):
            noun = _translate_tail([words[i + 1]])
            if noun:
                translated.append(f"{noun} {POSSESSIVE_FORMS[word]}")
            i += 2
            continue

        if word in SUBJECT_PRONOUNS:
            translated.append(SUBJECT_PRONOUNS[word])
        else:
            mapped = WORD_TRANSLATIONS.get(word, word)
            if mapped:
                translated.append(mapped)
        i += 1
    return " ".join(translated)


def _agree_adjective(word: str, subject: str | None) -> str:
    translated = WORD_TRANSLATIONS.get(word, word)
    if subject == "she":
        return FEMININE_ADJECTIVES.get(translated, f"{translated}a" if translated == word else translated)
    return translated


def _compose(head: str | None, tail_words: list[str] | None = None) -> str | None:
    if not head:
        return None
    tail = _translate_tail(tail_words or [])
    return f"{head} {tail}".strip()


def _translate_modal(subject: str, modal: str, verb: str, tail: list[str], negative: bool = False) -> str | None:
    if modal in {"can", "could"}:
        ability = _conjugate_subjunctive(subject, "can")
        action = _conjugate_subjunctive(subject, verb)
        if not ability or not action:
            return None
        if negative:
            ability = _negate_verb(ability)
        return _compose(f"{ability} {action}", tail)

    if modal in {"must", "should"}:
        action = _conjugate_subjunctive(subject, verb)
        if not action:
            return None
        need = NEED_FORMS.get(subject)
        if not need:
            return None
        if negative:
            return _compose(f"ma-{need}sh {action}", tail)
        return _compose(f"{need} {action}", tail)

    return None


def _translate_subject_verb(
    subject: str,
    verb: str,
    tail: list[str],
    tense: str = "present",
    negative: bool = False,
) -> str | None:
    if verb in {"be", "am", "are", "is", "was", "were"}:
        complement = tail[0] if tail else ""
        rest = tail[1:] if tail else []
        adjective = _agree_adjective(complement, subject) if complement else ""
        translated_tail = _translate_tail(rest)
        body = " ".join(part for part in [adjective, translated_tail] if part).strip()

        if tense == "past":
            head = BE_PAST_FORMS.get(subject)
            if not head:
                return None
            if negative:
                head = _negate_verb(head)
            return f"{head} {body}".strip()
        pronoun = SUBJECT_PRONOUNS.get(subject, "")
        if negative:
            return f"{pronoun} mashi {body}".strip()
        return f"{pronoun} {body}".strip()

    if verb == "have":
        have = HAVE_FORMS.get(subject)
        if not have:
            return None
        if negative:
            have = _negate_verb(have)
        return _compose(have, tail)

    if verb == "want":
        want = WANT_FORMS.get(subject)
        if not want:
            return None
        if len(tail) >= 3 and _subject_for_word(tail[0]) and tail[1] == "to":
            object_subject = _subject_for_word(tail[0])
            action = _conjugate_subjunctive(object_subject, tail[2])
            if action:
                if subject == "i" and object_subject == "you":
                    want_object = "bghitek"
                else:
                    want_object = f"{want} {SUBJECT_PRONOUNS.get(object_subject, tail[0])}"
                return _compose(want_object if not negative else _negate_verb(want_object), [action] + tail[3:])
        if tail and tail[0] == "to" and len(tail) > 1:
            action = _conjugate_subjunctive(subject, tail[1])
            if action:
                return _compose(want if not negative else _negate_verb(want), [action] + tail[2:])
        return _compose(want if not negative else _negate_verb(want), tail)

    if verb in {"need", "must", "should"}:
        action_words = tail[1:] if tail[:1] == ["to"] else tail
        action = _conjugate_subjunctive(subject, action_words[0]) if action_words else None
        need = NEED_FORMS.get(subject)
        if not need:
            return None
        if action:
            phrase = f"{need} {action}"
            return _compose(phrase if not negative else f"ma-{need}sh {action}", action_words[1:])
        return _compose(need if not negative else f"ma-{need}sh", tail)

    if tense == "past":
        conjugated = _conjugate_past(subject, verb)
    elif tense == "future":
        future_verb = _conjugate_subjunctive(subject, verb)
        conjugated = f"ghadi {future_verb}" if future_verb else None
    else:
        conjugated = _conjugate_present(subject, verb)

    if not conjugated:
        return None
    if negative:
        conjugated = _negate_verb(conjugated.replace(" ", "-"))
    return _compose(conjugated, tail)


def _try_possession(words: list[str]) -> str | None:
    if len(words) >= 2 and words[0] in POSSESSIVE_FORMS:
        noun = _translate_tail([words[1]])
        tail = _translate_tail(words[2:])
        return " ".join(part for part in [noun, POSSESSIVE_FORMS[words[0]], tail] if part)
    return None


def _try_question(words: list[str]) -> str | None:
    if not words:
        return None

    qword = None
    if words[0] in QUESTION_WORDS:
        qword = QUESTION_WORDS[words[0]]
        words = words[1:]
    elif words[0] in {"do", "does", "did", "will", "can", "could", "should", "must", "is", "are", "was", "were"}:
        qword = "wach"

    if not qword or not words:
        return None

    aux = words[0]
    if qword == "kifach" and aux in {"are", "is"} and len(words) >= 2 and _subject_for_word(words[1]):
        return None

    if aux in {"do", "does", "did", "will"} and len(words) >= 3:
        subject = _subject_for_word(words[1])
        if not subject:
            return None
        verb = _english_verb_lemma(words[2])
        tense = "past" if aux == "did" else "future" if aux == "will" else "present"
        phrase = _translate_subject_verb(subject, verb, words[3:], tense=tense)
        return f"{qword} {phrase}".strip() if phrase else None

    if aux in {"can", "could", "should", "must"} and len(words) >= 3:
        subject = _subject_for_word(words[1])
        if not subject:
            return None
        phrase = _translate_modal(subject, aux, _english_verb_lemma(words[2]), words[3:])
        return f"{qword} {phrase}".strip() if phrase else None

    if aux in {"is", "are", "was", "were"} and len(words) >= 2:
        subject = _subject_for_word(words[1])
        if not subject:
            return None
        tense = "past" if aux in {"was", "were"} else "present"
        phrase = _translate_subject_verb(subject, "be", words[2:], tense=tense)
        return f"{qword} {phrase}".strip() if phrase else None

    if len(words) >= 2:
        subject = _subject_for_word(words[0])
        if subject:
            phrase = _translate_subject_verb(subject, _english_verb_lemma(words[1]), words[2:])
            return f"{qword} {phrase}".strip() if phrase else None

    return None


def _grammar_sentence_guess(text: str) -> str | None:
    words = _normalized_words(text)
    if not words:
        return None

    possession = _try_possession(words)
    if possession:
        return _append_trailing_punct(text, possession)

    question = _try_question(words)
    if question:
        return _append_trailing_punct(text, question)

    if words[:2] == ["there", "is"]:
        if len(words) >= 3 and words[2] in {"no", "not"}:
            return _append_trailing_punct(text, _compose("ma-kaynsh", words[3:]) or "ma-kaynsh")
        return _append_trailing_punct(text, _compose("kayn", words[2:]) or "kayn")

    if words[:3] == ["there", "are", "no"]:
        return _append_trailing_punct(text, _compose("ma-kaynsh", words[3:]) or "ma-kaynsh")

    if words[:2] == ["do", "not"] and len(words) >= 3:
        action = _conjugate_subjunctive("you", _english_verb_lemma(words[2]))
        if action:
            return _append_trailing_punct(text, _compose(_negate_verb(action), words[3:]) or _negate_verb(action))

    if words[0] in {"please"} and len(words) >= 2:
        forms = _verb_forms(words[1])
        if forms:
            phrase = _compose(forms["imp"], words[2:])
            return _append_trailing_punct(text, f"{phrase} afak")

    if len(words) >= 2 and words[0] in {"can", "could", "should", "must"}:
        subject = _subject_for_word(words[1])
        if subject and len(words) >= 3:
            phrase = _translate_modal(subject, words[0], _english_verb_lemma(words[2]), words[3:])
            if phrase:
                return _append_trailing_punct(text, phrase)

    subject = _subject_for_word(words[0])
    if subject:
        if len(words) >= 4 and words[1] in {"do", "does"} and words[2] == "not":
            phrase = _translate_subject_verb(subject, _english_verb_lemma(words[3]), words[4:], negative=True)
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 4 and words[1] == "did" and words[2] == "not":
            phrase = _translate_subject_verb(
                subject,
                _english_verb_lemma(words[3]),
                words[4:],
                tense="past",
                negative=True,
            )
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 4 and words[1] == "will" and words[2] == "not":
            phrase = _translate_subject_verb(
                subject,
                _english_verb_lemma(words[3]),
                words[4:],
                tense="future",
                negative=True,
            )
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 5 and words[1:4] in (["am", "going", "to"], ["is", "going", "to"], ["are", "going", "to"]):
            phrase = _translate_subject_verb(subject, _english_verb_lemma(words[4]), words[5:], tense="future")
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] == "will":
            phrase = _translate_subject_verb(subject, _english_verb_lemma(words[2]), words[3:], tense="future")
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 4 and words[1] in {"can", "could", "should", "must"} and words[2] == "not":
            phrase = _translate_modal(subject, words[1], _english_verb_lemma(words[3]), words[4:], negative=True)
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] in {"can", "could", "should", "must"}:
            phrase = _translate_modal(subject, words[1], _english_verb_lemma(words[2]), words[3:])
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 4 and words[1] in {"am", "are", "is"} and words[2] == "not":
            phrase = _translate_subject_verb(subject, "be", words[3:], negative=True)
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] in {"am", "are", "is"}:
            next_verb = _english_verb_lemma(words[2])
            if words[2].endswith("ing") and next_verb in VERB_FORMS:
                phrase = _translate_subject_verb(subject, next_verb, words[3:])
            else:
                phrase = _translate_subject_verb(subject, "be", words[2:])
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 4 and words[1] in {"was", "were"} and words[2] == "not":
            phrase = _translate_subject_verb(subject, "be", words[3:], tense="past", negative=True)
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] in {"was", "were"}:
            phrase = _translate_subject_verb(subject, "be", words[2:], tense="past")
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] in {"have", "has"}:
            phrase = _translate_subject_verb(subject, "have", words[2:])
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 3 and words[1] == "had":
            phrase = _translate_subject_verb(subject, "have", words[2:], tense="past")
            if phrase:
                return _append_trailing_punct(text, phrase)

        if len(words) >= 2:
            verb = _english_verb_lemma(words[1])
            tense = "past" if _is_past_verb_word(words[1]) else "present"
            phrase = _translate_subject_verb(subject, verb, words[2:], tense=tense)
            if phrase:
                return _append_trailing_punct(text, phrase)

    if len(words) >= 1:
        forms = _verb_forms(words[0])
        if forms:
            return _append_trailing_punct(text, _compose(forms["imp"], words[1:]) or forms["imp"])

    if len(words) >= 5 and words[:4] in (
        ["i", "want", "you", "to"],
        ["i", "would", "like", "you"],
    ):
        if words[:4] == ["i", "would", "like", "you"]:
            if len(words) < 6 or words[4] != "to":
                return None
            verb_idx = 5
            tail_start = 6
        else:
            verb_idx = 4
            tail_start = 5

        stem = _stem_for_word(words[verb_idx])
        verb = _prefixed(stem, "t") if stem else words[verb_idx]
        tail = " ".join(words[tail_start:])
        out = f"bghitek {verb}".strip()
        if tail:
            out = f"{out} {tail}"
        return _append_trailing_punct(text, out)

    if len(words) >= 5 and words[:4] == ["i", "would", "like", "to"]:
        stem = _stem_for_word(words[4])
        verb = _prefixed(stem, "n") if stem else words[4]
        tail = " ".join(words[5:])
        out = f"bghit {verb}".strip()
        if tail:
            out = f"{out} {tail}"
        return _append_trailing_punct(text, out)

    if len(words) >= 5 and words[:4] == ["i'd", "like", "you", "to"]:
        stem = _stem_for_word(words[4])
        verb = _prefixed(stem, "t") if stem else words[4]
        tail = " ".join(words[5:])
        out = f"bghitek {verb}".strip()
        if tail:
            out = f"{out} {tail}"
        return _append_trailing_punct(text, out)

    if len(words) >= 4 and words[:3] == ["i", "want", "to"]:
        stem = _stem_for_word(words[3])
        verb = _prefixed(stem, "n") if stem else words[3]
        tail = " ".join(words[4:])
        out = f"bghit {verb}".strip()
        if tail:
            out = f"{out} {tail}"
        return _append_trailing_punct(text, out)

    if len(words) >= 3 and words[0] in QUESTION_WORDS and words[1] in WANT_FORMS:
        q = QUESTION_WORDS[words[0]]
        subject = words[1]
        verb = words[2]

        if verb == "want":
            head = WANT_FORMS[subject]
        else:
            stem = _stem_for_word(verb)
            if not stem:
                return None
            head = _conjugate_subject_verb(subject, stem)

        tail = " ".join(words[3:])
        out = f"{q} {head}".strip()
        if tail:
            out = f"{out} {tail}"
        return _append_trailing_punct(text, out)

    return None


def _rule_phrase_translate(text: str) -> str:
    phrase_map, max_len = _load_phrase_dictionary()
    raw_tokens = text.split()
    token_infos = [_split_token_parts(token) for token in raw_tokens]

    output_tokens = []
    i = 0
    while i < len(token_infos):
        matched = False
        max_window = min(max_len, len(token_infos) - i)

        for size in range(max_window, 0, -1):
            if not _span_is_phrase_compatible(token_infos, i, size):
                continue

            norms = [token_infos[j]["norm"] for j in range(i, i + size)]
            candidate_phrase = " ".join(norms)
            key = _normalize_phrase(candidate_phrase)
            translation = phrase_map.get(key)
            if not translation:
                continue

            decorated = _decorate_translation(
                translation=translation,
                prefix=token_infos[i]["prefix"],
                suffix=token_infos[i + size - 1]["suffix"],
            )
            output_tokens.append(decorated)
            i += size
            matched = True
            break

        if not matched:
            output_tokens.append(token_infos[i]["token"])
            i += 1

    return " ".join(output_tokens)


def _mock_translate(text: str) -> str:
    grammar_guess = _grammar_sentence_guess(text)
    if grammar_guess:
        return grammar_guess
    return _rule_phrase_translate(text)


def _model_translate(text: str) -> str:
    tokenizer, model, device = _load_model()
    normalized = normalize_english_for_darija(text)
    input_text = f"{TASK_PREFIX}{normalized}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=72,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def translate_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    force_mock = os.getenv("USE_MOCK_TRANSLATOR", "0") == "1"
    mode = TRANSLATOR_MODE

    if force_mock or mode == "rule":
        return _mock_translate(text)

    if not _model_ready(MODEL_PATH):
        return _mock_translate(text)

    try:
        model_output = _model_translate(text)
    except Exception:
        return _mock_translate(text)

    if mode == "model":
        return model_output or _mock_translate(text)

    # Hybrid mode: if model output is empty or echoes input, use rule-based guess.
    normalized_input = _normalize_phrase(text)
    normalized_model = _normalize_phrase(model_output)
    if not normalized_model or normalized_model == normalized_input:
        return _mock_translate(text)

    return model_output


def translate_with_source(text: str) -> dict[str, str | bool]:
    cleaned = text.strip()
    if not cleaned:
        return {"translation": "", "source": "none", "is_exact_dataset_match": False}

    exact = get_dataset_translation(cleaned)
    if exact:
        return {
            "translation": exact,
            "source": "dataset_exact",
            "is_exact_dataset_match": True,
        }

    return {
        "translation": translate_text(cleaned),
        "source": "ai_guess",
        "is_exact_dataset_match": False,
    }
