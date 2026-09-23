import csv
import os
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"
sys.path.append(str(APP_DIR))

os.environ["USE_MOCK_TRANSLATOR"] = "1"

from translator import translate_text  # noqa: E402

WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _normalize_key(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.lower()))


def _dataset_lookup(english_text: str) -> str | None:
    target = _normalize_key(english_text)
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = (row.get("en") or "").strip()
            darija = (row.get("darija") or "").strip()
            if _normalize_key(en) == target and darija:
                return darija
    return None


def test_translate_text_returns_known_mock_value():
    expected = _dataset_lookup("hello")
    assert expected is not None
    assert translate_text("hello") == expected


def test_translate_text_handles_empty_input():
    assert translate_text("   ") == ""


def test_translate_text_recognizes_multiple_known_parts():
    hello = _dataset_lookup("hello")
    thank_you = _dataset_lookup("thank you")
    assert hello is not None and thank_you is not None
    assert translate_text("hello thank you") == f"{hello} {thank_you}"


def test_translate_text_keeps_unknown_words():
    unknown = "xqzvword"
    result = translate_text(f"hello {unknown}")
    assert result.split()[-1] == unknown


def test_translate_text_matches_known_phrase_inside_sentence():
    unknown = "xqzvword"
    result = translate_text(f"How are you {unknown}?")
    phrase = _dataset_lookup("how are you?")
    assert phrase is not None
    assert result.startswith(phrase)
    assert unknown in result


def test_translate_text_i_want_to_pattern():
    assert translate_text("I want to go") == "bghit nmshi"


def test_translate_text_i_want_you_to_pattern():
    assert translate_text("I want you to help") == "bghitek t3awn"


def test_translate_text_i_would_like_you_to_pattern():
    assert translate_text("I would like you to help") == "bghitek t3awn"


def test_translate_text_id_like_you_to_pattern():
    assert translate_text("I'd like you to go") == "bghitek tmshi"


def test_translate_text_removes_do_in_question():
    assert translate_text("What do I want?") == "chno bghit?"


def test_translate_text_present_conjugation_by_subject():
    assert translate_text("I eat bread") == "kanakul khobz"
    assert translate_text("She writes a book") == "katktb ktab"
    assert translate_text("They drink water") == "kayshrbu lma"


def test_translate_text_future_conjugation():
    assert translate_text("I will go home tomorrow") == "ghadi nmshi dar ghedda"
    assert translate_text("We are going to eat couscous") == "ghadi naklu seksu"


def test_translate_text_past_conjugation():
    assert translate_text("I went home yesterday") == "mshit dar lbare7"
    assert translate_text("She ate bread") == "klat khobz"


def test_translate_text_negates_present_past_and_future():
    assert translate_text("I do not eat bread") == "ma-kanakulsh khobz"
    assert translate_text("I did not go home") == "ma-mshitsh dar"
    assert translate_text("I will not drink coffee") == "ma-ghadi-nshrbsh qahwa"


def test_translate_text_question_patterns():
    assert translate_text("Where do you go?") == "fin katmshi?"
    assert translate_text("Did she eat bread?") == "wach klat khobz?"
    assert translate_text("Can you help?") == "wach tqder t3awn?"


def test_translate_text_have_need_and_possession():
    assert translate_text("I have money") == "3endi flous"
    assert translate_text("I need to write a letter") == "khassni nktb brriya"
    assert translate_text("my book") == "ktab dyali"


def test_translate_text_be_and_adjective_agreement():
    assert translate_text("I am tired") == "ana 3yan"
    assert translate_text("She is tired") == "hiya 3yana"
    assert translate_text("She was tired") == "kanet 3yana"
    assert translate_text("There is no problem") == "ma-kaynsh mushkil"


def test_translate_text_imperatives():
    assert translate_text("Go home please") == "sir dar afak"
    assert translate_text("Do not go home") == "ma-tmshish dar"
