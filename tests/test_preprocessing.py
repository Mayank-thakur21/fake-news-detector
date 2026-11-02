from ml.preprocessing import clean_text, prepare, strip_source_leakage, word_count


def test_clean_text_lowercases_and_strips_noise():
    raw = "BREAKING: Visit https://x.com/a?b=1 <b>now</b> [video] 2024 covid19 news!!!"
    assert clean_text(raw) == "breaking visit now news"


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""  # type: ignore[arg-type]
    assert clean_text(123) == ""  # type: ignore[arg-type]


def test_clean_text_collapses_whitespace_and_underscores():
    assert clean_text("a\n\n b\t_c_") == "a b c"


def test_strip_source_leakage_removes_reuters_dateline():
    text = "WASHINGTON (Reuters) - The Senate voted on Tuesday. Reuters reported more."
    out = strip_source_leakage(text)
    assert "reuters" not in out.lower()
    assert out.startswith("The Senate voted")


def test_prepare_optionally_keeps_leakage():
    text = "LONDON (Reuters) - Markets rose."
    assert "reuters" in prepare(text, strip_leakage=False)
    assert "reuters" not in prepare(text, strip_leakage=True)


def test_word_count():
    assert word_count("Hello, world! 42") == 2
    assert word_count("") == 0
