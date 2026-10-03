from lips.wer import normalize, score


def test_normalize_drops_case_and_punctuation_but_keeps_apostrophes():
    assert normalize("  Don’t STOP, now!  ") == "don't stop now"


def test_perfect_hypothesis_scores_zero():
    report = score({"a": "the cat sat"}, {"a": "The cat sat."})
    assert report.wer == 0 and report.cer == 0


def test_word_errors_are_counted_per_clip():
    refs = {"a": "the cat sat on the mat", "b": "hello world"}
    hyps = {"a": "the cat sat on the mad", "b": "hello world"}
    report = score(refs, hyps)
    assert report.wer == 1 / 8
    assert report.worst[0].clip_id == "a" and report.worst[0].errors == 1


def test_missing_hypothesis_counts_as_all_words_deleted():
    report = score({"a": "one two", "b": "three"}, {"a": "one two"})
    assert report.missing == 1
    assert report.wer == 1 / 3


def test_insertions_count_as_errors():
    report = score({"a": "hi"}, {"a": "hi there you"})
    assert report.wer == 2.0
    assert report.worst[0].errors == 2
