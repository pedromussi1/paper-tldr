"""Tests for the deterministic, model-free pieces of src/baselines.py.

The actual generate() and load-model paths are not covered here — they need
a GPU and a multi-GB model download. Those are exercised end-to-end by the
training and eval scripts and verified via W&B run history.
"""
from __future__ import annotations

from src.baselines import (
    _build_prompt_messages,
    extractive_first_sentence,
    strip_llm_preamble,
)


class TestExtractiveFirstSentence:
    def test_simple_two_sentences(self):
        assert extractive_first_sentence("First sentence. Second sentence.") == "First sentence."

    def test_no_sentence_boundary_returns_full_input(self):
        text = "just one chunk no period"
        assert extractive_first_sentence(text) == text

    def test_strips_leading_trailing_whitespace(self):
        assert extractive_first_sentence("  First sentence. Second.") == "First sentence."

    def test_handles_question_mark(self):
        assert extractive_first_sentence("What is this? Then more.") == "What is this?"

    def test_handles_exclamation(self):
        assert extractive_first_sentence("Wow! Then more.") == "Wow!"

    def test_does_not_split_on_lowercase_following(self):
        # Splitting requires the next sentence to start with a capital — guards
        # against spurious splits on abbreviations like "e.g. foo".
        text = "Trained on data e.g. images. Then evaluated."
        assert extractive_first_sentence(text) == "Trained on data e.g. images."


class TestStripLlmPreamble:
    def test_heres_tldr_preamble(self):
        text = "Here's the TL;DR:\nActual summary content."
        assert strip_llm_preamble(text) == "Actual summary content."

    def test_heres_with_modifier(self):
        text = "Here's a rewritten TL;DR:\nThe core finding."
        assert strip_llm_preamble(text) == "The core finding."

    def test_short_label_line(self):
        text = "Plain-language summary of the paper:\nThis paper proposes X."
        assert strip_llm_preamble(text) == "This paper proposes X."

    def test_does_not_strip_long_first_line_with_colon(self):
        # The label-line regex is bounded to <=100 chars to avoid eating real content.
        long_first_line = "x" * 120 + ":"
        text = f"{long_first_line}\nbody"
        assert strip_llm_preamble(text).startswith("x")

    def test_strips_straight_quotes(self):
        assert strip_llm_preamble('"This is the summary."') == "This is the summary."

    def test_strips_curly_quotes(self):
        assert strip_llm_preamble("“This is the summary.”") == "This is the summary."

    def test_strips_single_asterisk_italic(self):
        assert strip_llm_preamble("*This is the summary.*") == "This is the summary."

    def test_does_not_strip_double_asterisk_bold(self):
        assert strip_llm_preamble("**This is the summary.**") == "**This is the summary.**"

    def test_clean_input_unchanged(self):
        text = "We propose a new method for X."
        assert strip_llm_preamble(text) == text

    def test_combined_preamble_and_quotes(self):
        text = "Here's the TL;DR:\n\"This is the summary.\""
        assert strip_llm_preamble(text) == "This is the summary."


def test_build_prompt_messages_structure():
    abstract = "Some abstract text describing a method."
    msgs = _build_prompt_messages(abstract)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert abstract in msgs[1]["content"]
    # Make sure the user prompt explicitly asks for the TL;DR
    assert "TL;DR" in msgs[1]["content"]
