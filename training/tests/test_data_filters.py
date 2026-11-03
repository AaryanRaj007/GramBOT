"""The filters in prepare_data.py that decide which pairs reach training."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_data import close, good_target, tidy, usable  # noqa: E402


class Close(unittest.TestCase):
    def test_keeps_a_small_correction(self):
        self.assertTrue(close("She go to school every day.", "She goes to school every day."))

    def test_drops_a_rewrite(self):
        self.assertFalse(close("The cat sat on the mat today.", "A dog was sleeping in the garden."))

    def test_drops_a_big_length_change(self):
        self.assertFalse(close("I like it.", "I like it a lot more than anything else I have seen."))


class Tidy(unittest.TestCase):
    def test_straightens_quotes(self):
        self.assertEqual(tidy("  “It’s fine,” she said. "), '"It\'s fine," she said.')


class Usable(unittest.TestCase):
    def test_plain_sentence(self):
        self.assertTrue(usable("We went to the market yesterday."))

    def test_rejects_quotes_lowercase_start_and_missing_full_stop(self):
        self.assertFalse(usable('He said "hi" to me.'))
        self.assertFalse(usable("we went home."))
        self.assertFalse(usable("We went home"))

    def test_rejects_too_short_and_non_ascii(self):
        self.assertFalse(usable("Go now."))
        self.assertFalse(usable("We ate crème brûlée."))


class GoodTarget(unittest.TestCase):
    VOCAB = {w: 100 for w in "she goes to school every day the".split()}

    def test_common_words_pass(self):
        self.assertTrue(good_target("She goes to school every day.", self.VOCAB))

    def test_rare_word_fails(self):
        self.assertFalse(good_target("She goes to skool every day.", self.VOCAB))

    def test_web_junk_fails(self):
        self.assertFalse(good_target("She goes to http school every day.", self.VOCAB))
        self.assertFalse(good_target("She does n't go to school.", self.VOCAB))


if __name__ == "__main__":
    unittest.main()
