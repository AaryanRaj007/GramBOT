"""Contracts of the synthetic error generators in noise.py."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import noise  # noqa: E402

SAMPLES = ["She goes to school every day.", "I have seen that movie before.", "It's not your fault, and you know it.",
           "They were happy with the information we sent.", "He could have told me about the meeting."]


class ErrorFunctions(unittest.TestCase):
    def test_weights_match_errors(self):
        self.assertEqual(len(noise.ERRORS), len(noise.WEIGHTS))

    def test_each_returns_changed_text_or_none(self):
        rng = random.Random(0)
        for fn in noise.ERRORS:
            for s in SAMPLES:
                out = fn(s, rng)
                with self.subTest(fn=fn.__name__, s=s):
                    self.assertTrue(out is None or isinstance(out, str))

    def test_short_input_is_left_alone(self):
        rng = random.Random(0)
        self.assertIsNone(noise.duplicate_word("Hi there.", rng))
        self.assertIsNone(noise.word_order("Hi there friend.", rng))


class Corrupt(unittest.TestCase):
    def test_same_seed_same_output(self):
        a = [noise.corrupt(s, random.Random(7)) for s in SAMPLES]
        b = [noise.corrupt(s, random.Random(7)) for s in SAMPLES]
        self.assertEqual(a, b)

    def test_output_differs_and_has_no_double_spaces(self):
        rng = random.Random(3)
        for s in SAMPLES * 20:
            out = noise.corrupt(s, rng)
            if out is None:
                continue
            self.assertNotEqual(out, s)
            self.assertNotIn("  ", out)


class Pairs(unittest.TestCase):
    def test_run_on_target_is_the_original_sentences(self):
        src, tgt = noise.run_on(SAMPLES[:2], random.Random(1))
        self.assertEqual(tgt, " ".join(SAMPLES[:2]))
        self.assertNotEqual(src, tgt)

    def test_speech_target_is_quoted(self):
        rng = random.Random(2)
        for _ in range(30):
            _, tgt = noise.speech(rng)
            self.assertIn(', "', tgt)
            self.assertTrue(tgt.endswith('"'))

    def test_reported_speech_is_never_quoted(self):
        rng = random.Random(2)
        for _ in range(30):
            _, tgt = noise.reported(rng)
            self.assertNotIn('"', tgt)
            self.assertTrue(tgt.endswith("."))


if __name__ == "__main__":
    unittest.main()
