"""Edit-level scoring in evaluate.py. Pure Python: no model, no torch."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import Score, edits, norm  # noqa: E402


class Norm(unittest.TestCase):
    def test_curly_quotes_and_spaces(self):
        self.assertEqual(norm("It’s  “fine”"), 'It\'s "fine"')


class Edits(unittest.TestCase):
    def test_identical_has_no_edits(self):
        self.assertEqual(edits("She goes home.", "She goes home."), set())

    def test_single_replacement(self):
        self.assertEqual(edits("She go home.", "She goes home."), {(1, 2, "goes")})

    def test_deletion_has_empty_target(self):
        self.assertEqual(edits("She went to the home.", "She went home."), {(2, 4, "")})


class Scoring(unittest.TestCase):
    def test_perfect_fix(self):
        s = Score()
        s.add("She go home.", "She goes home.", ["She goes home."])
        self.assertEqual(s.prf(), (1.0, 1.0, 1.0))

    def test_no_change_is_all_recall_loss(self):
        s = Score()
        s.add("She go home.", "She go home.", ["She goes home."])
        p, r, f = s.prf()
        self.assertEqual((s.tp, s.fp, s.fn), (0, 0, 1))
        self.assertEqual(r, 0.0)

    def test_wrong_fix_is_false_positive_and_miss(self):
        s = Score()
        s.add("She go home.", "She went home.", ["She goes home."])
        self.assertEqual((s.tp, s.fp, s.fn), (0, 1, 1))

    def test_closest_reference_wins(self):
        s = Score()
        s.add("She go home.", "She went home.", ["She goes home.", "She went home."])
        self.assertEqual((s.tp, s.fp, s.fn), (1, 0, 0))

    def test_f05_weighs_precision_more(self):
        precise, eager = Score(), Score()
        precise.tp, precise.fp, precise.fn = 1, 0, 1   # P 1.0, R 0.5
        eager.tp, eager.fp, eager.fn = 1, 1, 0         # P 0.5, R 1.0
        self.assertGreater(precise.prf()[2], eager.prf()[2])


if __name__ == "__main__":
    unittest.main()
