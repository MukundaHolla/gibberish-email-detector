"""Regression checks for the standalone product; fitting tests need requirements.txt."""

import json
from pathlib import Path
import tempfile
import unittest

from detector import batch
from engine import ROOT, grams, load_model, predict, preprocess
from training import choose_threshold, evaluate, read_lists, split_rows


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_normalization(self):
        for email, expected in [("  ANU+SHOP@mail.host.co.in ", ("anu+shop", "host")),
                                ("ravi@a.b.ck", None), ("ravi@sub.www.ck", ("ravi","www")),
                                ("ravi@sub.tenant.github.io", ("ravi","tenant"))]:
            parts, _ = preprocess(email)
            self.assertEqual((parts.username, parts.domain) if parts else None, expected)
        self.assertEqual(preprocess("hello@private.icloud.com")[1], "recognized_private_relay")
        self.assertEqual(preprocess("bad@@gmail.com")[1], "invalid_address_shape")
        self.assertEqual(preprocess("ನಮಸ್ಕಾರ@gmail.com")[1], "unsupported_address_form")

    def test_contributions_and_coverage(self):
        r = predict("ravi@gmail.com", self.model, explain=True)
        self.assertEqual(list(grams("ravi")), ["ra","av","vi","rav","avi","ravi"])
        self.assertEqual(r["coverage"]["user"]["known"], 3)
        self.assertEqual(r["coverage"]["user"]["total"], 6)
        self.assertAlmostEqual(r["math"]["intercept"] + sum(t["points"] for t in r["contributions"]),
                               r["math"]["margin"], places=12)
        self.assertAlmostEqual(r["score"], .385235, places=5)

    def test_threshold_does_not_change_score(self):
        r = predict("qzxv8801@gmail.com", self.model)
        self.assertEqual(predict("qzxv8801@gmail.com", self.model, threshold=r["score"])["decision"], "looks_gibberish")
        higher = predict("qzxv8801@gmail.com", self.model, threshold=.80)
        self.assertEqual(higher["score"], r["score"])
        self.assertEqual(higher["decision"], "looks_ordinary")
        uncertain = predict("||||||||@gmail.com", self.model, threshold=0)
        self.assertIsNone(uncertain["score"])
        self.assertEqual(uncertain["decision"], "unsure")

    def test_duplicate_and_class_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            f, o = Path(tmp)/"flagged.txt", Path(tmp)/"ordinary.txt"
            f.write_text("qzxv88@gmail.com\n")
            o.write_text("RAVI@mail.example.co.in\nravi@example.com\n")
            rows, stats = read_lists(f, o)
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats["ordinary"]["duplicate_lines"], 1)
            f.write_text("ravi@example.net\n")
            with self.assertRaisesRegex(ValueError, "contradictory labels"):
                read_lists(f, o)

    def test_equal_scores_and_no_candidate(self):
        self.assertEqual(choose_threshold([(1,.8),(0,.7),(1,.7),(0,None)], 0), .8)
        self.assertIsNone(choose_threshold([(0,.8),(1,.8)], 0))
        self.assertEqual(predict("ravi@gmail.com", dict(self.model, threshold=None))["decision"], "score_only")

    def test_holdout_and_batch_include_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/"flagged.txt").write_text("qzxv8801@gmail.com\nbroken@@mail.com\n")
            (d/"ordinary.txt").write_text("ravi@gmail.com\n")
            m = evaluate(d/"flagged.txt", d/"ordinary.txt", self.model)["results"]
            self.assertEqual((m["true_flags"],m["missed_flags"],m["abstained"]), (1,1,1))
            batch(d/"flagged.txt", d/"out.jsonl", self.model, None)
            out = [json.loads(line) for line in (d/"out.jsonl").read_text().splitlines()]
            self.assertEqual([r["line"] for r in out], [1,2])
            self.assertEqual(out[1]["decision"], "unsure")
            with self.assertRaises(ValueError):
                batch(d/"flagged.txt", d/"out.jsonl", self.model, None)

    def test_split_separates_groups(self):
        rows, _ = read_lists(ROOT/"examples/flagged.txt", ROOT/"examples/ordinary.txt")
        split = split_rows(rows)
        groups = [{p.username.partition("+")[0] or p.username for p, _ in part} for part in split.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


if __name__ == "__main__":
    unittest.main()
