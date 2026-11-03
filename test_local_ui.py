"""Checks for bulk upload parsing and predictions."""
import unittest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from engine import ROOT, load_model
from local_ui import JOBS, JOB_LOCK, model_history, parse_rows, review_training, run_training, score_batch, selected_rows, start_training, training_options
from training import evaluate, metrics


class BulkTests(unittest.TestCase):
    def test_txt_keeps_lines_duplicates_and_unsure(self):
        result = score_batch({'name': 'list.txt', 'content': '\ufeffmukunda@gmail.com\n\ninvalid\nmukunda@gmail.com\n'}, load_model())
        self.assertEqual([r['line'] for r in result], [1, 3, 4])
        self.assertEqual([r['decision'] for r in result], ['looks_ordinary', 'unsure', 'looks_ordinary'])
        self.assertEqual(result[1]['reason'], 'invalid_address_shape')

    def test_csv_column_selection_and_source_lines(self):
        rows = parse_rows({'name': 'list.csv', 'content': 'name,address\nMukunda,mukunda@gmail.com\nNobody,\nTest,qzxv8801@gmail.com\n', 'column': 'address'})
        self.assertEqual(rows, [(2, 'mukunda@gmail.com'), (4, 'qzxv8801@gmail.com')])
        with self.assertRaisesRegex(ValueError, 'column'):
            parse_rows({'name': 'list.csv', 'content': 'name,address\nx,a@example.com\n'})

    def test_empty_and_too_large(self):
        model = load_model()
        with self.assertRaisesRegex(ValueError, 'does not contain'):
            score_batch({'name': 'list.txt', 'content': '\n'}, model)
        with self.assertRaisesRegex(ValueError, '10,000'):
            score_batch({'name': 'list.txt', 'content': 'a@example.com\n' * 10001}, model)


class ReviewTests(unittest.TestCase):
    def test_conflicts_duplicates_exclusions_and_original_lines(self):
        data = {'ordinary': {'name': 'ordinary.csv', 'content': 'email\nmukunda@example.com\nMUKUNDA@mail.example.co.in\ninvalid\n'},
                'flagged': {'name': 'flagged.txt', 'content': 'mukunda@example.net\ninvalid\n'}}
        review = review_training(data)
        self.assertEqual(review['conflict_groups'], 2)
        self.assertEqual(review['counts']['ordinary']['duplicate_lines'], 1)
        self.assertEqual(review['counts']['ordinary']['unsupported_rows'], 1)
        self.assertEqual(len(review['issues']), 5)
        data['exclude'] = {'flagged': [1, 2]}
        clean = review_training(data)
        self.assertEqual(clean['conflict_groups'], 0)
        self.assertEqual(selected_rows(data['flagged'], [1, 2]), [])
        self.assertEqual(clean['counts']['ordinary']['eligible_unique'], 1)

    def test_no_issues(self):
        data = {'ordinary': {'name': 'a.txt', 'content': 'mukunda@gmail.com\n'},
                'flagged': {'name': 'b.txt', 'content': 'qzxv8801@gmail.com\n'}}
        self.assertEqual(review_training(data)['issues'], [])


class HistoryTests(unittest.TestCase):
    def test_valid_model_missing_or_invalid_report_still_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('missing', 'invalid'):
                folder = root / name
                folder.mkdir()
                (folder / 'model.json').write_text((ROOT / 'demo/model.json').read_text())
            (root / 'invalid/report.json').write_text('{broken')
            with patch('local_ui.RUNS', root):
                entries = {m['id']: m for m in model_history()}
            self.assertIn('demo', entries)
            self.assertIsNotNone(entries['demo']['report'])
            self.assertIsNone(entries['missing']['report'])
            self.assertIsNone(entries['invalid']['report'])
            self.assertTrue(entries['missing']['modified'].endswith('+00:00'))


class ThresholdTests(unittest.TestCase):
    def test_score_groups_reconstruct_metrics_with_abstentions_and_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            flagged, ordinary = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
            flagged.write_text('qzxv8801@gmail.com\nbroken@@example.com\n')
            ordinary.write_text('mukunda@gmail.com\n')
            result = evaluate(flagged, ordinary, load_model(), include_scores=True)
            groups = result['score_groups']
            self.assertEqual(sum(row[1] + row[2] for row in groups), 2)
            self.assertEqual(result['results']['abstained'], 1)
            for cutoff in (0, 1, groups[0][0], groups[-1][0]):
                fp = sum(ordinary_count for score, ordinary_count, _ in groups if score >= cutoff)
                tp = sum(flagged_count for score, _, flagged_count in groups if score >= cutoff)
                pairs = [(0, score) for score, n, _ in groups for _ in range(n)] + [(1, score) for score, _, n in groups for _ in range(n)]
                expected = metrics(pairs, cutoff, (0, 1))
                self.assertEqual((tp, fp, 2 - tp), (expected['true_flags'], expected['false_flags'], expected['missed_flags']))
            self.assertNotIn('score_groups', evaluate(flagged, ordinary, load_model()))

    def test_breakdowns_reconcile_and_keep_abstentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            flagged, ordinary = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
            flagged.write_text('qzxv8801@gmail.com\n||||||||@gmail.com\nbroken@@example.com\n')
            ordinary.write_text('mukunda@gmail.com\nanna@yahoo.com\n')
            result = evaluate(flagged, ordinary, load_model(), include_breakdowns=True)
            groups = result['breakdowns']
            self.assertEqual(groups['excluded_before_grouping'], 1)
            self.assertEqual({g['label'] for g in groups['domain_labels']}, {'gmail', 'yahoo'})
            for dimension in ('domain_labels', 'username_coverage'):
                rows = groups[dimension]
                self.assertEqual(sum(g['flagged'] for g in rows), 2)
                self.assertEqual(sum(g['ordinary'] for g in rows), 2)
                self.assertEqual(sum(g['true_flags'] for g in rows), result['results']['true_flags'])
                self.assertEqual(sum(g['missed_flags'] for g in rows) + 1, result['results']['missed_flags'])
                self.assertEqual(sum(g['abstained'] for g in rows) + 1, result['results']['abstained'])
            self.assertNotIn('breakdowns', evaluate(flagged, ordinary, load_model()))


class TrainingControlTests(unittest.TestCase):
    def test_options_and_full_training_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {'flagged': {'name': 'flagged.txt', 'content': (ROOT / 'examples/flagged.txt').read_text()},
                    'ordinary': {'name': 'ordinary.txt', 'content': (ROOT / 'examples/ordinary.txt').read_text()},
                    'name': 'Settings check', 'options': {'min_df': 1, 'min_coverage': .3, 'max_fpr': .1}}
            stages = []
            with patch('local_ui.RUNS', Path(tmp)):
                result = run_training(data, stages.append)
                model = load_model(Path(tmp) / result['id'])
            self.assertEqual(result['report']['min_df'], 1)
            self.assertEqual(result['report']['validation_max_ordinary_false_flag_rate'], .1)
            self.assertEqual(model['min_coverage'], .3)
            self.assertIn('Fitting vocabulary and model weights', stages)
            self.assertEqual(stages[-1], 'Saving model and reports')
        for options in ({'min_df': 0}, {'min_df': 1.5}, {'max_fpr': float('nan')}, {'min_coverage': 1.1}):
            with self.assertRaises(ValueError):
                training_options({'options': options})

    def test_only_one_background_job_at_a_time(self):
        entered, finish = threading.Event(), threading.Event()
        def slow_job(data, progress):
            progress('Fitting')
            entered.set()
            self.assertTrue(finish.wait(5))
            return {'id': 'test'}
        with patch('local_ui.run_training', slow_job), patch.dict(JOBS, {}, clear=True):
            job = start_training({})
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(ValueError, 'already in progress'):
                    start_training({})
            finally:
                finish.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with JOB_LOCK:
                    state = dict(JOBS[job])
                if state['status'] != 'running':
                    break
                time.sleep(.01)
            self.assertEqual(state['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
