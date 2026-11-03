"""Checks for the browser worker adapter, also runnable without Pyodide."""
import json
import unittest
from pathlib import Path

from engine import ROOT
from web import browser_backend as browser


class BrowserBackendTests(unittest.TestCase):
    def test_guide_detection_numbers_match_the_demo(self):
        result = browser.handle('score', {'model': 'demo', 'email': 'mukunda@gmail.com'}, lambda _: None)
        self.assertEqual((result['coverage']['user']['known'], result['coverage']['user']['total']), (5, 15))
        self.assertEqual((result['coverage']['domain']['known'], result['coverage']['domain']['total']), (9, 9))
        self.assertEqual(result['decision'], 'looks_ordinary')
        self.assertAlmostEqual(result['score'], .4126, places=4)
        self.assertAlmostEqual(result['threshold'], .4508, places=4)
        contribution = next(item for item in result['contributions'] if item['term'] == 'user:un')
        self.assertAlmostEqual(contribution['points'], -.1474, places=4)

    def test_demo_golden_and_model_import(self):
        browser.self_test()
        raw = (ROOT / 'demo/model.json').read_text()
        imported = browser.handle('import', {'content': raw}, lambda _: None)
        answer = browser.handle('score', {'model': imported['id'], 'email': 'mukunda@gmail.com'}, lambda _: None)
        self.assertAlmostEqual(answer['score'], .412573, places=5)
        self.assertEqual(answer['decision'], 'looks_ordinary')
        self.assertEqual(browser.handle('artifacts', {'model': imported['id']}, lambda _: None)['model.json']['format'], 'whole-email-char-tfidf-logistic-v1')
        with self.assertRaises(ValueError):
            browser.handle('import', {'content': json.dumps({'format': 'wrong'})}, lambda _: None)

    def test_preview_and_evaluation(self):
        data = {'flagged': {'name': 'flagged.txt', 'content': 'qzxv8801@gmail.com\nbroken@@example.com\n'},
                'ordinary': {'name': 'ordinary.csv', 'content': 'email\nmukunda@gmail.com\n'}}
        review = browser.handle('preview', data, lambda _: None)
        self.assertEqual(review['counts']['flagged']['unsupported_rows'], 1)
        report = browser.handle('evaluate', {**data, 'model': 'demo'}, lambda _: None)
        self.assertEqual((report['results']['true_flags'], report['results']['missed_flags']), (1, 1))
        self.assertIn('score_groups', report)
        self.assertIn('breakdowns', report)


if __name__ == '__main__':
    unittest.main()
