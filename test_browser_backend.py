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

    def test_explanation_counts_include_repeated_and_unknown_fragments(self):
        for email in ('mukunda@gmail.com', 'fdgafvfsd@gmail.com', 'aaaa@gmail.com'):
            result = browser.handle('score', {'model': 'demo', 'email': email}, lambda _: None)
            for field in ('user', 'domain'):
                self.assertEqual(len(result['fragments'][field]), result['coverage'][field]['total'])
                self.assertEqual(len(result['known_fragments'][field]), result['coverage'][field]['known'])
            self.assertEqual(result['min_coverage'], .2)

    def test_preview_and_evaluation(self):
        data = {'flagged': {'name': 'flagged.txt', 'content': 'qzxv8801@gmail.com\nbroken@@example.com\n'},
                'ordinary': {'name': 'ordinary.csv', 'content': 'email\nmukunda@gmail.com\n'}}
        review = browser.handle('preview', data, lambda _: None)
        self.assertEqual(review['counts']['flagged']['unsupported_rows'], 1)
        report = browser.handle('evaluate', {**data, 'model': 'demo'}, lambda _: None)
        self.assertEqual((report['results']['true_flags'], report['results']['missed_flags']), (1, 1))
        self.assertIn('score_groups', report)
        self.assertIn('breakdowns', report)

    def test_two_models_train_and_survive_export_import(self):
        flagged_domains = ''.join(f'bad{i}.example{i}.com\n' for i in range(20))
        ordinary_domains = ''.join(f'good{i}.sample{i}.org\n' for i in range(20))
        data = {'flagged': {'name': 'flagged.txt', 'content': (ROOT / 'examples/flagged.txt').read_text()},
                'ordinary': {'name': 'ordinary.txt', 'content': (ROOT / 'examples/ordinary.txt').read_text()},
                'domain_files': {'flagged': {'name': 'flagged.txt', 'content': flagged_domains},
                                 'ordinary': {'name': 'ordinary.txt', 'content': ordinary_domains}},
                'name': 'two model test'}
        trained = browser.handle('train', data, lambda _: None)
        artifacts = browser.handle('artifacts', {'model': trained['id']}, lambda _: None)
        model = artifacts['model.json']
        self.assertNotIn('flagged_domains', model)
        self.assertIn('domain_model', model)
        self.assertEqual(artifacts['report.json']['domain_model']['input_counts'], {'ordinary': 20, 'flagged': 20})
        self.assertNotIn('example0.com', json.dumps(model))
        imported = browser.handle('import', {'content': json.dumps(model)}, lambda _: None)
        answer = browser.handle('score', {'model': imported['id'], 'email': 'name@sub.example0.com'}, lambda _: None)
        self.assertEqual(answer['domain_result']['domain'], 'example0.com')
        self.assertIn(answer['decision'], ('looks_gibberish', 'looks_ordinary', 'unsure'))
        self.assertIn('email_decision', answer)
        for case in artifacts['golden.json']:
            actual = browser.handle('score', {'model': imported['id'], 'email': case['email']}, lambda _: None)
            self.assertEqual(actual['decision'], case['decision'])
            self.assertEqual(actual['reason'], case['reason'])
            if case['score'] is not None:
                self.assertAlmostEqual(actual['score'], case['score'], places=12)

    def test_domain_training_requires_both_files(self):
        with self.assertRaisesRegex(ValueError, 'both flagged and ordinary domain files'):
            browser.handle('train', {'domain_files': {'flagged': {'name': 'a.txt', 'content': 'example.com'}}}, lambda _: None)

    def test_large_domain_list_import_and_lookup(self):
        model = json.loads((ROOT / 'demo/model.json').read_text())
        model['flagged_domains'] = [f'domain{i}.example.com' for i in range(22000)]
        model['include_subdomains'] = True
        imported = browser.handle('import', {'content': json.dumps(model)}, lambda _: None)
        answer = browser.handle('score', {'model': imported['id'], 'email': 'name@sub.domain21999.example.com'}, lambda _: None)
        self.assertEqual(answer['decision'], 'flagged_domain')
        self.assertEqual(answer['matched_domain'], 'domain21999.example.com')


if __name__ == '__main__':
    unittest.main()
