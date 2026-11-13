"""Checks for the browser worker adapter, also runnable without Pyodide."""
import json
import unittest
from pathlib import Path

from engine import ROOT
from web import browser_backend as browser


class BrowserBackendTests(unittest.TestCase):
    def test_numeric_signals_and_optional_names_survive_training_and_import(self):
        def csv_data(path, name):
            emails = (ROOT / path).read_text().splitlines()
            return {'name': name + '.csv', 'content': 'email,full_name\n' + '\n'.join(
                f'{email},{email.split("@")[0] if name == "ordinary" else "Different Person"}' for email in emails) + '\n'}
        data = {'ordinary': csv_data('examples/ordinary.txt', 'ordinary'),
                'flagged': csv_data('examples/flagged.txt', 'flagged'),
                'options': {'min_coverage': 0}}
        trained = browser.handle('train', data, lambda _: None)
        artifacts = browser.handle('artifacts', {'model': trained['id']}, lambda _: None)
        model = artifacts['model.json']
        self.assertEqual(model['format'], 'whole-email-char-tfidf-signals-logistic-v2')
        self.assertGreater(artifacts['report.json']['training_names_available'], 0)
        self.assertIn('name_similarity', model['signal_weights'])
        imported = browser.handle('import', {'content': json.dumps(model)}, lambda _: None)
        answer = browser.handle('score', {'model': imported['id'], 'email': 'Mukunda123@gmail.com',
                                           'full_name': 'Mukunda'}, lambda _: None)
        self.assertTrue(answer['signals_used'])
        self.assertEqual(answer['signals']['capital_letters'], 1)
        self.assertEqual(answer['signals']['numbers_in_a_row'], 3)
        self.assertIsNotNone(answer['signals']['name_similarity'])
        self.assertAlmostEqual(answer['math']['intercept'] + answer['math']['user_sum'] +
                               answer['math']['domain_sum'] + answer['math']['signal_sum'], answer['math']['margin'])
        no_name = browser.handle('score', {'model': imported['id'], 'email': 'Mukunda123@gmail.com'}, lambda _: None)
        self.assertIsNone(no_name['signals']['name_similarity'])
        self.assertNotEqual(answer['score'], no_name['score'])
        evaluation = browser.handle('evaluate', {**data, 'model': imported['id']}, lambda _: None)
        self.assertGreater(evaluation['results']['ordinary'], 0)
        self.assertGreater(evaluation['results']['flagged'], 0)

    def test_signal_counts_and_invalid_colon(self):
        from engine import address_signals, preprocess
        parts, _ = preprocess('Abc-12@gmail.com', 'Abc')
        values = address_signals(parts)
        self.assertEqual((values['length'], values['alphanumeric'], values['non_alphanumeric']), (6, 5, 1))
        self.assertEqual((values['vowels'], values['consonants'], values['numbers']), (1, 2, 2))
        self.assertEqual((values['hyphens'], values['numbers_in_a_row'], values['capital_letters']), (1, 2, 1))
        self.assertAlmostEqual(values['vowel_ratio'], 1 / 6)
        self.assertEqual(preprocess('abc:12@gmail.com')[1], 'invalid_address_shape')

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
