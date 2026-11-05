import json
import tempfile
import unittest
from pathlib import Path
from domain_rules import parse_domain_file, match_domain
from engine import load_model, predict
from local_ui import score_batch
from training import evaluate
from domain_ml import canonical_domain, fit_domain_model


class DomainRuleTests(unittest.TestCase):
    def test_txt_csv_normalize_duplicates_and_errors(self):
        result = parse_domain_file({'name': 'domains.txt', 'content': '\ufeffExample.COM\nexample.com.\nother.net\n'})
        self.assertEqual(result, {'domains': ['example.com', 'other.net'], 'duplicates': 1})
        result = parse_domain_file({'name': 'domains.csv', 'content': 'name,domain\none,EXAMPLE.com\ntwo,other.net\n'})
        self.assertEqual(result['domains'], ['example.com', 'other.net'])
        self.assertEqual(parse_domain_file({'name':'domains.csv','content':'example.com\nother.net\n'})['domains'], ['example.com', 'other.net'])
        with self.assertRaisesRegex(ValueError, 'No changes saved'):
            parse_domain_file({'name': 'bad.txt', 'content': 'example.com\nhttps://bad.example\n'})

    def test_large_list_and_subdomain_matching_without_suffix_spoofing(self):
        content = ''.join(f'domain{i}.example.com\n' for i in range(22000))
        self.assertEqual(len(parse_domain_file({'name': 'large.txt', 'content': content})['domains']), 22000)
        rules = {'example.com'}
        self.assertEqual(match_domain('Name@EXAMPLE.COM', rules), 'example.com')
        self.assertIsNone(match_domain('x@sub.example.com', rules))
        self.assertEqual(match_domain('x@sub.example.com', rules, True), 'example.com')
        for address in ('x@notexample.com', 'x@example.com.evil.net', 'bad@@example.com'):
            self.assertIsNone(match_domain(address, rules, True))

    def test_serialized_model_flags_domain_and_batch(self):
        model = json.loads((Path(__file__).parent / 'demo/model.json').read_text())
        model['flagged_domains'] = ['example.com']
        model['include_subdomains'] = True
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'model.json'
            file.write_text(json.dumps(model))
            loaded = load_model(file)
            self.assertEqual(json.loads(json.dumps(loaded))['flagged_domains'], ['example.com'])
            direct = predict('x@sub.example.com', loaded, explain=True)
            self.assertEqual(direct['decision'], 'flagged_domain')
            self.assertIsNone(direct['score'])
            self.assertEqual(direct['matched_domain'], 'example.com')
            self.assertEqual(predict('mukunda@gmail.com', loaded)['decision'], 'looks_ordinary')
            rows = score_batch({'name':'list.txt','content':'x@example.com\nmukunda@gmail.com\n'}, loaded)
            self.assertEqual([r['decision'] for r in rows], ['flagged_domain', 'looks_ordinary'])

    def test_evaluation_counts_domain_flags_without_fabricating_scores(self):
        model = load_model()
        model['flagged_domains'] = ['example.com']
        model.domain_lookup = frozenset(model['flagged_domains'])
        with tempfile.TemporaryDirectory() as tmp:
            flagged, ordinary = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
            flagged.write_text('x@sub.example.com\nqzxv8801@gmail.com\n')
            ordinary.write_text('good@example.com\nmukunda@gmail.com\n')
            report = evaluate(flagged, ordinary, model, include_scores=True, include_breakdowns=True)
            self.assertEqual(report['domain_flags'], {'ordinary': 1, 'flagged': 1})
            self.assertEqual(report['results']['true_flags'], 2)
            self.assertEqual(report['results']['false_flags'], 1)
            self.assertEqual(report['results']['scored'], 2)
            self.assertEqual(report['results']['coverage'], .5)
            self.assertEqual(report['results']['precision'], 2/3)
            self.assertEqual(sum(o+f for _,o,f in report['score_groups']), 2)

    def test_domain_model_canonicalizes_and_rejects_cross_class_aliases(self):
        self.assertEqual(canonical_domain('mail.example.co.in'), 'example.co.in')
        with self.assertRaisesRegex(ValueError, 'both flagged and ordinary'):
            fit_domain_model(['a.example.com'], ['example.com'])

    def test_both_model_decisions_and_subdomains(self):
        model = load_model()
        model['domain_model'] = {'format':'registrable-domain-char-tfidf-logistic-v1',
                                 'terms':{'ex':[1.0, 4.0]}, 'intercept':-1.0,
                                 'min_coverage':0.0, 'threshold':0.5}
        flagged = predict('x@sub.example.com', model)
        self.assertEqual(flagged['email_decision'], 'unsure')
        self.assertEqual(flagged['domain_result']['domain'], 'example.com')
        self.assertEqual(flagged['domain_result']['decision'], 'looks_gibberish')
        self.assertEqual(flagged['decision'], 'looks_gibberish')
        unsure = predict('mukunda@gmail.com', model)
        self.assertEqual(unsure['email_decision'], 'looks_ordinary')
        self.assertEqual(unsure['domain_result']['decision'], 'unsure')
        self.assertEqual(unsure['decision'], 'unsure')
        flagged_by_email = predict('qzxv8801@gmail.com', model)
        self.assertEqual(flagged_by_email['decision'], 'looks_gibberish')
        self.assertEqual(flagged_by_email['domain_result']['decision'], 'unsure')
        model['domain_model']['terms']['gm'] = [1.0, -4.0]
        ordinary = predict('mukunda@gmail.com', model)
        self.assertEqual(ordinary['domain_result']['decision'], 'looks_ordinary')
        self.assertEqual(ordinary['decision'], 'looks_ordinary')

    def test_combined_evaluation_counts_domain_and_email_alerts(self):
        model = load_model()
        model['domain_model'] = {'format':'registrable-domain-char-tfidf-logistic-v1',
                                 'terms':{'ex':[1.0, 4.0]}, 'intercept':-1.0,
                                 'min_coverage':0.0, 'threshold':0.5}
        with tempfile.TemporaryDirectory() as tmp:
            flagged, ordinary = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
            flagged.write_text('x@sub.example.com\nqzxv8801@gmail.com\n')
            ordinary.write_text('good@example.com\nmukunda@gmail.com\n')
            report = evaluate(flagged, ordinary, model, include_scores=True, include_breakdowns=True)
            self.assertEqual(report['domain_flag_kind'], 'model')
            self.assertEqual(report['domain_flags'], {'ordinary': 1, 'flagged': 1})
            self.assertEqual(report['results']['true_flags'], 2)
            self.assertEqual(report['results']['false_flags'], 1)
            self.assertEqual(report['results']['precision'], 2/3)
            self.assertEqual(report['decision_coverage'], .75)


if __name__ == '__main__':
    unittest.main()
