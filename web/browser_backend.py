"""Browser worker adapter. All inputs and model artifacts live in Pyodide memory."""
import json
import math
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from engine import ROOT, load_model, predict
from domain_rules import parse_domain_file
from local_ui import parse_file, review_training, score_batch, selected_rows, training_options
from training import evaluate, train

_models = {'demo': load_model(ROOT / 'demo')}
_artifacts = {'demo': {name: json.loads((ROOT / 'demo' / name).read_text()) for name in ('model.json', 'report.json', 'golden.json')}}
_metadata = {'demo': {'id': 'demo', 'name': 'Demo · invented examples', 'modified': datetime.now(timezone.utc).isoformat()}}


def self_test():
    for case in _artifacts['demo']['golden.json']:
        actual = predict(case['email'], _models['demo'])
        expected = case['score']
        if (actual['decision'] != case['decision'] or actual['reason'] != case['reason'] or
                (actual['score'] is None if expected is not None else actual['score'] is not None) or
                (expected is not None and not math.isclose(actual['score'], expected, rel_tol=0, abs_tol=1e-12))):
            raise ValueError('Browser model self-test failed; this browser cannot safely score the bundled model.')


def handle(route, data, progress):
    def model():
        key = data.get('model', 'demo')
        if key not in _models:
            raise ValueError('Model not found. Import it again.')
        return _models[key]

    if route == 'models':
        return [{'id': item['id'], 'name': item['name']} for item in _metadata.values()]
    if route == 'history':
        return [{**item, 'report': _artifacts[key].get('report.json')} for key, item in _metadata.items()]
    if route == 'import':
        raw = data['content']
        if len(raw.encode('utf-8')) > 10 * 1024 * 1024:
            raise ValueError('Model exceeds the 10 MB import limit.')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            path.write_text(raw, encoding='utf-8')
            loaded = load_model(path)
        key = 'model-' + uuid.uuid4().hex[:12]
        _models[key] = loaded
        _artifacts[key] = {'model.json': loaded}
        _metadata[key] = {'id': key, 'name': loaded['name'], 'modified': datetime.now(timezone.utc).isoformat()}
        return {'id': key, 'name': loaded['name']}
    if route == 'artifacts':
        key = data.get('model', 'demo')
        if key not in _artifacts:
            raise ValueError('Model not found. Import it again.')
        return _artifacts[key]
    if route == 'preview':
        return review_training(data)
    if route == 'score':
        email = data.get('email', '').strip()
        if not email:
            raise ValueError('Enter an email address.')
        return predict(email, model(), explain=True)
    if route == 'batch':
        return {'results': score_batch(data['file'], model())}
    if route == 'evaluate':
        flagged, ordinary = (parse_file(data[k]) for k in ('flagged', 'ordinary'))
        if not flagged or not ordinary:
            raise ValueError('Both holdout files must contain email addresses.')
        with tempfile.TemporaryDirectory() as tmp:
            f, o = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
            f.write_text('\n'.join(flagged), encoding='utf-8')
            o.write_text('\n'.join(ordinary), encoding='utf-8')
            return evaluate(f, o, model(), include_scores=True, include_breakdowns=True)
    if route == 'train':
        opts = training_options(data)
        domain_files = data.get('domain_files') or {}
        if bool(domain_files.get('flagged')) != bool(domain_files.get('ordinary')):
            raise ValueError('Provide both flagged and ordinary domain files, or leave both empty.')
        domains = {k: parse_domain_file(domain_files[k])['domains'] for k in ('flagged', 'ordinary')} if domain_files.get('flagged') else None
        excluded = data.get('exclude', {})
        flagged, ordinary = ([email for _, email in selected_rows(data[k], excluded.get(k))] for k in ('flagged', 'ordinary'))
        if not flagged or not ordinary:
            raise ValueError('Both files must contain email addresses.')
        key = 'model-' + uuid.uuid4().hex[:12]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f, o, target = root / 'flagged.txt', root / 'ordinary.txt', root / key
            f.write_text('\n'.join(flagged), encoding='utf-8')
            o.write_text('\n'.join(ordinary), encoding='utf-8')
            result = train(f, o, target, name=data.get('name', '').strip()[:80] or key, progress=progress, domain_data=domains, **opts)
            artifacts = {name: json.loads((target / name).read_text()) for name in ('model.json', 'report.json', 'golden.json')}
            loaded = load_model(target)
        _models[key], _artifacts[key] = loaded, artifacts
        _metadata[key] = {'id': key, 'name': loaded['name'], 'modified': datetime.now(timezone.utc).isoformat()}
        return {'id': key, 'result': result, 'report': artifacts['report.json']}
    raise ValueError('Unknown operation.')
