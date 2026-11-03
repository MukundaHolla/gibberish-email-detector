"""Local-only web interface for the detector. Run with Python 3.11+."""
import csv
import io
import json
import math
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from engine import ROOT, load_model, predict, preprocess
from training import evaluate, train

RUNS = ROOT / 'runs'
UI = ROOT / 'ui'
MAX_BODY = 20 * 1024 * 1024
JOBS = {}
JOB_LOCK = threading.Lock()


def training_options(data):
    options = data.get('options', {})
    min_df = options.get('min_df', 2)
    min_coverage = options.get('min_coverage', .20)
    max_fpr = options.get('max_fpr', .05)
    if type(min_df) is not int or min_df < 1:
        raise ValueError('Minimum fragment frequency must be a positive whole number.')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in (min_coverage, max_fpr)):
        raise ValueError('Coverage and false-alert budget must be between 0 and 1.')
    return {'min_df': min_df, 'min_coverage': min_coverage, 'max_fpr': max_fpr}


def run_training(data, progress=None):
    options = training_options(data)
    excluded = data.get('exclude', {})
    flagged, ordinary = ([email for _, email in selected_rows(data[k], excluded.get(k))] for k in ('flagged', 'ordinary'))
    if not flagged or not ordinary:
        raise ValueError('Both files must contain email addresses.')
    RUNS.mkdir(exist_ok=True)
    identifier = 'model-' + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory() as tmp:
        f, o = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
        f.write_text('\n'.join(flagged), encoding='utf-8')
        o.write_text('\n'.join(ordinary), encoding='utf-8')
        result = train(f, o, RUNS / identifier, name=data.get('name', '').strip()[:80] or identifier,
                       progress=progress, **options)
    report = json.loads((RUNS / identifier / 'report.json').read_text())
    return {'id': identifier, 'result': result, 'report': report}


def start_training(data):
    training_options(data)
    with JOB_LOCK:
        if any(job['status'] == 'running' for job in JOBS.values()):
            raise ValueError('A training run is already in progress. Wait for it to finish.')
        while len(JOBS) >= 20:
            del JOBS[next(iter(JOBS))]
        identifier = uuid.uuid4().hex
        JOBS[identifier] = {'status': 'running', 'stage': 'Reading uploaded files'}

    def worker():
        def progress(stage):
            with JOB_LOCK:
                JOBS[identifier]['stage'] = stage
        try:
            result = run_training(data, progress)
            with JOB_LOCK:
                JOBS[identifier] = {'status': 'complete', 'stage': 'Complete', 'result': result}
        except ModuleNotFoundError as exc:
            message = 'Training requires scikit-learn. Install it with: python -m pip install -r requirements.txt' if exc.name == 'sklearn' else 'A required Python module is missing.'
            with JOB_LOCK:
                JOBS[identifier] = {'status': 'failed', 'stage': 'Training failed', 'error': message}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            with JOB_LOCK:
                JOBS[identifier] = {'status': 'failed', 'stage': 'Training failed', 'error': str(exc)}
        except Exception:
            with JOB_LOCK:
                JOBS[identifier] = {'status': 'failed', 'stage': 'Training failed', 'error': 'Training failed unexpectedly. Check the server terminal for details.'}
            import traceback
            traceback.print_exc()
    threading.Thread(target=worker, daemon=True).start()
    return identifier


def parse_file(data):
    return [email for _, email in parse_rows(data)]


def parse_rows(data):
    name = data.get('name', '')
    content = data.get('content', '')
    if name.lower().endswith('.txt'):
        return [(n, line.strip()) for n, line in enumerate(content.lstrip('\ufeff').splitlines(), 1) if line.strip()]
    if not name.lower().endswith('.csv'):
        raise ValueError('Choose a .txt or .csv file.')
    reader = csv.DictReader(io.StringIO(content.lstrip('\ufeff'), newline=''))
    if not reader.fieldnames:
        raise ValueError(f'{name}: CSV needs a header row.')
    column = data.get('column', '').strip()
    if not column:
        matches = [f for f in reader.fieldnames if f.strip().lower() in ('email', 'email address', 'email_address')]
        if len(matches) != 1:
            raise ValueError(f'{name}: enter the email column name. Available columns: {", ".join(reader.fieldnames)}')
        column = matches[0]
    if column not in reader.fieldnames:
        raise ValueError(f'{name}: column {column!r} was not found. Available columns: {", ".join(reader.fieldnames)}')
    rows = []
    for row in reader:
        if row.get(column) and row[column].strip():
            rows.append((reader.line_num, row[column].strip()))
    return rows


def score_batch(data, model):
    rows = parse_rows(data)
    if not rows:
        raise ValueError('The file does not contain any email addresses.')
    if len(rows) > 10000:
        raise ValueError('The interface supports up to 10,000 addresses per batch. Use the CLI for larger files.')
    return [{'line': line, 'email': email, **predict(email, model)} for line, email in rows]


def selected_rows(data, excluded=None):
    excluded = set(excluded or [])
    return [(line, email) for line, email in parse_rows(data) if line not in excluded]


def review_training(data):
    """Review the same representations and duplicate rules used by training."""
    excluded = data.get('exclude', {})
    groups = {}
    counts = {}
    for label in ('ordinary', 'flagged'):
        rows = selected_rows(data[label], excluded.get(label))
        counts[label] = {'rows': len(rows), 'eligible_unique': 0, 'duplicate_lines': 0, 'unsupported_rows': 0}
        for line, email in rows:
            parts, reason = preprocess(email)
            key = parts if parts is not None else email.strip().lower()
            groups.setdefault(key, []).append({'label': label, 'line': line, 'email': email, 'reason': reason})
            if reason:
                counts[label]['unsupported_rows'] += 1
    issues = []
    conflicts = 0
    for group in groups.values():
        labels = {r['label'] for r in group}
        conflict = len(labels) > 1
        conflicts += conflict
        for label in labels:
            same = [r for r in group if r['label'] == label]
            counts[label]['duplicate_lines'] += len(same) - 1
            if not same[0]['reason']:
                counts[label]['eligible_unique'] += 1
        for row in group:
            tags = []
            if conflict:
                tags.append('conflicting labels')
            if sum(r['label'] == row['label'] for r in group) > 1:
                tags.append('duplicate model input')
            if row['reason']:
                tags.append(row['reason'].replace('_', ' '))
            if tags:
                issues.append({**row, 'issues': tags})
    return {'counts': counts, 'conflict_groups': conflicts, 'issues': issues}


def models():
    result = [{'id': 'demo', 'name': 'Demo · invented examples'}]
    if RUNS.exists():
        for path in sorted(RUNS.iterdir(), reverse=True):
            if path.name != 'demo' and re.fullmatch(r'[a-zA-Z0-9_-]+', path.name) and path.is_dir() and (path / 'model.json').is_file():
                try:
                    model = load_model(path)
                    result.append({'id': path.name, 'name': model['name']})
                except (ValueError, OSError):
                    continue
    return result


def model_history():
    result = []
    for entry in models():
        directory = ROOT / 'demo' if entry['id'] == 'demo' else RUNS / entry['id']
        report_file = directory / 'report.json'
        report = None
        try:
            parsed = json.loads(report_file.read_text(encoding='utf-8'))
            if isinstance(parsed, dict):
                report = parsed
        except (OSError, ValueError):
            pass
        model_file = directory / 'model.json'
        modified = datetime.fromtimestamp(model_file.stat().st_mtime, timezone.utc).isoformat()
        result.append({**entry, 'modified': modified, 'report': report})
    return result


def get_model(key):
    if key == 'demo':
        return load_model()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', key or ''):
        raise ValueError('Invalid model selection.')
    return load_model(RUNS / key)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, kind='application/json; charset=utf-8'):
        body = json.dumps(data, allow_nan=False).encode() if kind.startswith('application/json') else data
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/models':
            return self.respond(200, models())
        if path == '/api/history':
            return self.respond(200, model_history())
        if path.startswith('/api/training-status/'):
            identifier = path.rsplit('/', 1)[-1]
            with JOB_LOCK:
                job = dict(JOBS[identifier]) if identifier in JOBS else None
            return self.respond(200, job) if job else self.respond(404, {'error': 'Training job not found; the server may have restarted.'})
        files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/review.js': ('review.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path in files:
            name, kind = files[path]
            return self.respond(200, (UI / name).read_bytes(), kind)
        self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        # Browser requests must originate from this local server.
        if self.headers.get('Origin') != f'http://{self.headers.get("Host")}' or self.headers.get('Host') not in ('127.0.0.1:8765', 'localhost:8765'):
            return self.respond(403, {'error': 'Invalid request origin.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BODY:
                raise ValueError('Request is empty or exceeds the 20 MB limit.')
            data = json.loads(self.rfile.read(length))
            path = urlparse(self.path).path
            if path == '/api/preview':
                self.respond(200, review_training(data))
            elif path == '/api/train':
                self.respond(200, run_training(data))
            elif path == '/api/train/start':
                self.respond(202, {'job': start_training(data)})
            elif path == '/api/evaluate':
                flagged, ordinary = (parse_file(data[k]) for k in ('flagged', 'ordinary'))
                if not flagged or not ordinary:
                    raise ValueError('Both holdout files must contain email addresses.')
                with tempfile.TemporaryDirectory() as tmp:
                    f, o = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
                    f.write_text('\n'.join(flagged), encoding='utf-8')
                    o.write_text('\n'.join(ordinary), encoding='utf-8')
                    result = evaluate(f, o, get_model(data.get('model', 'demo')), include_scores=True, include_breakdowns=True)
                self.respond(200, result)
            elif path == '/api/score':
                email = data.get('email', '').strip()
                if not email:
                    raise ValueError('Enter an email address.')
                result = predict(email, get_model(data.get('model', 'demo')), explain=True)
                self.respond(200, result)
            elif path == '/api/batch':
                result = score_batch(data['file'], get_model(data.get('model', 'demo')))
                self.respond(200, {'results': result})
            else:
                self.respond(404, {'error': 'Not found'})
        except ModuleNotFoundError as e:
            if e.name != 'sklearn':
                raise
            self.respond(400, {'error': 'Training requires scikit-learn. Install it with: python -m pip install -r requirements.txt'})
        except (ValueError, KeyError, TypeError, OSError, csv.Error, json.JSONDecodeError) as e:
            self.respond(400, {'error': str(e)})


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    print('Detector UI: http://127.0.0.1:8765', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
