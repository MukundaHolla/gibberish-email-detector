"""Local-only web interface for the detector. Run with Python 3.11+."""
import csv
import io
import json
import re
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from engine import ROOT, load_model, predict, preprocess
from training import train

RUNS = ROOT / 'runs'
UI = ROOT / 'ui'
MAX_BODY = 20 * 1024 * 1024


def parse_file(data):
    name = data.get('name', '')
    content = data.get('content', '')
    if name.lower().endswith('.txt'):
        return [line.strip() for line in content.splitlines() if line.strip()]
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
    return [row[column].strip() for row in reader if row.get(column) and row[column].strip()]


def models():
    result = [{'id': 'demo', 'name': 'Demo · invented examples'}]
    if RUNS.exists():
        for path in sorted(RUNS.iterdir(), reverse=True):
            if path.is_dir() and (path / 'model.json').is_file():
                try:
                    model = load_model(path)
                    result.append({'id': path.name, 'name': model['name']})
                except (ValueError, OSError):
                    continue
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
        files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
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
                result = {}
                for label in ('flagged', 'ordinary'):
                    rows = parse_file(data[label])
                    reasons = {}
                    for row in rows:
                        _, reason = preprocess(row)
                        if reason:
                            reasons[reason] = reasons.get(reason, 0) + 1
                    result[label] = {'rows': len(rows), 'eligible': len(rows) - sum(reasons.values()), 'excluded': reasons, 'sample': rows[:3]}
                self.respond(200, result)
            elif path == '/api/train':
                flagged, ordinary = (parse_file(data[k]) for k in ('flagged', 'ordinary'))
                if not flagged or not ordinary:
                    raise ValueError('Both files must contain email addresses.')
                RUNS.mkdir(exist_ok=True)
                identifier = 'model-' + uuid.uuid4().hex[:12]
                with tempfile.TemporaryDirectory() as tmp:
                    f, o = Path(tmp) / 'flagged.txt', Path(tmp) / 'ordinary.txt'
                    f.write_text('\n'.join(flagged), encoding='utf-8')
                    o.write_text('\n'.join(ordinary), encoding='utf-8')
                    result = train(f, o, RUNS / identifier, name=data.get('name', '').strip()[:80] or identifier)
                report = json.loads((RUNS / identifier / 'report.json').read_text())
                self.respond(200, {'id': identifier, 'result': result, 'report': report})
            elif path == '/api/score':
                email = data.get('email', '').strip()
                if not email:
                    raise ValueError('Enter an email address.')
                result = predict(email, get_model(data.get('model', 'demo')), explain=True)
                self.respond(200, result)
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
