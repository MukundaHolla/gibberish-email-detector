"""Build an explicit allowlist of browser assets for static Cloudflare hosting."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent.parent
out = root / 'web-dist'
out.mkdir(exist_ok=True)
files = {'engine.py':'engine.py', 'domain_rules.py':'domain_rules.py', 'domain_ml.py':'domain_ml.py', 'training.py':'training.py', 'local_ui.py':'local_ui.py',
         'assets/public_suffix_list.dat':'assets/public_suffix_list.dat',
         'assets/PUBLIC_SUFFIX_LICENSE.md':'assets/PUBLIC_SUFFIX_LICENSE.md',
         'demo/model.json':'demo/model.json', 'demo/report.json':'demo/report.json',
         'demo/golden.json':'demo/golden.json', 'ui/app.js':'app.js', 'ui/style.css':'style.css',
         'ui/review.js':'review.js', 'ui/routes.js':'routes.js', 'ui/learn.js':'learn.js', 'ui/guide.css':'guide.css', 'ui/theme.css':'theme.css', 'ui/favicon.svg':'favicon.svg', 'web/browser_backend.py':'browser_backend.py',
         'web/worker.js':'worker.js', 'web/browser-api.js':'browser-api.js',
         'web/model-controls.js':'model-controls.js'}
allowed = set(files.values()) | {'index.html'}
unexpected = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) not in allowed)
if unexpected:
    raise SystemExit('Refusing to build: web-dist contains unexpected files. Move them out before deployment: ' + ', '.join(unexpected))
for source, target in files.items():
    dest = out / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / source, dest)
html = (root / 'ui/index.html').read_text()
html = html.replace('Your data stays on this machine.', 'Your files stay in this browser.')
html = html.replace('The local app accepts up to 20 MiB for the entire training request, including both files and JSON encoding. The actual combined file size must be smaller. Training uses memory, so your computer can reach a practical limit sooner.', 'The browser trainer has no fixed file-size or row-count limit in the app. Both files and training data are processed in your browser memory, so the practical limit depends on your device and browser. Start small and increase gradually; large runs may be slow or run out of memory. Imported model files have a separate 10 MiB limit.')
html = html.replace('Running locally', 'In your browser')
html = html.replace('Training runs in the local server. Refreshing this page loses the progress view; stopping the server interrupts the run.', 'Training runs in this browser. Keep this page open until you download your model.')
html = html.replace('Saved models', 'Models in this session')
html = html.replace('Inspect saved runs and compare their training reports.', 'Inspect this session’s models and compare available training reports.')
html = html.replace('Select a model to use across testing, bulk analysis, and evaluation.', 'Select a session model to use across testing, bulk analysis, and evaluation.')
html = html.replace('Local by design &nbsp;·&nbsp; Built for exploration', 'Browser-based &nbsp;·&nbsp; Built for exploration')
html = html.replace('LOCAL WORKSPACE', 'BROWSER WORKSPACE')
html = html.replace('<!-- MODEL_NAV -->', '<a href="#models"><span>▣</span> Import &amp; export</a>')
html = html.replace('<!-- BROWSER_MODELS -->', '''<p id="runtime-status" class="runtime-indicator" role="status">Loading the browser runtime…</p><section class="feature-view" data-view="models" id="models" hidden><div class="section-heading"><div><div class="kicker">YOUR MODELS</div><h2>Import &amp; export</h2><p>Bring your own compatible model or save one for later.</p></div></div><section class="card browser-models"><div class="card-head"><div><h3>Use your own model</h3><p>Models are kept only for this browser session; download them before leaving.</p><p>The Python runtime downloads from jsDelivr. Your selected files are processed in your browser.</p></div><span class="badge">ON YOUR DEVICE</span></div><div class="browser-model-controls"><label>MODEL JSON<input id="model-import" type="file" accept=".json,application/json"></label><button class="secondary" id="model-import-button">Import model</button><button class="secondary" id="model-export">Download selected model</button><button class="secondary" id="report-export">Download report</button><button class="secondary" id="golden-export">Download reference cases</button></div><p id="model-transfer-status" class="status" role="status"></p><p class="training-note">Downloads use the model selected on the Test an email page. Newly trained or imported models are selected automatically.</p></section></section>''')
html = html.replace('<!-- LEARN_PAGE -->', (root / 'ui/learn.html').read_text())
html = html.replace('<script src="/review.js"></script><script src="/app.js"></script>', '<script src="./review.js"></script><script src="./browser-api.js"></script><script src="./app.js"></script><script src="./model-controls.js"></script>')
html = html.replace('href="/style.css"', 'href="./style.css"')
(out / 'index.html').write_text(html)
app = (out / 'app.js').read_text()
app = app.replace('Saved locally. Email metrics below are from its held-out test split.', 'Available in this session. Download the model before leaving. Email metrics below are from its held-out test split.')
app = app.replace(' · file modified ', ' · added to this session ')
(out / 'app.js').write_text(app)
with (out / 'style.css').open('a') as f:
    f.write('\n.browser-models{margin-top:22px}.browser-model-controls{display:flex;align-items:end;gap:9px;flex-wrap:wrap}.browser-model-controls label{font-size:9px;font-weight:750;letter-spacing:.1em;color:#74877a}.browser-model-controls input{display:block;margin-top:8px;font-size:11px;max-width:220px}.runtime-indicator{color:#789281;font-size:10px;margin:0 0 12px}\n')
print(f'Built {out} from {len(files)} allowlisted assets.')
