import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

let runtime;
let packagesReady = false;
async function initialize() {
  if (runtime) return runtime;
  const py = await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'});
  py.FS.mkdirTree('/app/assets'); py.FS.mkdirTree('/app/demo');
  for (const file of ['engine.py','training.py','local_ui.py','browser_backend.py','assets/public_suffix_list.dat','demo/model.json','demo/report.json','demo/golden.json']) {
    const response = await fetch('./' + file);
    if (!response.ok) throw new Error(`Could not load application asset: ${file}`);
    py.FS.writeFile('/app/' + file, new Uint8Array(await response.arrayBuffer()));
  }
  py.runPython("import sys, json; sys.path.insert(0, '/app'); import browser_backend; browser_backend.self_test()");
  runtime = py;
  return py;
}
let queue = Promise.resolve();
self.onmessage = event => {
  const {id, route, data} = event.data;
  queue = queue.then(async () => {
    try {
      self.postMessage({id, progress:'Loading the browser runtime'});
      const py = await initialize();
      if (route === 'train' && !packagesReady) {
        self.postMessage({id, progress:'Loading training software (first use may take a while)'});
        await py.loadPackage('scikit-learn');
        packagesReady = true;
      }
      py.globals.set('_request_json', JSON.stringify(data));
      py.globals.set('_request_route', route);
      py.globals.set('_progress_callback', message => self.postMessage({id, progress:String(message)}));
      const result = await py.runPythonAsync('json.dumps(browser_backend.handle(_request_route, json.loads(_request_json), _progress_callback), allow_nan=False)');
      py.globals.delete('_request_json');
      self.postMessage({id, result:JSON.parse(result)});
    } catch (error) {
      self.postMessage({id, error: String(error?.message || error).trim().split('\n').slice(-1)[0] || 'Browser operation failed.'});
    }
  });
};
