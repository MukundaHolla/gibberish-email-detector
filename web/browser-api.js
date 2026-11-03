/* Translate the existing UI's local API calls into browser-worker operations. */
const browserWorker = new Worker('./worker.js', {type:'module'});
const workerPending = new Map();
const browserJobs = new Map();
let sequence = 0;
let workerFailure = null;
function browserRequest(route, data={}, progress=()=>{}) {
  if (workerFailure) return Promise.reject(new Error(workerFailure));
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    workerPending.set(id, {resolve, reject, progress});
    browserWorker.postMessage({id, route, data});
  });
}
browserWorker.onmessage = event => {
  const {id, result, error, progress} = event.data;
  const call = workerPending.get(id);
  if (!call) return;
  if (progress) { call.progress(progress); return; }
  workerPending.delete(id);
  if (error) call.reject(new Error(error)); else call.resolve(result);
};
browserWorker.onerror = () => {
  workerFailure = 'The browser worker could not start. Check your connection and reload the page.';
  for (const call of workerPending.values()) call.reject(new Error(workerFailure));
  workerPending.clear();
};
const networkFetch = window.fetch.bind(window);
window.fetch = async (url, options={}) => {
  if (typeof url !== 'string' || !url.startsWith('/api/')) return networkFetch(url, options);
  try {
    let value;
    const payload = options.body ? JSON.parse(options.body) : {};
    if (url === '/api/train/start') {
      if ([...browserJobs.values()].some(job => job.status === 'running')) throw new Error('A training run is already in progress.');
      const id = crypto.randomUUID();
      const job = {status:'running',stage:'Preparing training'};
      browserJobs.set(id, job);
      browserRequest('train', payload, progress => {job.stage=progress;}).then(result=>{
        job.status='complete'; job.result=result; job.stage='Complete';
      }).catch(error=>{job.status='failed';job.error=error.message;});
      value = {job:id};
    } else if (url.startsWith('/api/training-status/')) {
      value = browserJobs.get(decodeURIComponent(url.split('/').pop()));
      if (!value) throw new Error('Training job not found.');
    } else {
      const route = {'/api/models':'models','/api/history':'history','/api/preview':'preview','/api/score':'score','/api/batch':'batch','/api/evaluate':'evaluate'}[url];
      if (!route) throw new Error('Unknown operation.');
      value = await browserRequest(route, payload, message => {
        const notice = document.getElementById('runtime-status');
        if (notice) notice.textContent = message + '…';
      });
      const notice = document.getElementById('runtime-status');
      if (notice) notice.textContent = 'Ready · processing happens in your browser';
    }
    return new Response(JSON.stringify(value), {status:200,headers:{'Content-Type':'application/json'}});
  } catch (error) {
    const notice = document.getElementById('runtime-status');
    if (notice && workerFailure) notice.textContent=workerFailure;
    return new Response(JSON.stringify({error:error.message}), {status:400,headers:{'Content-Type':'application/json'}});
  }
};
