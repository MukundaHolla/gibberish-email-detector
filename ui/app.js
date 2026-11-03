const $ = id => document.getElementById(id);
const escapeHTML = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = n => n == null ? '—' : `${(n * 100).toFixed(1)}%`;
const metric = (value, label) => `<div class="metric"><strong>${escapeHTML(value)}</strong><span>${escapeHTML(label)}</span></div>`;
async function api(path, payload) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Request failed');
  return data;
}
async function files() {
  const result = {};
  for (const label of ['flagged','ordinary']) {
    const file = $(label).files[0];
    if (!file) throw Error(`Choose a ${label} file.`);
    result[label] = {name:file.name, content:await file.text(), column:$(`${label}-column`).value};
  }
  return result;
}
function status(id, text, error=false) { $(id).textContent=text; $(id).classList.toggle('error', error); }
async function refreshModels(selected) {
  const response = await fetch('/api/models');
  const models = await response.json();
  $('model').innerHTML = models.map(m => `<option value="${escapeHTML(m.id)}">${escapeHTML(m.name)}</option>`).join('');
  if (selected) $('model').value=selected;
}
$('preview').onclick = async () => {
  try {
    status('training-status','Checking files…');
    const data = await api('/api/preview', await files());
    $('preview-result').innerHTML = `<div class="result"><h4>File preview</h4><div class="metrics">${Object.entries(data).map(([k,v]) => metric(v.rows, `${k} rows`) + metric(v.eligible, `${k} eligible before deduplication`)).join('')}</div>${Object.entries(data).map(([k,v])=>`<p><b>${escapeHTML(k)}</b> · excluded: ${escapeHTML(JSON.stringify(v.excluded))}<br>Sample: ${v.sample.map(escapeHTML).join(', ') || '—'}</p>`).join('')}<p>Training also checks duplicates, conflicting labels and whether there are enough independent groups.</p></div>`;
    status('training-status','Preview complete. Review your files before training.');
  } catch(e) { status('training-status',e.message,true); }
};
$('train-button').onclick = async () => {
  const button = $('train-button');
  try {
    const input = await files();
    button.disabled=true; status('training-status','Training in progress. This may take a moment…');
    const data = await api('/api/train', {...input,name:$('model-name').value});
    const r = data.report, t = r.test;
    $('train-result').innerHTML = `<div class="result"><h4>Model trained · ${escapeHTML(r.name)}</h4><p>Saved locally. Results below are from the held-out test split, including abstentions.</p><div class="metrics">${metric(pct(t.coverage),'scored coverage')}${metric(pct(t.recall),'recall')}${metric(pct(t.precision),'precision')}${metric(pct(t.ordinary_false_flag_rate),'ordinary false-flag rate')}</div><p>Test: ${t.true_flags} true flags, ${t.false_flags} false flags, ${t.missed_flags} missed flags, ${t.abstained} abstained. Threshold: ${r.chosen_threshold == null ? 'none — score only' : r.chosen_threshold.toFixed(4)}.</p><p>Split: ${r.split_counts.train} training · ${r.split_counts.validation} validation · ${r.split_counts.test} test. Check the saved report and evaluate on independent data before relying on these results.</p></div>`;
    await refreshModels(data.id); status('training-status','Training complete. Your new model is selected below.');
  } catch(e) { status('training-status',e.message,true); } finally { button.disabled=false; }
};
$('score').onclick = async () => {
  try {
    status('score-status','Analyzing…');
    const r = await api('/api/score',{email:$('email').value,model:$('model').value});
    let html = `<div class="result"><h4>${escapeHTML(r.decision.replaceAll('_',' '))}</h4><div class="metrics">${metric(r.score == null ? '—' : r.score.toFixed(4),'model score')}${metric(r.threshold == null ? '—' : r.threshold.toFixed(4),'decision threshold')}${r.coverage ? metric(pct(r.coverage.user.fraction),'username coverage') + metric(pct(r.coverage.domain.fraction),'domain coverage') : ''}</div>`;
    if (r.reason) html += `<p>Reason: ${escapeHTML(r.reason.replaceAll('_',' '))}</p>`;
    if (r.parts) html += `<p>Model inputs: <b>${escapeHTML(r.parts.username)}</b> (username) and <b>${escapeHTML(r.parts.domain_label)}</b> (domain label).</p>`;
    if (r.contributions?.length) html += `<h4>Strongest fragment contributions</h4><table><thead><tr><th>FRAGMENT</th><th>WEIGHT</th><th>CONTRIBUTION</th></tr></thead><tbody>${r.contributions.slice(0,8).map(x=>`<tr><td>${escapeHTML(x.term)}</td><td>${x.weight.toFixed(4)}</td><td>${x.points >= 0 ? '+' : ''}${x.points.toFixed(4)}</td></tr>`).join('')}</tbody></table><p>Positive contributions push toward flagged; negative contributions push toward ordinary.</p>`;
    $('score-result').innerHTML=html+'</div>'; status('score-status','');
  } catch(e) { status('score-status',e.message,true); }
};
refreshModels().catch(e=>status('score-status',e.message,true));
