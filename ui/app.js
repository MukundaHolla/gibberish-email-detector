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
  const current = Object.fromEntries(['model','batch-model','evaluate-model'].map(id => [id, selected || $(id).value]));
  const response = await fetch('/api/models');
  const models = await response.json();
  $('model').innerHTML = models.map(m => `<option value="${escapeHTML(m.id)}">${escapeHTML(m.name)}</option>`).join('');
  $('batch-model').innerHTML = $('model').innerHTML;
  $('evaluate-model').innerHTML = $('model').innerHTML;
  for (const id of Object.keys(current)) if (models.some(m => m.id === current[id])) $(id).value=current[id];
  await refreshHistory(selected);
}
let trainingExcluded = {flagged: [], ordinary: []};
function clearReview() {
  trainingExcluded = {flagged: [], ordinary: []};
  $('preview-result').innerHTML = '';
  status('training-status', '');
}
for (const id of ['flagged', 'ordinary', 'flagged-column', 'ordinary-column']) {
  $(id).addEventListener(id.endsWith('column') ? 'input' : 'change', clearReview);
}
async function reviewFiles() {
  try {
    status('training-status','Checking files…');
    const data = await api('/api/preview', {...await files(), exclude:trainingExcluded});
    const totalExcluded = trainingExcluded.flagged.length + trainingExcluded.ordinary.length;
    const table = data.issues.length ? `<p>Showing ${Math.min(data.issues.length, 200)} of ${data.issues.length} rows needing review. Select rows to leave out of this training run, then review again.</p><div class="table-wrap"><table><thead><tr><th>EXCLUDE</th><th>FILE</th><th>LINE</th><th>EMAIL</th><th>ISSUE</th></tr></thead><tbody>${data.issues.slice(0,200).map(r=>`<tr><td><input type="checkbox" class="review-exclude" data-label="${r.label}" data-line="${r.line}" aria-label="Exclude ${escapeHTML(r.label)} line ${r.line}"></td><td>${escapeHTML(r.label)}</td><td>${r.line}</td><td>${escapeHTML(r.email)}</td><td>${escapeHTML(r.issues.join(', '))}</td></tr>`).join('')}</tbody></table></div><button class="secondary" id="review-apply">Exclude selected &amp; review again</button>` : '<p>No duplicates, conflicting labels, or unsupported addresses found.</p>';
    $('preview-result').innerHTML = `<div class="result"><h4>Training data review</h4><p>Counts reflect the rows currently included. Eligible unique counts may include conflicting groups until resolved.</p><div class="metrics">${Object.entries(data.counts).map(([label,r])=>metric(r.rows, `${label} rows`) + metric(r.eligible_unique, `${label} eligible unique`) + metric(r.duplicate_lines, `${label} duplicate rows`) + metric(r.unsupported_rows, `${label} unsupported rows`)).join('')}${metric(data.conflict_groups, 'conflicting input groups')}</div><p>${totalExcluded} rows manually excluded for this run. Excluded rows are not removed from your files.</p>${table}${totalExcluded ? '<button class="secondary" id="review-reset">Restore excluded rows</button>' : ''}<p>Training requires both classes and enough independent username groups; a clean review does not guarantee training will succeed.</p></div>`;
    const apply = $('review-apply');
    if (apply) apply.onclick = () => {
      const selected = [...document.querySelectorAll('.review-exclude:checked')];
      if (!selected.length) { status('training-status','Select at least one row to exclude.',true); return; }
      for (const box of selected) trainingExcluded[box.dataset.label].push(Number(box.dataset.line));
      reviewFiles();
    };
    const reset = $('review-reset');
    if (reset) reset.onclick = () => { trainingExcluded = {flagged: [], ordinary: []}; reviewFiles(); };
    status('training-status', data.conflict_groups ? 'Resolve conflicting labels before training.' : 'Review complete.');
  } catch(e) { $('preview-result').innerHTML = ''; status('training-status',e.message,true); }
}
$('preview').onclick = reviewFiles;
$('train-button').onclick = async () => {
  const button = $('train-button');
  try {
    const input = await files();
    const minDfText = $('train-min-df').value, coverageText = $('train-coverage').value, fprText = $('train-fpr').value;
    const minDf = Number(minDfText), coverage = Number(coverageText), fpr = Number(fprText);
    if (!minDfText || !Number.isInteger(minDf) || minDf < 1) throw Error('Minimum fragment frequency must be a positive whole number.');
    if (!coverageText || !fprText || !Number.isFinite(coverage) || !Number.isFinite(fpr) || coverage < 0 || coverage > 100 || fpr < 0 || fpr > 100) throw Error('Coverage and false-alert budget must be percentages from 0 to 100.');
    button.disabled=true; $('train-result').innerHTML = ''; status('training-status','Starting training…');
    const started = await api('/api/train/start', {...input,exclude:trainingExcluded,name:$('model-name').value,options:{min_df:minDf,min_coverage:coverage/100,max_fpr:fpr/100}});
    let job;
    while (true) {
      const response = await fetch(`/api/training-status/${encodeURIComponent(started.job)}`);
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Could not check training status.');
      job = data;
      if (job.status === 'failed') throw Error(job.error);
      if (job.status === 'complete') break;
      status('training-status', `Training: ${job.stage}…`);
      await new Promise(resolve => setTimeout(resolve, 700));
    }
    const data = job.result;
    const r = data.report, t = r.test;
    $('train-result').innerHTML = `<div class="result"><h4>Model trained · ${escapeHTML(r.name)}</h4><p>Saved locally. Results below are from the held-out test split, including abstentions.</p><div class="metrics">${metric(pct(t.coverage),'scored coverage')}${metric(pct(t.recall),'recall')}${metric(pct(t.precision),'precision')}${metric(pct(t.ordinary_false_flag_rate),'ordinary false-flag rate')}</div><p>Test: ${t.true_flags} true flags, ${t.false_flags} false flags, ${t.missed_flags} missed flags, ${t.abstained} abstained. Threshold: ${r.chosen_threshold == null ? 'none — score only' : r.chosen_threshold.toFixed(4)}.</p><p>Split: ${r.split_counts.train} training · ${r.split_counts.validation} validation · ${r.split_counts.test} test. Check the saved report and evaluate on independent data before relying on these results.</p></div>`;
    await refreshModels(data.id); status('training-status','Training complete. Your new model is selected below.');
  } catch(e) {
    const message = button.disabled && e instanceof TypeError ? 'Connection lost. Training may still be running; check model history before starting another run.' : e.message;
    status('training-status',message,true);
  } finally { button.disabled=false; }
};
function scoreStory(r, address) {
  const names = {looks_gibberish:'Looks flagged', looks_ordinary:'Looks ordinary', unsure:'Unsure', score_only:'Score only'};
  const reasonText = {empty_input:'The address is empty.', unsupported_address_form:'This address uses a form the model does not support.', invalid_address_shape:'The address is not in a supported email format.', recognized_private_relay:'This is a recognized private relay address, which the model does not score.', no_recognized_registrable_domain:'The domain does not have a recognized registrable label.', component_too_short:'The username or domain label is too short to score.'};
  let conclusion;
  if (!r.parts) conclusion = reasonText[r.reason] || 'This address could not be prepared for scoring.';
  else if (r.decision === 'unsure') {
    const weak = ['user','domain'].filter(f => r.coverage[f].known === 0 || r.coverage[f].fraction < r.min_coverage).map(f => f === 'user' ? 'username' : 'domain label');
    conclusion = `The ${weak.join(' and ')} ${weak.length === 1 ? 'does' : 'do'} not have enough known pieces. Each part must have at least one known piece and reach the model’s ${pct(r.min_coverage)} minimum. The model stops here, so there is no score or flagged/ordinary decision.`;
  } else if (r.decision === 'score_only') conclusion = 'The model produced a score, but it has no saved decision threshold. It cannot label this address flagged or ordinary.';
  else conclusion = `The score is ${r.decision === 'looks_gibberish' ? 'at or above' : 'below'} the saved threshold, so the model suggests ${r.decision === 'looks_gibberish' ? 'flagged' : 'ordinary'}. This is a spelling-pattern suggestion, not a human label or proof about the address.`;
  const title = names[r.decision] || r.decision;
  let html = `<div class="result score-story"><div class="story-result-head"><div><span class="story-kicker">YOUR RESULT · ${escapeHTML(r.model)}</span><h3>${escapeHTML(title)}</h3><p>${escapeHTML(conclusion)}</p></div><span class="story-result-icon">${r.decision === 'unsure' ? '?' : '✳'}</span></div>`;
  if (!r.parts) return html + `<div class="story-invalid"><b>What happened?</b><p>${escapeHTML(conclusion)} No fragments or coverage were calculated.</p></div></div>`;
  const raw = address.trim().toLowerCase();
  html += `<div class="score-step"><div class="score-step-label"><span>01</span><div><b>First, split the address</b><p>The model makes a lowercase copy for analysis.</p></div></div><div class="address-split"><div><small>USERNAME · BEFORE @</small><strong>${escapeHTML(r.parts.username)}</strong></div><i>@</i><div><small>FULL DOMAIN · AFTER @</small><strong>${escapeHTML(raw.split('@')[1])}</strong></div></div><p>From the full domain, the model selects <b>${escapeHTML(r.parts.domain_label)}</b> as the registrable domain label. It leaves out other domain parts, such as subdomains and the suffix.</p></div>`;
  html += `<div class="score-step"><div class="score-step-label"><span>02</span><div><b>Break each part into pieces</b><p>Overlapping groups of 2, 3 and 4 characters are called fragments. A repeated piece counts each time it appears.</p></div></div><div class="fragment-legend"><span class="known-key"></span> Known to this model <span class="unknown-key"></span> Not known to this model</div><div class="coverage-panels">`;
  for (const [field,label,part] of [['user','Username',r.parts.username],['domain','Domain label',r.parts.domain_label]]) {
    const stats = r.coverage[field], known = new Set(r.known_fragments[field]);
    const pieces = r.fragments[field].map(g=>`<span class="score-fragment ${known.has(g) ? 'known' : 'unknown'}" title="${known.has(g) ? 'Known' : 'Not known'}">${escapeHTML(g)}</span>`).join('');
    html += `<div class="coverage-panel"><div class="coverage-heading"><div><small>${escapeHTML(label.toUpperCase())}</small><b>${escapeHTML(part)}</b></div><strong>${stats.known} <i>/</i> ${stats.total}</strong></div><div class="score-fragments">${pieces}</div><div class="coverage-equation"><b>${stats.known} known</b> ÷ ${stats.total} total = <strong>${pct(stats.fraction)} coverage</strong></div><div class="coverage-bar"><span style="width:${Math.max(0,Math.min(100,stats.fraction*100))}%"></span></div><p>${stats.known === 0 || stats.fraction < r.min_coverage ? 'Below the requirement (or no known pieces).' : 'Meets the coverage requirement.'}</p></div>`;
  }
  html += `</div><p class="coverage-explainer">The minimum is <b>${pct(r.min_coverage)}</b> for each part, with at least one known piece. “Known” means the fragment is in this model’s vocabulary; it does not mean the complete address was in the training data.</p></div>`;
  html += `<div class="score-step"><div class="score-step-label"><span>03</span><div><b>How the model reached its result</b><p>${r.score == null ? 'The model stopped before calculating a score.' : 'Known pieces contribute in different directions according to what the model learned.'}</p></div></div>`;
  if (r.math) {
    const num = n => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(4)}`;
    html += `<div class="score-math"><div><small>STARTING VALUE</small><b>${num(r.math.intercept)}</b></div><span>+</span><div><small>USERNAME PIECES</small><b>${num(r.math.user_sum)}</b></div><span>+</span><div><small>DOMAIN PIECES</small><b>${num(r.math.domain_sum)}</b></div><span>=</span><div><small>COMBINED VALUE</small><b>${num(r.math.margin)}</b></div></div><p>The model combines its starting value with the effects of the known pieces, then converts that value to a score between 0 and 1. Positive effects push toward flagged; negative effects push toward ordinary. Values above are rounded.</p><div class="score-verdict"><div><small>MODEL SCORE</small><b>${r.score.toFixed(6)}</b></div><span>${r.threshold == null ? '→' : r.decision === 'looks_gibberish' ? '≥' : '<'}</span><div><small>SAVED THRESHOLD</small><b>${r.threshold == null ? 'None' : r.threshold.toFixed(6)}</b></div><span>→</span><div><small>RESULT</small><b>${escapeHTML(title)}</b></div></div><p>${escapeHTML(conclusion)} Comparisons use full precision; displayed numbers are rounded.</p>`;
    if (r.contributions?.length) html += `<details class="score-details"><summary>See the strongest individual fragment effects</summary><div class="table-wrap"><table><thead><tr><th>FRAGMENT</th><th>LEARNED WEIGHT</th><th>EFFECT ON COMBINED VALUE</th></tr></thead><tbody>${r.contributions.slice(0,12).map(x=>`<tr><td>${escapeHTML(x.term)}</td><td>${x.weight.toFixed(4)}</td><td>${num(x.points)}</td></tr>`).join('')}</tbody></table></div><p>Effect combines the learned weight with the fragment’s normalized value; it is not a percentage.</p></details>`;
  } else html += `<div class="score-stop"><b>Coverage check → Unsure</b><p>${escapeHTML(conclusion)}</p></div>`;
  return html + `</div></div>`;
}
$('score').onclick = async () => {
  const button = $('score');
  try {
    const email = $('email').value;
    button.disabled = true;
    $('score-result').innerHTML = '';
    status('score-status','Analyzing…');
    const r = await api('/api/score',{email,model:$('model').value});
    $('score-result').innerHTML = scoreStory(r,email);
    status('score-status','');
  } catch(e) { $('score-result').innerHTML = ''; status('score-status',e.message,true); }
  finally { button.disabled = false; }
};
let batchResults = [];
let batchPage = 0;
let reviewsDirty = false;
window.addEventListener('beforeunload', event => {
  if (reviewsDirty) { event.preventDefault(); event.returnValue = ''; }
});
function reviewSummary() {
  const labeled = label => batchResults.filter(r => ReviewHelpers.labelForReview(r.decision, r.review) === label);
  const flagged = labeled('flagged'), ordinary = labeled('ordinary');
  const uncertain = batchResults.filter(r => r.review === 'uncertain').length;
  const invalidLines = [...flagged, ...ordinary].filter(r => /[\r\n]/.test(r.email)).length;
  $('review-count').textContent = `${flagged.length} flagged · ${ordinary.length} ordinary · ${uncertain} uncertain. ${invalidLines ? `${invalidLines} multiline addresses will be omitted from TXT exports; they remain in the CSV.` : ''}`;
  $('review-csv').disabled = !batchResults.some(r => r.review);
  $('review-flagged').disabled = !flagged.some(r => !/[\r\n]/.test(r.email));
  $('review-ordinary').disabled = !ordinary.some(r => !/[\r\n]/.test(r.email));
}
function renderBatch() {
  const filter = $('batch-filter').value;
  const matches = batchResults.map((row,index) => ({row,index})).filter(({row}) => filter === 'all' || row.decision === filter);
  const pages = Math.max(1, Math.ceil(matches.length / 100));
  batchPage = Math.min(batchPage, pages - 1);
  const start = batchPage * 100;
  $('batch-count').textContent = `Showing ${matches.length ? start + 1 : 0}–${Math.min(start + 100, matches.length)} of ${matches.length} matching results. Download includes all decisions.`;
  $('batch-page').textContent = `Page ${batchPage + 1} of ${pages}`;
  $('batch-prev').disabled = batchPage === 0;
  $('batch-next').disabled = batchPage >= pages - 1;
  $('batch-rows').innerHTML = matches.slice(start, start + 100).map(({row:r,index}) => {
    const binary = r.decision === 'looks_gibberish' || r.decision === 'looks_ordinary';
    const options = binary ? [['correct','Correct'],['incorrect','Incorrect'],['uncertain','Uncertain']] : [['flagged','Flagged'],['ordinary','Ordinary'],['uncertain','Uncertain']];
    return `<tr><td>${r.line}</td><td>${escapeHTML(r.email)}</td><td>${escapeHTML(r.decision.replaceAll('_',' '))}</td><td>${r.score == null ? '—' : r.score.toFixed(4)}</td><td>${escapeHTML((r.reason || '—').replaceAll('_',' '))}</td><td><select class="review-choice" data-index="${index}" aria-label="Review line ${r.line}"><option value="">Not reviewed</option>${options.map(([value,label])=>`<option value="${value}" ${r.review === value ? 'selected' : ''}>${label}</option>`).join('')}</select></td></tr>`;
  }).join('') || '<tr><td colspan="6">No results match this filter.</td></tr>';
  reviewSummary();
}
$('batch-run').onclick = async () => {
  const button = $('batch-run');
  if (batchResults.some(r => r.review) && !window.confirm('Starting a new batch clears the current reviews. Export them first if you want to keep them. Continue?')) return;
  batchResults = [];
  reviewsDirty = false;
  $('batch-output').hidden = true;
  try {
    const file = $('batch-file').files[0];
    if (!file) throw Error('Choose a TXT or CSV file.');
    button.disabled = true;
    status('batch-status', 'Analyzing your list…');
    const data = await api('/api/batch', {model:$('batch-model').value, file:{name:file.name, content:await file.text(), column:$('batch-column').value}});
    batchResults = data.results;
    const count = decision => batchResults.filter(r => r.decision === decision).length;
    $('batch-summary').innerHTML = `<h4>Analysis complete</h4><div class="metrics">${metric(batchResults.length, 'addresses')}${metric(count('looks_gibberish'), 'looks gibberish')}${metric(count('looks_ordinary'), 'looks ordinary')}${metric(count('unsure'), 'unsure')}${metric(count('score_only'), 'score only')}</div>`;
    $('batch-filter').value = 'all';
    batchPage = 0;
    renderBatch();
    $('batch-output').hidden = false;
    status('batch-status', '');
  } catch(e) { status('batch-status', e.message, true); } finally { button.disabled = false; }
};
$('batch-filter').onchange = () => { batchPage = 0; renderBatch(); };
$('batch-prev').onclick = () => { batchPage--; renderBatch(); };
$('batch-next').onclick = () => { batchPage++; renderBatch(); };
$('batch-rows').onchange = event => {
  if (!event.target.matches('.review-choice')) return;
  batchResults[Number(event.target.dataset.index)].review = event.target.value;
  reviewsDirty = true;
  reviewSummary();
};
function downloadFile(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = document.createElement('a');
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('review-csv').onclick = () => { downloadFile('reviewed-emails.csv', ReviewHelpers.reviewedCsv(batchResults), 'text/csv;charset=utf-8'); reviewsDirty = false; };
$('review-flagged').onclick = () => downloadFile('reviewed-flagged.txt', ReviewHelpers.labeledText(batchResults, 'flagged'), 'text/plain;charset=utf-8');
$('review-ordinary').onclick = () => downloadFile('reviewed-ordinary.txt', ReviewHelpers.labeledText(batchResults, 'ordinary'), 'text/plain;charset=utf-8');
$('batch-download').onclick = () => {
  const columns = ['line','email','decision','score','threshold','model','user_coverage','domain_coverage','reason'];
  const safe = value => {
    let text = String(value ?? '');
    if (/^[\s\u0000-\u001f]*[=+\-@]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const lines = [columns.map(safe).join(',')];
  for (const r of batchResults) {
    const row = {...r, user_coverage:r.coverage?.user.fraction, domain_coverage:r.coverage?.domain.fraction};
    lines.push(columns.map(key => safe(row[key])).join(','));
  }
  const url = URL.createObjectURL(new Blob(['\ufeff' + lines.join('\r\n') + '\r\n'], {type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'email-analysis.csv'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('evaluate-run').onclick = async () => {
  const button = $('evaluate-run');
  $('evaluate-result').innerHTML = '';
  try {
    const input = {};
    for (const label of ['flagged','ordinary']) {
      const file = $(`evaluate-${label}`).files[0];
      if (!file) throw Error(`Choose the ${label} holdout file.`);
      input[label] = {name:file.name, content:await file.text(), column:$(`evaluate-${label}-column`).value};
    }
    button.disabled = true;
    status('evaluate-status', 'Evaluating holdout data…');
    const report = await api('/api/evaluate', {...input, model:$('evaluate-model').value});
    const r = report.results;
    const ingestion = Object.entries(report.ingestion).map(([label, stats]) => {
      const excluded = Object.values(stats.excluded_unique).reduce((a, b) => a + b, 0);
      const reasons = Object.entries(stats.excluded_unique).map(([reason, n]) => `${reason.replaceAll('_',' ')}: ${n}`).join(', ');
      return `<tr><td>${escapeHTML(label)}</td><td>${stats.nonblank_lines}</td><td>${stats.eligible_unique}</td><td>${stats.duplicate_lines}</td><td>${excluded}${reasons ? ` (${escapeHTML(reasons)})` : ''}</td></tr>`;
    }).join('');
    $('evaluate-result').innerHTML = `<div class="result"><h4>Holdout results · ${escapeHTML(report.model)}</h4><p>Saved threshold: ${report.threshold == null ? 'none — this model returns scores without flagging' : report.threshold.toFixed(6)}. Only a gibberish decision counts as an alert; unsure and score-only results do not alert.</p><div class="metrics">${metric(pct(r.coverage),'scored coverage')}${metric(pct(r.recall),'recall')}${metric(pct(r.precision),'precision')}${metric(pct(r.ordinary_false_flag_rate),'ordinary false-flag rate')}</div><p>Of ${r.flagged} flagged examples, ${r.true_flags} were alerted and ${r.missed_flags} were not alerted (including abstentions). Of ${r.ordinary} ordinary examples, ${r.false_flags} were falsely alerted and ${r.ordinary_unflagged} were not alerted. ${r.abstained} examples had no score.</p><h4>Input review</h4><div class="table-wrap"><table><thead><tr><th>LABEL</th><th>NONBLANK</th><th>ELIGIBLE UNIQUE</th><th>DUPLICATES</th><th>EXCLUDED UNIQUE</th></tr></thead><tbody>${ingestion}</tbody></table></div><p>Duplicate model inputs count once. Excluded inputs are included as abstentions; excluded flagged examples count as missed flags. Precision depends on the holdout’s class balance, so use naturally sampled data to estimate real alert volume.</p><div class="threshold-panel"><div class="threshold-header"><div><h4>Explore a cutoff</h4><p>See how a different threshold changes alerts on these examples. This experiment does not change your saved model.</p></div><span class="threshold-value" id="threshold-value"></span></div><label class="threshold-label" for="threshold-range">EXPERIMENTAL THRESHOLD</label><input id="threshold-range" type="range" min="0" max="1" step="any" value="${report.threshold ?? 0.5}"><div class="threshold-ends"><span>0 · More alerts</span><span>1 · Fewer alerts</span></div><div id="threshold-metrics" class="metrics"></div><p id="threshold-detail"></p>${report.threshold == null ? '<p>This model has no saved cutoff; the slider is an explicit experiment.</p>' : '<button type="button" class="secondary" id="threshold-reset">Use saved threshold</button>'}<p class="threshold-note">Choosing a cutoff using this holdout makes it part of model development. Measure the chosen cutoff on a different independent holdout before relying on its results.</p></div></div>`;
    const slider = $('threshold-range');
    const updateThreshold = () => {
      const cutoff = Number(slider.value);
      let tp = 0, fp = 0;
      for (const [score, ordinary, flagged] of report.score_groups) {
        if (score >= cutoff) { tp += flagged; fp += ordinary; }
      }
      const missed = r.flagged - tp, precision = tp + fp ? tp / (tp + fp) : null;
      $('threshold-value').textContent = cutoff.toFixed(6);
      $('threshold-metrics').innerHTML = metric(tp + fp, 'total alerts') + metric(tp, 'true flags') + metric(fp, 'false flags') + metric(missed, 'missed flags') + metric(pct(r.flagged ? tp / r.flagged : null), 'recall') + metric(pct(precision), 'precision') + metric(pct(r.ordinary ? fp / r.ordinary : null), 'ordinary false-flag rate');
      $('threshold-detail').textContent = `${r.abstained} inputs remain unscored at every cutoff and do not alert; flagged abstentions remain missed flags.`;
    };
    slider.addEventListener('input', updateThreshold);
    if ($('threshold-reset')) $('threshold-reset').onclick = () => { slider.value = report.threshold; updateThreshold(); };
    updateThreshold();
    const breakdown = report.breakdowns;
    const table = groups => `<div class="table-wrap"><table class="group-table"><thead><tr><th>GROUP</th><th>FLAGGED</th><th>ORDINARY</th><th>TRUE FLAGS</th><th>MISSED FLAGS</th><th>FALSE FLAGS</th><th>UNSCORED</th><th>RECALL</th><th>FALSE-ALERT RATE</th></tr></thead><tbody>${groups.map(g=>`<tr><td>${escapeHTML(g.label)}</td><td>${g.flagged}</td><td>${g.ordinary}</td><td>${g.true_flags}</td><td>${g.missed_flags}</td><td>${g.false_flags}</td><td>${g.abstained}</td><td>${pct(g.recall)}</td><td>${pct(g.ordinary_false_flag_rate)}</td></tr>`).join('') || '<tr><td colspan="9">No eligible examples.</td></tr>'}</tbody></table></div>`;
    $('evaluate-result').insertAdjacentHTML('beforeend', `<div class="result"><h4>Explore performance by group</h4><p>These counts use the saved threshold shown above; the experimental slider does not change them. ${breakdown.excluded_before_grouping} unsupported examples were excluded before grouping but remain in the overall results. Unscored eligible examples remain in their group and flagged ones count as missed flags.</p><h4>Username fragment coverage</h4><p>Coverage measures how much of the username is represented in the model’s vocabulary; it does not identify language, ethnicity, or whether a name is legitimate.</p>${table(breakdown.username_coverage)}<h4>Registrable domain label</h4><p>Showing ${Math.min(100, breakdown.domain_labels.length)} of ${breakdown.domain_labels.length} labels, largest groups first. A label such as “example” can represent different full domains; this is not a provider identity or proof that a provider was seen during training.</p>${table(breakdown.domain_labels.slice(0,100))}<p>Small groups can give misleading rates, especially when a class is absent. Compare results on a later, independent sample and investigate patterns before drawing conclusions.</p></div>`);
    status('evaluate-status', 'Evaluation complete.');
  } catch(e) { status('evaluate-status', e.message, true); } finally { button.disabled = false; }
};
let historyModels = [];
function showComparison() {
  const a = historyModels.find(m => m.id === $('compare-first').value);
  const b = historyModels.find(m => m.id === $('compare-second').value);
  if (!a || !b) { $('compare-output').innerHTML = '<p>Two models are needed for comparison.</p>'; return; }
  const get = (model, key) => {
    const report = model.report;
    if (!report) return '—';
    const t = report.test || {}, split = report.split_counts || {}, ingestion = report.ingestion || {};
    const vals = {
      threshold: report.chosen_threshold == null ? 'No cutoff' : Number(report.chosen_threshold).toFixed(4),
      inputs: `${ingestion.flagged?.eligible_unique ?? '—'} flagged / ${ingestion.ordinary?.eligible_unique ?? '—'} ordinary`,
      split: `${split.train ?? '—'} / ${split.validation ?? '—'} / ${split.test ?? '—'}`,
      features: report.feature_count ?? '—', min_df: report.min_df ?? '—',
      min_coverage: pct(report.min_coverage), budget: pct(report.validation_max_ordinary_false_flag_rate),
      coverage: pct(t.coverage), recall: pct(t.recall), precision: pct(t.precision), fpr: pct(t.ordinary_false_flag_rate),
      counts: `${t.true_flags ?? '—'} / ${t.false_flags ?? '—'} / ${t.missed_flags ?? '—'} / ${t.abstained ?? '—'}`
    };
    return vals[key];
  };
  const rows = [['Eligible inputs','inputs'],['Train / validation / test','split'],['Features','features'],['Minimum document frequency','min_df'],['Minimum component coverage','min_coverage'],['Validation false-alert budget','budget'],['Chosen threshold','threshold'],['Test coverage','coverage'],['Test recall','recall'],['Test precision','precision'],['Test ordinary false-alert rate','fpr'],['Test true flags / false flags / misses / abstentions','counts']];
  $('compare-output').innerHTML = `<div class="table-wrap"><table class="comparison-table"><thead><tr><th>MEASURE</th><th>${escapeHTML(a.name)}</th><th>${escapeHTML(b.name)}</th></tr></thead><tbody>${rows.map(([label,key])=>`<tr><td>${escapeHTML(label)}</td><td>${escapeHTML(get(a,key))}</td><td>${escapeHTML(get(b,key))}</td></tr>`).join('')}</tbody></table></div>`;
}
async function refreshHistory(selected) {
  const response = await fetch('/api/history');
  if (!response.ok) throw Error('Could not load model history.');
  historyModels = await response.json();
  const first = $('compare-first').value, second = $('compare-second').value;
  const options = historyModels.map(m=>`<option value="${escapeHTML(m.id)}">${escapeHTML(m.name)} (${escapeHTML(m.id)})</option>`).join('');
  $('compare-first').innerHTML = options;
  $('compare-second').innerHTML = options;
  if (historyModels.some(m=>m.id===first)) $('compare-first').value=first;
  if (historyModels.some(m=>m.id===second)) $('compare-second').value=second;
  else if (historyModels.length > 1) $('compare-second').selectedIndex=1;
  if (selected && historyModels.some(m=>m.id===selected)) $('compare-second').value=selected;
  $('history-list').innerHTML = historyModels.map(m=>`<div class="model-row"><div><strong>${escapeHTML(m.name)}</strong><span>${escapeHTML(m.id)} · file modified ${escapeHTML(new Date(m.modified).toLocaleString())}${m.id === 'demo' ? ' · invented demo data' : ''}${m.report ? '' : ' · report unavailable'}</span></div><button class="secondary model-use" data-model="${escapeHTML(m.id)}">Use model</button></div>`).join('');
  document.querySelectorAll('.model-use').forEach(button=>button.onclick=()=>{
    for (const id of ['model','batch-model','evaluate-model']) $(id).value=button.dataset.model;
    const model = historyModels.find(m=>m.id===button.dataset.model);
    status('history-status', `${model.name} selected for testing and evaluation.`);
  });
  showComparison();
}
$('compare-first').onchange = showComparison;
$('compare-second').onchange = showComparison;
$('history-refresh').onclick = async () => {
  try { await refreshModels(); status('history-status', 'Model list refreshed.'); }
  catch(e) { status('history-status', e.message, true); }
};
refreshModels().catch(e=>status('score-status',e.message,true));
