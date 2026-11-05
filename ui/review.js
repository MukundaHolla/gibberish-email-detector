/* Pure review and export helpers, shared with the local regression checks. */
(function (root) {
  function labelForReview(decision, review) {
    if (review === 'flagged' || review === 'ordinary') return review;
    if (review !== 'correct' && review !== 'incorrect') return null;
    if (decision !== 'looks_gibberish' && decision !== 'looks_ordinary' && decision !== 'flagged_domain') return null;
    const predicted = decision === 'looks_gibberish' || decision === 'flagged_domain' ? 'flagged' : 'ordinary';
    return review === 'correct' ? predicted : predicted === 'flagged' ? 'ordinary' : 'flagged';
  }
  function csvCell(value) {
    let text = String(value ?? '');
    if (/^[\s\u0000-\u001f]*[=+\-@]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  }
  function reviewedCsv(rows) {
    const columns = ['line','email','decision','score','model','review','label','reason','matched_domain','decision_source','email_decision','domain_model_decision','domain_model_score'];
    const lines = [columns.map(csvCell).join(',')];
    for (const r of rows.filter(row => row.review)) {
      const row = {...r, label:labelForReview(r.decision, r.review), domain_model_decision:r.domain_result?.decision, domain_model_score:r.domain_result?.score};
      lines.push(columns.map(key=>csvCell(row[key])).join(','));
    }
    return '\ufeff' + lines.join('\r\n') + '\r\n';
  }
  function labeledText(rows, label) {
    return rows.filter(r => labelForReview(r.decision, r.review) === label && !/[\r\n]/.test(r.email))
      .map(r => r.email).join('\n') + '\n';
  }
  const api = {labelForReview, reviewedCsv, labeledText};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewHelpers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
