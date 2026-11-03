const modelFile = document.getElementById('model-import');
const modelStatus = document.getElementById('model-transfer-status');
document.getElementById('model-import-button').onclick = async () => {
  try {
    const file = modelFile.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.json')) throw Error('Choose a model.json file.');
    if (file.size > 10 * 1024 * 1024) throw Error('Model exceeds the 10 MB import limit.');
    modelStatus.textContent = 'Validating model…';
    const model = await browserRequest('import', {content:await file.text()});
    await refreshModels(model.id);
    modelStatus.textContent = `Imported ${model.name}. It is selected for testing and evaluation.`;
  } catch (error) { modelStatus.textContent=error.message; }
};
for (const [id, filename] of [['model-export','model.json'],['report-export','report.json'],['golden-export','golden.json']]) {
  document.getElementById(id).onclick = async () => {
    try {
      const artifacts = await browserRequest('artifacts', {model:document.getElementById('model').value});
      if (!artifacts[filename]) throw Error(`This model does not have ${filename}.`);
      downloadFile(filename, JSON.stringify(artifacts[filename], null, 2) + '\n', 'application/json;charset=utf-8');
      modelStatus.textContent = `${filename} downloaded.`;
    } catch (error) { modelStatus.textContent=error.message; }
  };
}
