(() => {
  const $ = id => document.getElementById(id);
  const flows = {
    training: {
      overline: 'THE TRAINING JOURNEY', title: 'From examples to a model',
      steps: [
        {short:'Label', title:'Start with reviewed examples', icon:'01', text:'People place complete email addresses into flagged or ordinary lists. The model will learn to imitate those spelling labels.', example:'FLAGGED  ·  qzxv8801@gmail.com    /    ORDINARY  ·  mukunda@gmail.com'},
        {short:'Prepare', title:'Prepare and check the data', icon:'02', text:'The detector extracts a username and a registrable domain label. It removes duplicate model inputs and stops if the labels conflict.', example:'MUKUNDA@mail.vendor.co.in  →  mukunda  +  vendor'},
        {short:'Split', title:'Put examples into separate groups', icon:'03', text:'Related username groups stay together. Roughly three folds train, one selects a threshold, and one tests the finished model.', example:'TRAIN ≈ 60%    →    VALIDATE ≈ 20%    →    TEST ≈ 20%'},
        {short:'Fragments', title:'Turn spelling into numbers', icon:'04', text:'The model counts 2-, 3-, and 4-character fragments in the username and domain. TF-IDF gives the known fragments numerical values.', example:'mukunda  →  mu · uk · ku · un · nd · da · muk · …'},
        {short:'Learn', title:'Learn the weights', icon:'05', text:'Logistic regression fits weights that help distinguish the two labels. Minimum fragment frequency controls which fragments enter the vocabulary.', example:'KNOWN FRAGMENTS  +  LEARNED WEIGHTS  +  INTERCEPT'},
        {short:'Measure', title:'Choose a cutoff and test', icon:'06', text:'Validation chooses a cutoff within the observed false-alert budget. The frozen model is then measured on the test split and saved.', example:'MODEL.JSON  +  REPORT.JSON  +  REFERENCE CASES'}
      ]
    },
    prediction: {
      overline: 'THE DETECTION JOURNEY', title: 'From one address to a decision',
      steps: [
        {short:'Input', title:'Start with an address', icon:'01', text:'You select a saved model and provide an address. This detector looks at spelling; it does not contact the mailbox.', example:'mukunda@gmail.com'},
        {short:'Prepare', title:'Extract the model inputs', icon:'02', text:'The feature copy is lowercased, and the username and registrable domain label are extracted. Unsupported forms return unsure.', example:'mukunda@gmail.com  →  mukunda  +  gmail'},
        {short:'Fragments', title:'Find familiar fragments', icon:'03', text:'Overlapping fragments are generated for both components and matched against the model vocabulary.', example:'mukunda  →  mu · uk · ku · un · nd · da · …'},
        {short:'Coverage', title:'Check for enough coverage', icon:'04', text:'Each component must meet the model’s coverage floor. If too few fragments are known, the detector returns unsure rather than guessing.', example:'DEMO EXAMPLE  ·  USER 5 / 15 = 33.3%    /    DOMAIN 9 / 9 = 100%'},
        {short:'Score', title:'Combine the learned weights', icon:'05', text:'Known fragments are converted to normalized TF-IDF values. Their contributions and the intercept produce a margin and a score.', example:'INTERCEPT  +  USER CONTRIBUTIONS  +  DOMAIN CONTRIBUTIONS  →  SCORE'},
        {short:'Decide', title:'Apply the threshold', icon:'06', text:'If the score reaches the cutoff, the decision is looks gibberish; otherwise it is looks ordinary. A missing cutoff yields score only.', example:'DEMO EXAMPLE  ·  0.4126 < 0.4508  →  LOOKS ORDINARY'}
      ]
    }
  };
  let mode = 'training', step = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, timer;
  function render() {
    const flow = flows[mode], item = flow.steps[step];
    $('flow-overline').textContent = flow.overline;
    $('flow-title').textContent = flow.title;
    $('flow-count').textContent = `${String(step+1).padStart(2,'0')} / ${String(flow.steps.length).padStart(2,'0')}`;
    $('flow-rail').innerHTML = flow.steps.map((s,i) => `<button class="flow-node ${i===step?'current':''} ${i<step?'done':''}" data-step="${i}" aria-label="Step ${i+1}: ${s.short}" ${i===step?'aria-current="step"':''}><span>${String(i+1).padStart(2,'0')}</span><small>${s.short}</small></button>`).join('');
    $('flow-icon').textContent = item.icon;
    $('flow-stage-label').textContent = `STEP ${item.icon} / ${mode === 'training' ? 'TRAINING' : 'DETECTION'}`;
    $('flow-stage-title').textContent = item.title;
    $('flow-stage-text').textContent = item.text;
    $('flow-example').textContent = item.example;
    $('flow-toggle').innerHTML = playing ? 'Pause animation <span>Ⅱ</span>' : 'Play animation <span>▶</span>';
  }
  function restartTimer() {
    clearInterval(timer);
    if (playing) timer = setInterval(() => {
      if (document.querySelector('[data-view="learn"]').hidden || document.hidden) return;
      step = (step + 1) % flows[mode].steps.length; render();
    }, 3500);
  }
  document.querySelectorAll('[data-flow]').forEach(button => button.addEventListener('click', () => {
    mode = button.dataset.flow; step = 0;
    document.querySelectorAll('[data-flow]').forEach(b => { const active = b === button; b.classList.toggle('selected',active); b.setAttribute('aria-pressed',String(active)); });
    render(); restartTimer();
  }));
  $('flow-rail').addEventListener('click', event => {
    const button = event.target.closest('[data-step]'); if (!button) return;
    step = Number(button.dataset.step); playing = false; render(); restartTimer();
  });
  $('flow-prev').addEventListener('click', () => {step = (step + flows[mode].steps.length - 1) % flows[mode].steps.length; playing = false; render(); restartTimer();});
  $('flow-next').addEventListener('click', () => {step = (step + 1) % flows[mode].steps.length; playing = false; render(); restartTimer();});
  $('flow-toggle').addEventListener('click', () => {playing = !playing; render(); restartTimer();});
  document.querySelectorAll('[data-scroll]').forEach(button=>button.addEventListener('click',()=>document.getElementById(button.dataset.scroll).scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'})));
  render(); restartTimer();
})();
