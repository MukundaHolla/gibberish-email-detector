(() => {
  const available = new Set([...document.querySelectorAll('[data-view]')].map(el => el.dataset.view));
  function showPage() {
    const requested = location.hash.slice(1) || 'home';
    const page = available.has(requested) ? requested : 'home';
    document.querySelectorAll('[data-view]').forEach(el => { el.hidden = el.dataset.view !== page; });
    document.querySelectorAll('nav a[href^="#"]').forEach(link => {
      const active = link.getAttribute('href') === '#' + page;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current','page'); else link.removeAttribute('aria-current');
    });
    window.scrollTo(0,0);
    document.title = page === 'learn' ? 'How it works · Signal Studio' : 'Signal Studio · Email detector';
  }
  const toggle = document.getElementById('menu-toggle');
  toggle.addEventListener('click', () => {
    const collapsed = document.body.classList.toggle('nav-collapsed');
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute('aria-label', collapsed ? 'Expand navigation' : 'Collapse navigation');
  });
  window.addEventListener('hashchange', showPage);
  showPage();
})();
