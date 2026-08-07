/* Mycol documentation - shared behaviour.
   Used by index.html, faq.html and functionality.html. */

// Highlight the sticky-nav entry for whichever section is in view.
(function () {
  const links = [...document.querySelectorAll('.section-nav a[href^="#"]')];
  if (!links.length) return;

  const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const sections = [...document.querySelectorAll('main .page')];
  if (!sections.length) return;

  function setActive(id) {
    links.forEach(a => a.classList.toggle('active', a === byId.get(id)));
  }

  const observer = new IntersectionObserver((entries) => {
    // pick the entry nearest the top of the viewport that is still visible
    const visible = entries.filter(e => e.isIntersecting);
    if (!visible.length) return;
    visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    setActive(visible[0].target.id);
  }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });

  sections.forEach(s => observer.observe(s));
})();
