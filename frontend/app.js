/* ============================================================
   ECONAI — app.js
   Navbar: scroll effects · active links · mobile menu
   ============================================================ */

(function () {
  'use strict';

  /* ── Element References ─────────────────────────────────── */
  const navbar = document.getElementById('navbar');
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('navLinks');
  const allNavLinks = document.querySelectorAll('.navbar__link');

  /* Sections that correspond to each nav link */
  const sections = [
    { id: 'dashboard', navId: 'nav-dashboard' },
    { id: 'predict-gdp', navId: 'nav-predict' },
    { id: 'classify', navId: 'nav-classify' },
    { id: 'insights', navId: 'nav-insights' },
  ];

  /* ============================================================
     1. SCROLL-AWARE NAVBAR (glassmorphism intensity)
     ============================================================ */
  function handleScroll() {
    if (window.scrollY > 12) {
      navbar.classList.add('navbar--scrolled');
    } else {
      navbar.classList.remove('navbar--scrolled');
    }
  }

  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll(); // run once on load

  /* ============================================================
     2. ACTIVE LINK TRACKING (IntersectionObserver)
     Highlights the nav link whose section is most visible
     ============================================================ */
  function setActiveLink(sectionId) {
    allNavLinks.forEach(link => link.classList.remove('navbar__link--active'));

    const match = sections.find(s => s.id === sectionId);
    if (match) {
      const activeLink = document.getElementById(match.navId);
      if (activeLink) activeLink.classList.add('navbar__link--active');
    }
  }

  /* Track which section is currently intersecting */
  const visibleSections = new Map();

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        visibleSections.set(entry.target.id, entry.intersectionRatio);
      });

      /* Pick the section with the highest intersection ratio */
      let topSection = null;
      let topRatio = 0;

      visibleSections.forEach((ratio, id) => {
        if (ratio > topRatio) {
          topRatio = ratio;
          topSection = id;
        }
      });

      if (topSection) setActiveLink(topSection);
    },
    {
      rootMargin: `-${getNavbarHeight()}px 0px -40% 0px`,
      threshold: [0, 0.1, 0.25, 0.5, 0.75, 1.0],
    }
  );

  sections.forEach(({ id }) => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });

  /* ============================================================
     3. MOBILE HAMBURGER MENU
     ============================================================ */
  function openMenu() {
    navLinks.classList.add('is-open');
    hamburger.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden'; // prevent scroll behind menu
  }

  function closeMenu() {
    navLinks.classList.remove('is-open');
    hamburger.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  function toggleMenu() {
    const isOpen = hamburger.getAttribute('aria-expanded') === 'true';
    isOpen ? closeMenu() : openMenu();
  }

  hamburger.addEventListener('click', toggleMenu);

  /* Close menu when a link is clicked */
  navLinks.addEventListener('click', (e) => {
    if (e.target.classList.contains('navbar__link')) {
      closeMenu();
    }
  });

  /* Close menu on outside click */
  document.addEventListener('click', (e) => {
    const isOpen = hamburger.getAttribute('aria-expanded') === 'true';
    if (isOpen && !navbar.contains(e.target)) {
      closeMenu();
    }
  });

  /* Close menu on Escape key */
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMenu();
  });

  /* Close and reopen on resize (avoid stuck-open state) */
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
      closeMenu();
    }
  });

  /* ============================================================
     4. SMOOTH SCROLL (with navbar offset)
     ============================================================ */
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (!target) return;

      e.preventDefault();
      closeMenu();

      const offset = getNavbarHeight() + 16;
      const targetTop = target.getBoundingClientRect().top + window.scrollY - offset;

      window.scrollTo({ top: targetTop, behavior: 'smooth' });
    });
  });

  /* ============================================================
     UTILITIES
     ============================================================ */
  function getNavbarHeight() {
    return navbar ? navbar.offsetHeight : 68;
  }

})();
