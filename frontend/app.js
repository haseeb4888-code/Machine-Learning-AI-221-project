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

/* ============================================================
   HERO — PARTICLE CANVAS ANIMATION
   ============================================================
   How it works:
   1. We grab the <canvas id="heroCanvas"> and set its pixel
      dimensions to match the hero section's size.
   2. We create an array of Particle objects. Each has a random
      position (x, y), velocity (vx, vy), radius, and colour
      (either blue or teal from our design system).
   3. On every animation frame (via requestAnimationFrame) we:
        a. Clear the canvas
        b. Move every particle by its velocity
        c. "Bounce" particles off the canvas edges
        d. Draw each particle as a filled circle
        e. For every pair of nearby particles (distance < 120px)
           draw a faint line between them — this creates the
           "connected dots" network effect.
   4. A ResizeObserver watches the hero so the canvas redraws
      at the correct size whenever the window changes.         */

(function () {
  'use strict';

  const canvas  = document.getElementById('heroCanvas');
  if (!canvas) return;           // guard: exit if element missing

  const ctx     = canvas.getContext('2d');
  const hero    = canvas.parentElement;  // the .hero section

  /* Brand colours with varying opacity */
  const COLOURS = [
    'rgba(59,  130, 246, 0.55)',   // blue
    'rgba(6,   182, 212, 0.55)',   // teal
    'rgba(59,  130, 246, 0.30)',   // blue (dim)
    'rgba(6,   182, 212, 0.30)',   // teal (dim)
  ];

  const PARTICLE_COUNT   = 55;   // total dots
  const MAX_LINK_DIST    = 130;  // px — max distance to draw a connecting line
  const SPEED            = 0.4;  // base movement speed (pixels per frame)

  let particles = [];
  let animId;                    // requestAnimationFrame handle (for cleanup)
  let W, H;                      // canvas dimensions

  /* ── Particle class ──────────────────────────────────── */
  function Particle() {
    this.reset(true);  // true = randomise Y across full height on init
  }

  Particle.prototype.reset = function (fullHeight) {
    this.x      = Math.random() * W;
    this.y      = fullHeight ? Math.random() * H : -10;  // start off-screen on re-spawn
    this.vx     = (Math.random() - 0.5) * SPEED * 2;    // random direction
    this.vy     = (Math.random() - 0.5) * SPEED * 2;
    this.r      = Math.random() * 2 + 1.5;               // radius: 1.5–3.5 px
    this.colour = COLOURS[Math.floor(Math.random() * COLOURS.length)];
  };

  Particle.prototype.update = function () {
    this.x += this.vx;
    this.y += this.vy;

    /* Bounce off left / right edges */
    if (this.x < 0)  { this.x  = 0;  this.vx *= -1; }
    if (this.x > W)  { this.x  = W;  this.vx *= -1; }

    /* Bounce off top / bottom edges */
    if (this.y < 0)  { this.y  = 0;  this.vy *= -1; }
    if (this.y > H)  { this.y  = H;  this.vy *= -1; }
  };

  Particle.prototype.draw = function () {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    ctx.fillStyle = this.colour;
    ctx.fill();
  };

  /* ── Setup / resize ──────────────────────────────────── */
  function resize() {
    /* Match canvas pixel size to the hero's rendered size.
       Without this the canvas would be blurry or clipped.   */
    W = canvas.width  = hero.offsetWidth;
    H = canvas.height = hero.offsetHeight;

    /* Recreate particles so they're distributed across the
       new dimensions rather than all bunched in a corner.   */
    particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle());
  }

  /* ── Draw connections between nearby particles ───────── */
  function drawLinks() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx   = particles[i].x - particles[j].x;
        const dy   = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < MAX_LINK_DIST) {
          /* The line fades out as particles move apart:
             alpha = 1 when dist=0, 0 when dist=MAX_LINK_DIST */
          const alpha = 1 - dist / MAX_LINK_DIST;

          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(59, 130, 246, ${alpha * 0.25})`;
          ctx.lineWidth   = 1;
          ctx.stroke();
        }
      }
    }
  }

  /* ── Main animation loop ─────────────────────────────── */
  function animate() {
    /* clearRect wipes the previous frame so particles don't
       leave a trail behind them.                            */
    ctx.clearRect(0, 0, W, H);

    particles.forEach(p => {
      p.update();
      p.draw();
    });

    drawLinks();

    animId = requestAnimationFrame(animate);
  }

  /* ── ResizeObserver ──────────────────────────────────── */
  /* ResizeObserver fires whenever the hero element changes
     size (e.g. window resize, orientation change). We call
     resize() to update canvas dimensions and re-seed particles. */
  const ro = new ResizeObserver(() => {
    resize();
  });
  ro.observe(hero);

  /* Kick everything off */
  resize();
  animate();

})();


/* ============================================================
   HERO — STAT CARD COUNT-UP ANIMATION
   ============================================================
   How it works:
   1. We find every <span class="stat-card__number"> in the page.
      Each has:
        data-target  = the final number to count up to  (e.g. "82")
        data-suffix  = text appended after the number   (e.g. "%")
        data-prefix  = text prepended before the number (e.g. "0.")
   2. We use a single IntersectionObserver. When the #heroStats
      container scrolls into view (threshold: 30%) we start
      animating all three counters simultaneously.
   3. Each counter uses requestAnimationFrame and an easing
      function so the number accelerates then decelerates,
      giving a satisfying "snap to final value" feel.          */

(function () {
  'use strict';

  const statsContainer = document.getElementById('heroStats');
  if (!statsContainer) return;

  const counters = statsContainer.querySelectorAll('.stat-card__number');
  let hasRun = false;   // only animate once

  /* ── Easing: easeOutExpo ─────────────────────────────── */
  /* t goes from 0 → 1 over the animation duration.
     easeOutExpo makes early frames fast and late frames slow,
     so the number "lands" on the target smoothly.            */
  function easeOutExpo(t) {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
  }

  /* ── Animate a single counter ────────────────────────── */
  function animateCounter(el) {
    const target   = parseFloat(el.dataset.target) || 0;  // final value
    const suffix   = el.dataset.suffix  || '';             // e.g. "%"
    const prefix   = el.dataset.prefix  || '';             // e.g. "0."
    const duration = 1600;                                 // ms
    const start    = performance.now();

    function tick(now) {
      /* t: how far through the animation (0 = start, 1 = end) */
      const elapsed = now - start;
      const t       = Math.min(elapsed / duration, 1);
      const eased   = easeOutExpo(t);

      /* current value rounded to nearest integer */
      const current = Math.round(eased * target);

      /* Write the formatted string into the element.
         e.g. prefix="0." target=78 suffix="" → "0.78"        */
      el.textContent = prefix + current + suffix;

      /* Keep looping until t reaches 1 */
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        /* Ensure we land on exactly the target (no rounding drift) */
        el.textContent = prefix + target + suffix;
      }
    }

    requestAnimationFrame(tick);
  }

  /* ── IntersectionObserver ────────────────────────────── */
  /* We wait until the stats grid is 30% visible before
     triggering the animation, so it starts just as the user
     scrolls to it — not silently in the background.          */
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !hasRun) {
          hasRun = true;
          counters.forEach(animateCounter);
          io.disconnect();  // stop observing once done
        }
      });
    },
    { threshold: 0.3 }
  );

  io.observe(statsContainer);

})();
