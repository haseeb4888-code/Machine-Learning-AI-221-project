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


/* ============================================================
   GDP PREDICTOR — Form submit → API → Result card
   ============================================================ */
(function () {
  'use strict';

  /* ── Element references ─────────────────────────────────── */
  const form          = document.getElementById('gdpForm');
  const submitBtn     = document.getElementById('gdpSubmitBtn');
  const errorEl       = document.getElementById('gdpFormError');
  const resultCard    = document.getElementById('gdpResult');
  const resultValue   = document.getElementById('gdpResultValue');
  const confidenceFill= document.getElementById('gdpConfidenceFill');
  const confidencePct = document.getElementById('gdpConfidencePct');
  const modelUsed     = document.getElementById('gdpModelUsed');
  const pillsContainer  = document.getElementById('gdpModelPills');
  const gdpCompareBtn   = document.getElementById('gdpCompareAllBtn');
  const gdpCompareWrap  = document.getElementById('gdpCompareTableWrap');
  const gdpCompareTbody = document.getElementById('gdpCompareTableBody');

  if (!form) return;   // guard: section not in DOM yet

  /* API base URL — change to your deployed URL in production */
  const API_BASE = 'http://localhost:8000';

  /* All 6 regression models */
  const ALL_REGRESSORS = [
    { key: 'random_forest',      label: 'Random Forest'      },
    { key: 'linear_regression',  label: 'Linear Regression'  },
    { key: 'gradient_boosting',  label: 'Gradient Boosting'  },
    { key: 'xgboost',            label: 'XGBoost'            },
    { key: 'svm',                label: 'SVM'                },
    { key: 'mlp',                label: 'MLP'                },
  ];

  /* Currently selected model — 'best' means API picks automatically */
  let selectedModel = 'best';

  /* ── Pill selector ──────────────────────────────────────── */
  if (pillsContainer) {
    pillsContainer.addEventListener('click', (e) => {
      const pill = e.target.closest('.model-pill');
      if (!pill) return;
      pillsContainer.querySelectorAll('.model-pill').forEach(p =>
        p.classList.remove('model-pill--active')
      );
      pill.classList.add('model-pill--active');
      selectedModel = pill.dataset.model;
    });
  }

  /* ── Field names expected by the FastAPI /predict/gdp endpoint ── */
  const FIELD_NAMES = [
    'population', 'area', 'pop_density', 'coastline',
    'net_migration', 'infant_mortality', 'gdp_per_capita',
    'literacy', 'phones_per_1000', 'arable', 'crops', 'other',
    'climate', 'birthrate', 'deathrate',
    'agriculture', 'industry', 'service'
  ];
  const REGION_ID = 'region';   // string field handled separately

  /* ── Helpers ────────────────────────────────────────────── */

  /* Read all form fields and return a plain object.
     Returns null if any required field is empty/invalid. */
  function collectFormData() {
    const data = {};
    let valid = true;

    FIELD_NAMES.forEach(name => {
      const el  = document.getElementById(name);
      const val = el ? parseFloat(el.value) : NaN;

      if (el) el.classList.remove('is-invalid');

      if (isNaN(val) || el.value.trim() === '') {
        if (el) el.classList.add('is-invalid');
        valid = false;
      } else {
        data[name] = val;
      }
    });

    // region is a required string field
    const regionEl = document.getElementById(REGION_ID);
    if (regionEl) regionEl.classList.remove('is-invalid');
    if (!regionEl || regionEl.value.trim() === '') {
      if (regionEl) regionEl.classList.add('is-invalid');
      valid = false;
    } else {
      data['region'] = regionEl.value.trim();
    }

    return valid ? data : null;
  }

  /* Format a number as USD currency string, e.g. 12450 → "$12,450" */
  function formatUSD(value) {
    return '$' + Math.round(value).toLocaleString('en-US');
  }

  /* Set the submit button into loading / normal state */
  function setLoading(isLoading) {
    if (isLoading) {
      submitBtn.classList.add('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Predicting…';
    } else {
      submitBtn.classList.remove('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Predict GDP';
    }
  }

  /* Show an error message below the form */
  function showError(msg) {
    errorEl.textContent = msg;
  }

  function clearError() {
    errorEl.textContent = '';
  }

  /* Reveal the result card with the slide-up animation.
     `hidden` attribute keeps it out of layout; removing it
     lets the CSS opacity/transform transition play.          */
  function showResult(prediction, confidence, model) {
    /* 1. Write the values */
    resultValue.textContent  = formatUSD(prediction) + ' per capita';
    confidencePct.textContent = Math.round(confidence * 100) + '%';
    modelUsed.textContent    = 'Model used: ' + model;

    /* 2. Remove hidden so the card enters the layout */
    resultCard.removeAttribute('hidden');

    /* 3. Tiny delay lets the browser register the element before
          transitioning (avoids the "no-transition on first render" bug) */
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        resultCard.classList.add('is-visible');
        /* Animate confidence bar width */
        confidenceFill.style.width = Math.round(confidence * 100) + '%';
      });
    });

    /* 4. Scroll smoothly to the result card */
    setTimeout(() => {
      const navbar = document.getElementById('navbar');
      const offset = (navbar ? navbar.offsetHeight : 68) + 16;
      const top    = resultCard.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }, 100);
  }

  /* Hide and reset the result card (e.g. on a new submission) */
  function hideResult() {
    resultCard.classList.remove('is-visible');
    confidenceFill.style.width = '0%';
    setTimeout(() => resultCard.setAttribute('hidden', ''), 400);
  }

  /* ── Main: form submit handler ──────────────────────────── */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();
    hideResult();

    /* Validate */
    const data = collectFormData();
    if (!data) {
      showError('Please fill in all fields with valid numbers.');
      return;
    }

    /* Show loading state */
    setLoading(true);

    try {
      // Append model_name query param unless user chose 'best'
      const url = selectedModel && selectedModel !== 'best'
        ? `${API_BASE}/predict/gdp?model_name=${encodeURIComponent(selectedModel)}`
        : `${API_BASE}/predict/gdp`;

      const response = await fetch(url, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(data),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      /* FastAPI returns: { predicted_value, confidence, model_used, input_features } */
      showResult(result.predicted_value, result.confidence, result.model_used);

    } catch (err) {
      /* Network error or API down — show a friendly message */
      showError('Could not reach the API. Is the FastAPI server running on port 8000?');
      console.error('[GDP Predictor]', err);
    } finally {
      setLoading(false);
    }
  });

  /* Clear .is-invalid highlight as soon as the user starts typing */
  FIELD_NAMES.forEach(name => {
    const el = document.getElementById(name);
    if (el) el.addEventListener('input', () => el.classList.remove('is-invalid'));
  });

  /* ── Compare All regressors ─────────────────────────────── */
  if (gdpCompareBtn) {
    gdpCompareBtn.addEventListener('click', async () => {
      const data = collectFormData();
      if (!data) { showError('Fill in all fields first, then click Compare All.'); return; }
      clearError();

      // Show table with pending rows immediately
      if (gdpCompareWrap) gdpCompareWrap.removeAttribute('hidden');
      if (gdpCompareTbody) {
        gdpCompareTbody.innerHTML = ALL_REGRESSORS.map(m =>
          `<tr id="gtr-${m.key}">
            <td>${m.label}</td>
            <td style="color:var(--color-text-faint)">…</td>
            <td style="color:var(--color-text-faint)">Pending</td>
          </tr>`
        ).join('');
      }

      // Fire all 6 requests in parallel
      const requests = ALL_REGRESSORS.map(m =>
        fetch(`${API_BASE}/predict/gdp?model_name=${encodeURIComponent(m.key)}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(data),
        })
        .then(r => r.json())
        .then(json => ({ model: m, result: json, ok: true }))
        .catch(err => ({ model: m, error: err, ok: false }))
      );

      const results = await Promise.allSettled(requests);

      results.forEach(settled => {
        const item = settled.value || settled.reason;
        const row  = document.getElementById(`gtr-${item.model.key}`);
        if (!row) return;

        if (!item.ok || !item.result || item.result.predicted_value == null) {
          row.cells[1].textContent = 'Error';
          row.cells[1].style.color = '#f87171';
          row.cells[2].textContent = 'API error';
          return;
        }

        const val = formatUSD(item.result.predicted_value);
        row.cells[1].innerHTML =
          `<strong style="color:var(--color-text)">${val}</strong> <span style="color:var(--color-text-faint);font-size:0.75rem">per capita</span>`;
        row.cells[2].textContent = item.result.model_used || item.model.label;
      });

      setTimeout(() => {
        if (gdpCompareWrap) gdpCompareWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    });
  }

})();


/* ============================================================
   GDP CATEGORY CLASSIFIER — Form submit → API → Badge card
   ============================================================ */
(function () {
  'use strict';

  /* ── Element references ─────────────────────────────────── */
  const form         = document.getElementById('classifyForm');
  const submitBtn    = document.getElementById('classifySubmitBtn');
  const errorEl      = document.getElementById('classifyFormError');
  const resultCard   = document.getElementById('classifyResult');
  const badgeEl      = document.getElementById('categoryBadge');
  const emojiEl      = document.getElementById('categoryEmoji');
  const labelEl      = document.getElementById('categoryLabel');
  const rangeEl      = document.getElementById('categoryRange');
  const explanEl     = document.getElementById('categoryExplanation');
  const modelEl      = document.getElementById('classifyModelUsed');
  const autofillBtn  = document.getElementById('classifyAutofillBtn');
  const pillsContainer = document.getElementById('classifyModelPills');
  const compareAllBtn  = document.getElementById('compareAllBtn');
  const compareWrap    = document.getElementById('compareTableWrap');
  const compareTbody   = document.getElementById('compareTableBody');

  if (!form) return;

  const API_BASE = 'http://localhost:8000';

  /* Field names — same 18 as the GDP predictor, but IDs are
     prefixed with "c-" to avoid colliding with predictor inputs */
  const FIELD_NAMES = [
    'population', 'area', 'pop_density', 'coastline',
    'net_migration', 'infant_mortality', 'gdp_per_capita',
    'literacy', 'phones_per_1000', 'arable', 'crops', 'other',
    'climate', 'birthrate', 'deathrate',
    'agriculture', 'industry', 'service'
  ];
  const REGION_ID = 'c-region';  // string field handled separately

  /* All 7 classifiers (key matches API model_name param) */
  const ALL_MODELS = [
    { key: 'random_forest',       label: 'Random Forest'       },
    { key: 'logistic_regression', label: 'Logistic Regression' },
    { key: 'knn',                 label: 'KNN'                 },
    { key: 'xgboost',             label: 'XGBoost'             },
    { key: 'svm',                 label: 'SVM'                 },
    { key: 'gaussian_naive_bayes',label: 'Naive Bayes'         },
    { key: 'mlp',                 label: 'MLP'                 },
  ];

  /* Currently selected model key — 'best' means use API default */
  let selectedModel = 'best';

  /* ── Pill selector ──────────────────────────────────────── */
  if (pillsContainer) {
    pillsContainer.addEventListener('click', (e) => {
      const pill = e.target.closest('.model-pill');
      if (!pill) return;
      pillsContainer.querySelectorAll('.model-pill').forEach(p =>
        p.classList.remove('model-pill--active')
      );
      pill.classList.add('model-pill--active');
      selectedModel = pill.dataset.model;
    });
  }

  /* ── Category metadata ──────────────────────────────────── */
  /* Each category the API can return maps to a colour theme,
     badge text, GDP range label, and an explanation.          */
  const CATEGORIES = {
    low: {
      emoji      : '🔴',
      label      : 'Low GDP Economy',
      range      : '< $5,000 per capita',
      explanation: 'Countries in the low-GDP tier often face significant '
                 + 'developmental challenges, including limited industrial '
                 + 'capacity, high dependence on subsistence agriculture, '
                 + 'and constrained access to education and healthcare.',
      modifier   : 'low',
    },
    medium: {
      emoji      : '🟡',
      label      : 'Medium GDP Economy',
      range      : '$5,000 – $15,000 per capita',
      explanation: 'Middle-income economies are typically transitioning from '
                 + 'agriculture-led growth toward manufacturing and services. '
                 + 'They show improving human development scores but may still '
                 + 'experience income inequality and infrastructure gaps.',
      modifier   : 'medium',
    },
    high: {
      emoji      : '🟢',
      label      : 'High GDP Economy',
      range      : '> $15,000 per capita',
      explanation: 'High-GDP countries typically have advanced infrastructure, '
                 + 'high human development indices, and diversified industrial '
                 + 'and service-based economies with strong institutional '
                 + 'frameworks and technological innovation.',
      modifier   : 'high',
    },
  };

  /* ── Helpers ────────────────────────────────────────────── */

  function collectFormData() {
    const data = {};
    let valid = true;

    FIELD_NAMES.forEach(name => {
      const el  = document.getElementById('c-' + name);
      const val = el ? parseFloat(el.value) : NaN;

      if (el) el.classList.remove('is-invalid');

      if (isNaN(val) || el.value.trim() === '') {
        if (el) el.classList.add('is-invalid');
        valid = false;
      } else {
        data[name] = val;
      }
    });

    // region string field
    const regionEl = document.getElementById(REGION_ID);
    if (regionEl) regionEl.classList.remove('is-invalid');
    if (!regionEl || regionEl.value.trim() === '') {
      if (regionEl) regionEl.classList.add('is-invalid');
      valid = false;
    } else {
      data['region'] = regionEl.value.trim();
    }

    return valid ? data : null;
  }

  function setLoading(isLoading) {
    if (isLoading) {
      submitBtn.classList.add('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Classifying…';
    } else {
      submitBtn.classList.remove('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Classify Economy';
    }
  }

  function showError(msg) { errorEl.textContent = msg; }
  function clearError()   { errorEl.textContent = ''; }

  /* Resolve a raw category string from the API (e.g. "Low", "Medium GDP",
     "high income") to one of our three keys: 'low' | 'medium' | 'high'  */
  function resolveCategory(raw) {
    const s = (raw || '').toLowerCase();
    if (s.includes('low'))    return 'low';
    if (s.includes('medium') || s.includes('mid')) return 'medium';
    if (s.includes('high'))   return 'high';
    return null;
  }

  function showResult(categoryKey, modelName) {
    const meta = CATEGORIES[categoryKey];
    if (!meta) { showError('Unrecognised category: ' + categoryKey); return; }

    /* Write content */
    emojiEl.textContent  = meta.emoji;
    labelEl.textContent  = meta.label;
    rangeEl.textContent  = meta.range;
    explanEl.textContent = meta.explanation;
    modelEl.textContent  = 'Model used: ' + (modelName || 'Random Forest Classifier');

    /* Reset colour classes, apply the right one */
    resultCard.classList.remove('category-card--low', 'category-card--medium', 'category-card--high');
    badgeEl.classList.remove('category-badge--low', 'category-badge--medium', 'category-badge--high');
    resultCard.classList.add('category-card--' + meta.modifier);
    badgeEl.classList.add('category-badge--' + meta.modifier);

    /* Reveal card */
    resultCard.removeAttribute('hidden');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resultCard.classList.add('is-visible'));
    });

    /* Smooth scroll to result */
    setTimeout(() => {
      const navbar = document.getElementById('navbar');
      const offset = (navbar ? navbar.offsetHeight : 68) + 16;
      const top    = resultCard.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }, 100);
  }

  function hideResult() {
    resultCard.classList.remove('is-visible');
    setTimeout(() => resultCard.setAttribute('hidden', ''), 400);
  }

  /* ── Autofill button ────────────────────────────────────── */
  /* Copies values from the GDP predictor form (ids without "c-")
     into the classifier form (ids with "c-").                  */
  if (autofillBtn) {
    autofillBtn.addEventListener('click', () => {
      let copied = 0;
      FIELD_NAMES.forEach(name => {
        const src  = document.getElementById(name);          // predictor field
        const dest = document.getElementById('c-' + name);  // classifier field
        if (src && dest && src.value.trim() !== '') {
          dest.value = src.value;
          dest.classList.remove('is-invalid');
          copied++;
        }
      });
      // Also copy the region dropdown
      const srcRegion  = document.getElementById('region');
      const destRegion = document.getElementById('c-region');
      if (srcRegion && destRegion && srcRegion.value) {
        destRegion.value = srcRegion.value;
        destRegion.classList.remove('is-invalid');
        copied++;
      }
      if (copied === 0) {
        autofillBtn.textContent = 'Predictor form is empty ↑';
        setTimeout(() => { autofillBtn.textContent = 'Copy values from Predictor ↑'; }, 2500);
      } else {
        autofillBtn.textContent = `✓ Copied ${copied} values`;
        setTimeout(() => { autofillBtn.textContent = 'Copy values from Predictor ↑'; }, 2000);
      }
    });
  }

  /* ── Form submit ────────────────────────────────────────── */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();
    hideResult();

    const data = collectFormData();
    if (!data) { showError('Please fill in all fields with valid numbers.'); return; }

    setLoading(true);

    try {
      // Build URL — append model_name query param unless user chose 'best'
      const url = selectedModel && selectedModel !== 'best'
        ? `${API_BASE}/predict/gdp-category?model_name=${encodeURIComponent(selectedModel)}`
        : `${API_BASE}/predict/gdp-category`;

      const response = await fetch(url, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(data),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      /* FastAPI /predict/gdp-category returns { predicted_category, model_used, ... } */
      const key = resolveCategory(result.predicted_category);
      if (!key) throw new Error('Unexpected category value: ' + result.predicted_category);
      showResult(key, result.model_used);

    } catch (err) {
      showError('Could not reach the API. Is the FastAPI server running on port 8000?');
      console.error('[Classifier]', err);
    } finally {
      setLoading(false);
    }
  });

  /* Clear invalid state on input */
  FIELD_NAMES.forEach(name => {
    const el = document.getElementById('c-' + name);
    if (el) el.addEventListener('input', () => el.classList.remove('is-invalid'));
  });

  /* ── Compare All button ─────────────────────────────────── */
  /* Calls all 7 classifiers in parallel with Promise.allSettled
     and renders a comparison table row for each result.       */
  if (compareAllBtn) {
    compareAllBtn.addEventListener('click', async () => {
      const data = collectFormData();
      if (!data) { showError('Fill in all fields first, then click Compare All.'); return; }
      clearError();

      // Show table with loading rows immediately
      if (compareWrap) compareWrap.removeAttribute('hidden');
      if (compareTbody) {
        compareTbody.innerHTML = ALL_MODELS.map(m =>
          `<tr id="ctr-${m.key}">
            <td>${m.label}</td>
            <td><span class="ct-badge ct-badge--loading">…</span></td>
            <td style="color:var(--color-text-faint)">Pending</td>
          </tr>`
        ).join('');
      }

      // Fire all requests in parallel
      const requests = ALL_MODELS.map(m =>
        fetch(`${API_BASE}/predict/gdp-category?model_name=${encodeURIComponent(m.key)}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(data),
        })
        .then(r => r.json())
        .then(json => ({ model: m, result: json, ok: true }))
        .catch(err => ({ model: m, error: err, ok: false }))
      );

      const results = await Promise.allSettled(requests);

      results.forEach(settled => {
        const item   = settled.value || settled.reason;
        const row    = document.getElementById(`ctr-${item.model.key}`);
        if (!row) return;

        if (!item.ok || !item.result) {
          row.cells[1].innerHTML = '<span class="ct-badge ct-badge--error">Error</span>';
          row.cells[2].textContent = 'API error';
          return;
        }

        const cat = (item.result.predicted_category || '').toLowerCase();
        const tier = cat.includes('low') ? 'low' : cat.includes('medium') ? 'medium' : cat.includes('high') ? 'high' : 'loading';
        const emoji = tier === 'low' ? '🔴' : tier === 'medium' ? '🟡' : '🟢';

        row.cells[1].innerHTML =
          `<span class="ct-badge ct-badge--${tier}">${emoji} ${item.result.predicted_category || '?'}</span>`;
        row.cells[2].textContent = item.result.model_used || item.model.label;
      });

      // Scroll to table
      setTimeout(() => {
        if (compareWrap) compareWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    });
  }

})();


/* ============================================================
   COUNTRY CLUSTER ANALYSIS — KMeans via /analyze/clusters
   ============================================================ */
(function () {
  'use strict';

  const API_BASE = 'http://localhost:8000';

  /* ── Element references ────────────────────────────────── */
  const form         = document.getElementById('clusterForm');
  const submitBtn    = document.getElementById('clusterSubmitBtn');
  const errorEl      = document.getElementById('clusterFormError');
  const resultCard   = document.getElementById('clusterResult');
  const clusterIdEl  = document.getElementById('clusterIdEl');
  const clusterNameEl= document.getElementById('clusterNameEl');
  const clusterDescEl= document.getElementById('clusterDescEl');
  const clusterSilEl = document.getElementById('clusterSilhouetteEl');
  const clusterModEl = document.getElementById('clusterModelEl');
  const autofillBtn  = document.getElementById('clusterAutofillBtn');
  const clusterPillsContainer = document.getElementById('clusterModelPills');
  const clusterCompareBtn   = document.getElementById('clusterCompareAllBtn');
  const clusterCompareWrap  = document.getElementById('clusterCompareTableWrap');
  const clusterCompareTbody = document.getElementById('clusterCompareTableBody');

  /* Stats bar elements */
  const csClusterEl   = document.querySelector('#cs-clusters .cluster-stat-item__value');
  const csSilEl       = document.querySelector('#cs-silhouette .cluster-stat-item__value');
  const csDaviesEl    = document.querySelector('#cs-davies .cluster-stat-item__value');

  if (!form) return;

  /* Field names — prefixed with "k-" in HTML */
  const FIELD_NAMES = [
    'population', 'area', 'pop_density', 'coastline',
    'net_migration', 'infant_mortality', 'gdp_per_capita',
    'literacy', 'phones_per_1000', 'arable', 'crops', 'other',
    'climate', 'birthrate', 'deathrate',
    'agriculture', 'industry', 'service'
  ];
  const REGION_ID = 'k-region';

  /* All clustering models */
  const ALL_CLUSTERING_MODELS = [
    { key: 'kmeans',       label: 'KMeans'       },
    { key: 'hierarchical', label: 'Hierarchical' },
    { key: 'dbscan',       label: 'DBSCAN'       },
  ];

  /* Currently selected clustering model */
  let selectedClusterModel = 'kmeans';

  /* ── Pill selector ──────────────────────────────────────── */
  if (clusterPillsContainer) {
    clusterPillsContainer.addEventListener('click', (e) => {
      const pill = e.target.closest('.model-pill');
      if (!pill) return;
      clusterPillsContainer.querySelectorAll('.model-pill').forEach(p =>
        p.classList.remove('model-pill--active')
      );
      pill.classList.add('model-pill--active');
      selectedClusterModel = pill.dataset.model;
    });
  }

  /* ── Load cluster summary on page load ─────────────────── */
  async function loadClusterSummary() {
    try {
      const res = await fetch(`${API_BASE}/analyze/clusters/summary`);
      if (!res.ok) return;  // silently skip if API not running
      const data = await res.json();

      if (csClusterEl)  csClusterEl.textContent  = data.n_clusters ?? '—';
      if (csSilEl)      csSilEl.textContent       = data.silhouette_score != null
                                                    ? data.silhouette_score.toFixed(3) : '—';
      if (csDaviesEl)   csDaviesEl.textContent    = data.davies_bouldin_score != null
                                                    ? data.davies_bouldin_score.toFixed(3) : '—';
    } catch (_) { /* API not running — leave dashes */ }
  }

  loadClusterSummary();

  /* ── Helpers ────────────────────────────────────────────── */
  function collectFormData() {
    const data = {};
    let valid = true;

    FIELD_NAMES.forEach(name => {
      const el  = document.getElementById('k-' + name);
      const val = el ? parseFloat(el.value) : NaN;
      if (el) el.classList.remove('is-invalid');
      if (isNaN(val) || !el || el.value.trim() === '') {
        if (el) el.classList.add('is-invalid');
        valid = false;
      } else {
        data[name] = val;
      }
    });

    const regionEl = document.getElementById(REGION_ID);
    if (regionEl) regionEl.classList.remove('is-invalid');
    if (!regionEl || !regionEl.value) {
      if (regionEl) regionEl.classList.add('is-invalid');
      valid = false;
    } else {
      data['region'] = regionEl.value;
    }

    return valid ? data : null;
  }

  function setLoading(on) {
    if (on) {
      submitBtn.classList.add('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Analysing…';
    } else {
      submitBtn.classList.remove('is-loading');
      submitBtn.querySelector('.btn-label').textContent = 'Find My Cluster';
    }
  }

  function showError(msg) { errorEl.textContent = msg; }
  function clearError()   { errorEl.textContent = ''; }

  function showResult(data) {
    /* data: { cluster_assignment, cluster_name, model_used, silhouette_score } */
    clusterIdEl.textContent   = data.cluster_assignment ?? '?';
    clusterNameEl.textContent = data.cluster_name       || 'Unknown Cluster';
    clusterDescEl.textContent = getClusterDesc(data.cluster_assignment, data.cluster_name);
    clusterSilEl.textContent  = data.silhouette_score != null
      ? 'Silhouette: ' + data.silhouette_score.toFixed(3) : 'Silhouette: N/A';
    clusterModEl.textContent  = 'Model: ' + (data.model_used || 'KMeans');

    resultCard.removeAttribute('hidden');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resultCard.classList.add('is-visible'));
    });

    setTimeout(() => {
      const navbar = document.getElementById('navbar');
      const offset = (navbar ? navbar.offsetHeight : 68) + 16;
      const top = resultCard.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }, 100);
  }

  function hideResult() {
    resultCard.classList.remove('is-visible');
    setTimeout(() => resultCard.setAttribute('hidden', ''), 400);
  }

  /* Human-readable description for each cluster */
  function getClusterDesc(id, name) {
    const n = (name || '').toLowerCase();
    if (n.includes('develop') && !n.includes('emerging'))
      return 'Developing economies typically have high population growth, '
           + 'lower literacy rates, higher infant mortality, and rely '
           + 'heavily on agriculture. They represent significant growth potential.';
    if (n.includes('emerg') || n.includes('market'))
      return 'Emerging market economies are in active transition, showing '
           + 'rising industrialisation, improving infrastructure, and '
           + 'growing middle-class consumption alongside persistent inequality.';
    if (n.includes('develop') || n.includes('nation') || n.includes('rich'))
      return 'Developed nations feature high GDP per capita, advanced '
           + 'infrastructure, high literacy, diversified service economies, '
           + 'and strong institutional frameworks.';
    return 'This cluster groups countries with similar socioeconomic profiles '
         + 'across indicators such as GDP, literacy, birthrate, and land use.';
  }

  /* ── Autofill from predictor ───────────────────────────── */
  if (autofillBtn) {
    autofillBtn.addEventListener('click', () => {
      let copied = 0;
      FIELD_NAMES.forEach(name => {
        const src  = document.getElementById(name);
        const dest = document.getElementById('k-' + name);
        if (src && dest && src.value.trim() !== '') {
          dest.value = src.value;
          dest.classList.remove('is-invalid');
          copied++;
        }
      });
      const srcReg  = document.getElementById('region');
      const destReg = document.getElementById('k-region');
      if (srcReg && destReg && srcReg.value) {
        destReg.value = srcReg.value;
        destReg.classList.remove('is-invalid');
        copied++;
      }
      autofillBtn.textContent = copied
        ? `✓ Copied ${copied} values`
        : 'Predictor form is empty ↑';
      setTimeout(() => { autofillBtn.textContent = 'Copy values from Predictor ↑'; }, 2000);
    });
  }

  /* ── Form submit ───────────────────────────────────────── */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError();
    hideResult();

    const data = collectFormData();
    if (!data) { showError('Please fill in all fields with valid values.'); return; }

    setLoading(true);
    try {
      // Append model_name query param
      const url = `${API_BASE}/analyze/clusters?model_name=${encodeURIComponent(selectedClusterModel)}`;

      const res = await fetch(url, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(data),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${res.status}`);
      }

      const result = await res.json();
      /* ClusteringResponse: { cluster_assignment, cluster_name, model_used,
                               silhouette_score, input_features }            */
      showResult(result);

    } catch (err) {
      showError('Could not reach the API. Is the FastAPI server running on port 8000?');
      console.error('[Clustering]', err);
    } finally {
      setLoading(false);
    }
  });

  /* Clear invalid state on input */
  FIELD_NAMES.forEach(name => {
    const el = document.getElementById('k-' + name);
    if (el) el.addEventListener('input', () => el.classList.remove('is-invalid'));
  });

  /* ── Compare All clustering models ──────────────────────── */
  if (clusterCompareBtn) {
    clusterCompareBtn.addEventListener('click', async () => {
      const data = collectFormData();
      if (!data) { showError('Fill in all fields first, then click Compare All.'); return; }
      clearError();

      // Show table with pending rows immediately
      if (clusterCompareWrap) clusterCompareWrap.removeAttribute('hidden');
      if (clusterCompareTbody) {
        clusterCompareTbody.innerHTML = ALL_CLUSTERING_MODELS.map(m =>
          `<tr id="ctr-${m.key}">
            <td>${m.label}</td>
            <td style="color:var(--color-text-faint)">…</td>
            <td style="color:var(--color-text-faint)">Pending</td>
          </tr>`
        ).join('');
      }

      // Fire all requests in parallel
      const requests = ALL_CLUSTERING_MODELS.map(m =>
        fetch(`${API_BASE}/analyze/clusters?model_name=${encodeURIComponent(m.key)}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(data),
        })
        .then(async r => {
          const json = await r.json().catch(() => ({}));
          if (!r.ok) throw json;
          return json;
        })
        .then(json => ({ model: m, result: json, ok: true }))
        .catch(err => ({ model: m, error: err, ok: false }))
      );

      const results = await Promise.allSettled(requests);

      results.forEach(settled => {
        const item = settled.value || settled.reason;
        const row  = document.getElementById(`ctr-${item.model.key}`);
        if (!row) return;

        if (!item.ok || !item.result || item.result.cluster_assignment == null) {
          // Check if it's the 501 error for predict not supported
          let errText = 'API error';
          if (item.error && item.error.detail) errText = 'Not supported for new points';
          row.cells[1].textContent = 'Error';
          row.cells[1].style.color = '#f87171';
          row.cells[2].textContent = errText;
          return;
        }

        const clusterId = item.result.cluster_assignment;
        const silScore = item.result.silhouette_score != null ? item.result.silhouette_score.toFixed(3) : 'N/A';
        row.cells[1].innerHTML =
          `<strong style="color:var(--color-text)">Cluster ${clusterId}</strong> <span style="color:var(--color-text-faint);font-size:0.75rem">(${item.result.cluster_name})</span>`;
        row.cells[2].textContent = silScore;
      });

      setTimeout(() => {
        if (clusterCompareWrap) clusterCompareWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    });
  }

})();
