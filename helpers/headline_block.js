// Returns the page rectangle of an article's headline block: kicker line,
// headline, subtitle, byline and date, measured from TEXT boxes and stopping
// at the lead picture. Used by screenshot.capture_headline_block.
new Promise(done => {
  document.documentElement.style.setProperty('overflow', 'auto', 'important');
  document.body.style.setProperty('overflow', 'auto', 'important');
  for (const el of document.querySelectorAll('body *')) {
    const p = getComputedStyle(el).position;
    if (p === 'fixed' || p === 'sticky') el.style.setProperty('display', 'none', 'important');
  }
  const visible = el => { const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 40 && r.height > 10 && s.visibility !== 'hidden' && s.opacity !== '0'; };
  const size = el => parseFloat(getComputedStyle(el).fontSize) || 0;
  // The headline is the biggest visible h1 (a hidden modal or logo h1 comes first
  // in the DOM on Substack and Fortune), else the biggest h2.
  let hs = [...document.querySelectorAll('h1')].filter(visible);
  if (!hs.length) hs = [...document.querySelectorAll('h2')].filter(visible);
  const h = hs.sort((a, b) => size(b) - size(a))[0];
  if (h) {
    h.scrollIntoView({block: 'center'});
    h.setAttribute('data-zeta', '1');
    // The headline's TEXT box: its element often spans the whole page width,
    // which made everything above it look like part of its column (CNN).
    const rng = document.createRange(); rng.selectNodeContents(h);
    const hb = rng.getBoundingClientRect(), hs0 = size(h);
    let kicker = null, kickerBottom = -Infinity;
    // The headline's own column: a GitHub sidebar or a related-links rail sits
    // beside it at the same height and must not widen the crop.
    const inCol = r => r.left >= hb.left - 30 && r.right <= hb.right + 60;
    const leafText = el => {
      if (el.closest('button, nav, aside, header nav, [role=button], audio, figure')) return false;
      if (el.querySelector('img, svg, button, video, audio, picture, h1')) return false;
      const t = (el.innerText || '').trim();
      return t.length > 0 && t.length < 280 && [...el.children].every(c =>
        ['A', 'SPAN', 'B', 'STRONG', 'EM', 'TIME', 'I'].includes(c.tagName));
    };
    // Nothing under the lead picture belongs to the headline block.
    const pic = [...document.querySelectorAll('img, figure, video, picture, iframe')]
      .map(e => e.getBoundingClientRect()).filter(r => r.height > 80 && r.top > hb.bottom)
      .map(r => r.top).sort((a, b) => a - b)[0] || Infinity;
    for (const el of document.querySelectorAll('p, span, a, div, time, h2, h3, h4')) {
      if (el === h || h.contains(el) || !visible(el) || !leafText(el) || size(el) >= hs0) continue;
      const r = el.getBoundingClientRect();
      if (!inCol(r)) continue;
      // A kicker sits right above the headline and inside its width (CNN's
      // market ticker sits above too, but wider than the headline column).
      const above = r.bottom <= hb.top + 2 && r.bottom > hb.top - 45 && r.height < 40 &&
                    r.right <= hb.right + 30;
      const below = r.top >= hb.bottom - 2 && r.bottom < Math.min(hb.bottom + 150, pic);
      if (below) el.setAttribute('data-zeta', '1');
      // Only the single nearest line above counts as the kicker.
      if (above && r.bottom > kickerBottom) { kicker = el; kickerBottom = r.bottom; }
    }
    if (kicker) kicker.setAttribute('data-zeta', '1');
  }
  // Union of the TEXT boxes of everything marked: element boxes are often
  // full-width containers that drag in tickers and photos.
  // Measured LAST, after the page settles: a late ad above the article
  // (TechCrunch) shifted the headline between measuring and shooting.
  setTimeout(() => {
    const boxes = [...document.querySelectorAll('[data-zeta]')].map(e => {
      const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect();
    }).filter(r => r.width > 0 && r.height > 0);
    if (!boxes.length) return done(null);
    const x0 = Math.min(...boxes.map(b => b.left)), y0 = Math.min(...boxes.map(b => b.top));
    const x1 = Math.max(...boxes.map(b => b.right)), y1 = Math.max(...boxes.map(b => b.bottom));
    done({x: x0 + scrollX, y: y0 + scrollY, w: x1 - x0, h: y1 - y0});
  }, 1200);
})
