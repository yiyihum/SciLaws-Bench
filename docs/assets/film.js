// SciLaws-Bench animated overview: ten SVG scenes driven by one clock.
(() => {
  // ---------- helpers ----------
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const d = s => `style="--d:${s}s"`;
  const dt = (s, t) => `style="--d:${s}s;--t:${t}s"`;
  const path = (pts) => 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
  const curve = (f, x0, x1, n = 120) => Array.from({ length: n + 1 }, (_, i) => { const x = x0 + (x1 - x0) * i / n; return [x, f(x)]; });
  const axes = (x, y, w, h) => `<path class="axis" d="M${x} ${y} V${y + h} H${x + w}"/>`;
  const svg = inner => `<svg viewBox="0 0 1600 900" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;
  const eyebrow = (txt, cls = 'mute') => `<text x="110" y="120" font-size="26" letter-spacing="6" class="${cls} fade">${txt}</text>`;

  // ---------- scenes ----------
  const scenes = [];

  // 1. Kepler
  (() => {
    const planets = [['Mercury', .387, .241], ['Venus', .723, .615], ['Earth', 1, 1], ['Mars', 1.524, 1.881], ['Jupiter', 5.203, 11.86], ['Saturn', 9.537, 29.46]];
    const X = la => 160 + (la + 0.5) / 1.6 * 820, Y = lt => 780 - (lt + 0.75) / 2.35 * 600;
    let g = axes(160, 180, 840, 600);
    g += `<text x="580" y="840" text-anchor="middle" font-size="24" class="mute fade">semi-major axis a (AU, log scale)</text>`;
    g += `<text x="110" y="480" text-anchor="middle" font-size="24" class="mute fade" transform="rotate(-90 110 480)">orbital period T (years, log scale)</text>`;
    g += `<path class="line draw" ${dt(3.2, 2)} pathLength="1" stroke="var(--real)" d="${path([[X(-0.5), Y(-0.75)], [X(1.1), Y(1.65)]])}"/>`;
    planets.forEach(([n, a, t], i) => {
      const x = X(Math.log10(a)), y = Y(Math.log10(t));
      g += `<g class="pop" ${d(0.4 + i * 0.35)}><circle cx="${x}" cy="${y}" r="11" fill="var(--chalk)"/></g>`;
      g += `<text x="${x + 20}" y="${y + 36}" font-size="22" class="mute fade" ${d(0.6 + i * 0.35)}>${n}</text>`;
    });
    g += `<text x="1090" y="380" font-size="130" class="disp real rise" ${d(4.6)}>T² ∝ a³</text>`;
    g += `<text x="1095" y="440" font-size="28" class="mute rise" ${d(5)}>Kepler’s third law · 1619</text>`;
    g += `<text x="1095" y="560" font-size="56" class="disp rise" ${d(6)}>v = H₀ d</text>`;
    g += `<text x="1095" y="605" font-size="26" class="mute rise" ${d(6.3)}>Hubble’s law · 1929</text>`;
    scenes.push({ name: 'Laws', dur: 9.5, svg: svg(g),
      cap: 'From Kepler to Hubble, scientific laws compress <b>imperfect measurements</b> into compact mathematics.' });
  })();

  // 2. Title
  (() => {
    let g = '';
    for (let i = 0; i < 60; i++) {
      const x = 80 + rnd() * 1440, y = 80 + rnd() * 740, c = rnd() < .5 ? 'var(--real)' : 'var(--par)';
      g += `<circle class="fade" ${d(rnd() * 2)} cx="${x}" cy="${y}" r="${2 + rnd() * 3}" fill="${c}" opacity=".5"/>`;
    }
    g += `<text x="800" y="400" text-anchor="middle" font-size="170" class="disp rise" ${d(0.3)}>SciLaws<tspan class="real">-</tspan>Bench</text>`;
    g += `<text x="800" y="490" text-anchor="middle" font-size="40" class="disp mute rise" font-style="italic" ${d(1)}>Can LLMs Discover Scientific Laws in Real and Parallel Worlds?</text>`;
    g += `<text x="800" y="580" text-anchor="middle" font-size="38" class="rise" ${d(1.8)}>One benchmark, two settings: <tspan class="real">Real</tspan> and <tspan class="par">Parallel</tspan></text>`;
    scenes.push({ name: 'Question', dur: 7, svg: svg(g),
      cap: 'LLMs are entering scientific research. Can they <b>actually discover</b> scientific laws? And how would we know?' });
  })();

  // 3. Dilemma
  (() => {
    let g = eyebrow('THE EVALUATION DILEMMA');
    g += `<g class="rise" ${d(0.4)}>
      <rect x="110" y="190" width="640" height="470" rx="10" fill="none" stroke="var(--chalk)" stroke-opacity=".25"/>
      <text x="150" y="260" font-size="34" font-weight="500">Rediscovering textbook laws</text>
      <text x="150" y="380" font-size="64" class="disp">E = ħω</text>
      <text x="150" y="480" font-size="52" class="disp">θ₁ = arcsin(n sin θ₂)</text>
      <text x="150" y="610" font-size="26" class="mute">A model may simply recall the answer</text></g>`;
    g += `<g class="pop" ${d(1.8)}><g transform="rotate(-10 600 330)">
      <rect x="460" y="290" width="280" height="80" rx="8" fill="none" stroke="var(--rose)" stroke-width="4"/>
      <text x="600" y="345" text-anchor="middle" font-size="40" class="rose" font-weight="700">Recall?</text></g></g>`;
    g += `<g class="rise" ${d(2.6)}>
      <rect x="850" y="190" width="640" height="470" rx="10" fill="none" stroke="var(--chalk)" stroke-opacity=".25"/>
      <text x="890" y="260" font-size="34" font-weight="500">Counterfactual rewrites</text>
      <text x="890" y="380" font-size="58" class="disp">F = G m₁m₂ / r²</text>
      <text x="890" y="470" font-size="58" class="disp par">F = G′ m₁m₂ / r¹·⁵</text>
      <text x="890" y="610" font-size="26" class="mute">Avoids recall, but drifts from real data</text></g>`;
    g += `<text x="800" y="770" text-anchor="middle" font-size="36" class="rise" ${d(5)}>We need tasks that go <tspan class="real">beyond familiar laws</tspan> yet stay <tspan class="real">grounded in real data</tspan></text>`;
    scenes.push({ name: 'Dilemma', dur: 9, svg: svg(g),
      cap: 'Existing tests either ask for textbook laws, which memory can supply, or rewrite them into counterfactuals that leave real science behind.' });
  })();

  // 4. Curation
  (() => {
    let g = eyebrow('BUILDING SCILAWS-BENCH');
    for (let i = 0; i < 14; i++) {
      const y = 200 + (i % 7) * 70, del = (i * 0.28).toFixed(2);
      g += `<g class="flow" ${d(del)}><rect x="110" y="${y}" width="74" height="52" rx="4" fill="var(--ink-2)" stroke="var(--chalk)" stroke-opacity=".5"/>
        <path d="M122 ${y + 14} h48 M122 ${y + 26} h40 M122 ${y + 38} h46" stroke="var(--mute)" stroke-width="3"/></g>`;
    }
    g += `<text x="147" y="740" text-anchor="middle" font-size="22" class="mute fade">papers &amp; datasets</text>`;
    g += `<g class="rise" ${d(0.8)}>
      <rect x="700" y="210" width="380" height="460" rx="12" fill="var(--ink-2)" stroke="var(--real)" stroke-width="2.5"/>
      <text x="890" y="265" text-anchor="middle" font-size="30" class="real" font-weight="500">One task package</text></g>`;
    ['Scientific problem', 'Data: train / held-out', 'Published reference laws', 'Validity rubrics'].forEach((t, i) => {
      g += `<g class="rise" ${d(1.4 + i * .45)}><rect x="735" y="${300 + i * 88}" width="310" height="66" rx="6" fill="none" stroke="var(--chalk)" stroke-opacity=".3"/>
        <text x="760" y="${342 + i * 88}" font-size="25">${t}</text></g>`;
    });
    g += `<text x="890" y="730" text-anchor="middle" font-size="22" class="mute fade" ${d(3.4)}>agent-assisted curation · human verification</text>`;
    [['381', 'papers'], ['118', 'problems'], ['291', 'candidate laws'], ['~8M', 'data points'], ['6', 'disciplines']].forEach(([n, t], i) => {
      g += `<g class="rise" ${d(2.6 + i * .35)}><text x="1180" y="${270 + i * 110}" font-size="78" class="disp real" data-count="${n}">${n}</text>
        <text x="1370" y="${262 + i * 110}" font-size="28">${t}</text></g>`;
    });
    scenes.push({ name: 'Curation', dur: 10, svg: svg(g),
      cap: 'Starting from <b>381 papers</b>, we built <b>118 problems</b> from ongoing research. Each package links the problem, its data, published reference laws, and scientific-validity rubrics.' });
  })();

  // 5. Two worlds
  (() => {
    seed = 11;
    let g = `<line x1="800" y1="90" x2="800" y2="820" stroke="var(--chalk)" stroke-opacity=".2" stroke-dasharray="6 10"/>`;
    // Real
    g += `<text x="110" y="140" font-size="44" class="disp real fade">SciLaws-Real</text>`;
    g += `<text x="110" y="185" font-size="24" class="mute fade" ${d(.3)}>fixed data · open law</text>`;
    g += axes(120, 240, 600, 430);
    const fR = x => 640 - 330 * (1 - Math.exp(-(x - 120) / 190));
    for (let i = 0; i < 46; i++) { const x = 140 + rnd() * 570; g += `<circle class="fade" ${d(.4 + rnd() * .8)} cx="${x}" cy="${fR(x) + gauss() * 22}" r="7" fill="var(--real)" opacity=".85"/>`; }
    g += `<path class="line draw" ${dt(1.8, 1.2)} pathLength="1" stroke="var(--chalk)" stroke-opacity=".45" stroke-dasharray="1" d="${path(curve(x => 640 - (x - 120) * .5, 120, 720))}"/>`;
    g += `<path class="line draw" ${dt(2.6, 1.4)} pathLength="1" stroke="var(--chalk)" d="${path(curve(fR, 120, 720))}"/>`;
    g += `<text x="120" y="740" font-size="25" class="fade" ${d(3.6)}>Scored on held-out <tspan class="real">predictive fit</tspan></text>`;
    g += `<text x="120" y="785" font-size="25" class="fade" ${d(4)}>+ source-grounded <tspan class="real">scientific validity</tspan></text>`;
    // Parallel
    g += `<text x="880" y="140" font-size="44" class="disp par fade" ${d(4.6)}>SciLaws-Parallel</text>`;
    g += `<text x="880" y="185" font-size="24" class="mute fade" ${d(4.9)}>fixed hidden law · the model queries the data</text>`;
    g += `<g class="rise" ${d(5.2)}><rect x="1250" y="250" width="250" height="150" rx="10" fill="var(--ink-2)" stroke="var(--par)" stroke-width="2.5"/>
      <text x="1375" y="315" text-anchor="middle" font-size="28" class="par">Simulator</text>
      <text x="1375" y="355" text-anchor="middle" font-size="22" class="mute">hidden equation</text></g>`;
    g += axes(890, 250, 330, 420);
    const fP = x => 300 + 0.0019 * (x - 890) ** 2 * 0.9 + (x - 890) * .25;
    [960, 1180, 1060, 920, 1120, 1010, 1200, 990].forEach((x, i) => {
      const t = 6 + i * .32;
      g += `<path class="line draw" ${dt(t, .3)} pathLength="1" stroke="var(--par)" stroke-width="2" stroke-opacity=".5" d="M1250 ${330} L${x} ${fP(x)}"/>`;
      g += `<g class="pop" ${d(t + .25)}><circle cx="${x}" cy="${fP(x) + gauss() * 10}" r="8" fill="var(--par)"/></g>`;
    });
    g += `<text x="880" y="740" font-size="22" class="mono fade" ${d(8.8)}>log₁₀Y = f(M) + (b₄+b₅M) log₁₀√(R²+b₆²) <tspan class="par">− b₁₁R</tspan></text>`;
    g += `<text x="880" y="785" font-size="22" class="mute fade" ${d(9.2)}>Akkar &amp; Bommer (2010) ground-motion law + a new anelastic term</text>`;
    scenes.push({ name: 'Two worlds', dur: 11.5, svg: svg(g),
      cap: 'Every problem has two settings. <b>Real</b>: the data is fixed; find a better law. <i>Parallel</i>: a hidden structural variant of a published law; query the simulator and recover it.' });
  })();

  // 6. Agents
  (() => {
    let g = eyebrow('EXPERIMENTS');
    g += `<text x="110" y="230" font-size="120" class="disp real rise" ${d(.3)}>14</text>`;
    g += `<text x="250" y="215" font-size="34" class="rise" ${d(.5)}>frontier LLMs</text>`;
    g += `<text x="110" y="290" font-size="24" class="mute rise" ${d(.8)}>GPT-6 · Claude Opus 5.5 · Gemini 3.5 Flash · Kimi K3 · DeepSeek-V4 Pro …</text>`;
    // ReAct loop
    const cx = 420, cy = 560, r = 160;
    g += `<circle class="draw" ${dt(1.2, 1.5)} pathLength="1" cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--chalk)" stroke-opacity=".35" stroke-width="3"/>`;
    [['Think', -90], ['Code', 30], ['Observe', 150]].forEach(([t, a], i) => {
      const x = cx + r * Math.cos(a * Math.PI / 180), y = cy + r * Math.sin(a * Math.PI / 180);
      g += `<g class="pop" ${d(1.6 + i * .4)}><circle cx="${x}" cy="${y}" r="50" fill="var(--ink-2)" stroke="var(--real)" stroke-width="2.5"/>
        <text x="${x}" y="${y + 9}" text-anchor="middle" font-size="22">${t}</text></g>`;
    });
    g += `<text x="${cx}" y="${cy + 12}" text-anchor="middle" font-size="40" class="mono real fade" ${d(3)}><tspan id="fm-turn">1</tspan>/30</text>`;
    g += `<text x="${cx}" y="${cy + 48}" text-anchor="middle" font-size="20" class="mute fade" ${d(3)}>turns</text>`;
    // styles
    g += `<text x="900" y="400" font-size="30" class="fade" ${d(4.4)}>Two discovery styles</text>`;
    g += `<text x="900" y="470" font-size="26" class="real fade" ${d(4.8)}>Breadth</text><text x="1030" y="470" font-size="22" class="mute fade" ${d(4.8)}>many candidates, few turns</text>`;
    for (let i = 0; i < 18; i++) g += `<g class="pop" ${d(5 + i * .05)}><circle cx="${915 + (i % 9) * 52}" cy="${515 + Math.floor(i / 9) * 46}" r="13" fill="var(--real)" opacity=".8"/></g>`;
    g += `<text x="900" y="670" font-size="26" class="par fade" ${d(6.2)}>Depth</text><text x="1030" y="670" font-size="22" class="mute fade" ${d(6.2)}>few candidates, refined over many turns</text>`;
    for (let i = 0; i < 3; i++) g += `<g class="pop" ${d(6.4 + i * .3)}><circle cx="${935 + i * 150}" cy="735" r="${16 + i * 9}" fill="none" stroke="var(--par)" stroke-width="4"/><circle cx="${935 + i * 150}" cy="735" r="${6 + i * 4}" fill="var(--par)"/></g>`;
    scenes.push({ name: 'Experiments', dur: 9.5, svg: svg(g), turns: true,
      cap: 'Fourteen LLMs from seven families run in the same ReAct agent framework, with a Python sandbox and up to 30 turns per trial. Their trajectories show two styles: <b>breadth</b> and <i>depth</i>.' });
  })();

  // 7. Finding 1
  (() => {
    seed = 23;
    let g = eyebrow('FINDING 1', 'real');
    g += `<text x="110" y="190" font-size="46" class="disp rise" ${d(.2)}>A good fit is not a valid law</text>`;
    g += axes(130, 260, 820, 460);
    const base = x => 640 - 300 / (1 + Math.exp(-(x - 450) / 70));
    const xs = [];
    for (let i = 0; i < 32; i++) { let x = 150 + rnd() * 780; if (x > 590 && x < 700) x -= 140; xs.push(x); }
    xs.forEach(x => g += `<circle class="fade" ${d(.6 + rnd() * .6)} cx="${x}" cy="${base(x) + gauss() * 14}" r="7" fill="var(--chalk)" opacity=".85"/>`);
    g += `<path class="line draw" ${dt(1.6, 2)} pathLength="1" stroke="var(--real)" d="${path(curve(x => base(x) - 120 * Math.exp(-(((x - 645) / 18) ** 2)), 130, 950, 300))}"/>`;
    g += `<g class="fade" ${d(3.8)}><circle cx="645" cy="${base(645) - 120}" r="38" fill="none" stroke="var(--rose)" stroke-width="3" stroke-dasharray="8 8"/>
      <text x="695" y="${base(645) - 135}" font-size="24" class="rose">a resonance peak that doesn’t exist</text></g>`;
    g += `<text x="140" y="770" font-size="22" class="mute fade" ${d(1)}>illustration: a model’s best-fitting formula on a nuclear-physics task</text>`;
    g += `<g class="rise" ${d(4.6)}><rect x="1060" y="320" width="420" height="250" rx="10" fill="var(--ink-2)" stroke="var(--chalk)" stroke-opacity=".25"/>
      <text x="1095" y="390" font-size="28">Held-out predictive fit</text><text x="1440" y="390" text-anchor="end" font-size="36" class="real">✓</text>
      <line x1="1095" y1="430" x2="1445" y2="430" stroke="var(--chalk)" stroke-opacity=".15"/>
      <text x="1095" y="490" font-size="28">Scientific validity</text><text x="1440" y="492" text-anchor="end" font-size="36" class="rose">✗</text>
      <text x="1095" y="540" font-size="20" class="mute">introduces nonexistent physics</text></g>`;
    scenes.push({ name: 'Finding 1', dur: 8.5, svg: svg(g),
      cap: 'On a nuclear-physics task, a model’s best-fitting formula adds a <b>nonexistent resonance peak</b>. It predicts well, and still fails the validity checks.' });
  })();

  // 8. Finding 2
  (() => {
    let g = eyebrow('FINDING 2', 'real');
    g += `<text x="110" y="190" font-size="46" class="disp rise" ${d(.2)}>Recalling a known law is not discovering new structure</text>`;
    g += `<text x="110" y="320" font-size="26" class="mute fade" ${d(.8)}>The model writes down the published form from memory</text>`;
    g += `<text x="110" y="420" font-size="40" class="mono rise" ${d(1.2)}>log₁₀Y = f(M) + (b₄+b₅M) log₁₀√(R²+b₆²)</text>`;
    g += `<text x="1100" y="420" font-size="56" class="real pop" ${d(2.2)}>✓</text>`;
    g += `<text x="110" y="540" font-size="26" class="mute fade" ${d(3)}>The equation actually hidden in the parallel world</text>`;
    g += `<text x="110" y="640" font-size="40" class="mono rise" ${d(3.4)}>log₁₀Y = f(M) + (b₄+b₅M) log₁₀√(R²+b₆²)</text>`;
    g += `<g class="pop" ${d(4.4)}><rect x="1080" y="590" width="250" height="80" rx="8" fill="none" stroke="var(--par)" stroke-width="3" stroke-dasharray="10 8"/>
      <text x="1205" y="640" text-anchor="middle" font-size="44" class="mono par">− b₁₁R</text></g>`;
    g += `<text x="1360" y="645" font-size="56" class="rose pop" ${d(5.2)}>?</text>`;
    g += `<text x="110" y="770" font-size="28" class="fade" ${d(6)}>Beating the reference is more common where <tspan class="real">recall offers little help</tspan></text>`;
    scenes.push({ name: 'Finding 2', dur: 9, svg: svg(g),
      cap: 'Models use memory to reproduce familiar laws, but recovering the <i>new structural terms</i> is much harder. Full-law recovery varies widely across models.' });
  })();

  // 9. Finding 3
  (() => {
    seed = 41;
    let g = eyebrow('FINDING 3', 'real');
    g += `<text x="110" y="190" font-size="46" class="disp rise" ${d(.2)}>Models find better laws than they submit</text>`;
    g += axes(130, 250, 1000, 470);
    g += `<text x="630" y="780" text-anchor="middle" font-size="22" class="mute fade">turns within a trajectory →</text>`;
    g += `<text x="90" y="485" text-anchor="middle" font-size="22" class="mute fade" transform="rotate(-90 90 485)">candidate law quality →</text>`;
    const pts = [];
    for (let i = 0; i < 26; i++) { const x = 170 + i * 36, y = 650 - 260 * (1 - Math.exp(-i / 8)) + gauss() * 45; pts.push([x, Math.max(280, Math.min(700, y))]); }
    let best = 0; pts.forEach((p, i) => { if (p[1] < pts[best][1]) best = i; });
    pts[25][1] = Math.max(pts[25][1], pts[best][1] + 150);
    pts.forEach((p, i) => g += `<g class="pop" ${d(.6 + i * .1)}><circle cx="${p[0]}" cy="${p[1]}" r="9" fill="var(--chalk)" opacity=".7"/></g>`);
    const B = pts[best], S = pts[25];
    g += `<g class="pop" ${d(3.6)}><circle cx="${B[0]}" cy="${B[1]}" r="17" fill="var(--par)"/></g>`;
    g += `<circle class="pulse" ${d(4)} cx="${B[0]}" cy="${B[1]}" r="28" fill="none" stroke="var(--par)" stroke-width="2"/>`;
    g += `<text x="${B[0]}" y="${B[1] - 44}" text-anchor="middle" font-size="24" class="par fade" ${d(3.8)}>best candidate in the pool</text>`;
    g += `<g class="pop" ${d(4.6)}><circle cx="${S[0]}" cy="${S[1]}" r="24" fill="none" stroke="var(--rose)" stroke-width="4"/></g>`;
    g += `<text x="${S[0] + 40}" y="${S[1] + 8}" font-size="24" class="rose fade" ${d(4.8)}>submitted</text>`;
    g += `<path class="line draw" ${dt(5.4, .8)} pathLength="1" stroke="var(--rose)" stroke-width="2.5" stroke-dasharray="1" d="M1180 ${B[1]} H1210 V${S[1]} H1180"/>`;
    g += `<text x="1230" y="${(B[1] + S[1]) / 2 + 10}" font-size="28" class="rose fade" ${d(5.8)}>selection gap</text>`;
    scenes.push({ name: 'Finding 3', dur: 9.5, svg: svg(g),
      cap: 'Scoring every intermediate candidate shows models often find a better-fitting law than the one they submit. In best-of-N search, <b>self-selection</b> captures little of the available gain.' });
  })();

  // 10. Ending
  (() => {
    let g = '';
    g += `<text x="800" y="330" text-anchor="middle" font-size="64" class="disp rise" ${d(.3)}>LLMs can discover <tspan class="real">some</tspan> scientific laws,</text>`;
    g += `<text x="800" y="420" text-anchor="middle" font-size="64" class="disp rise" ${d(1)}>but not yet <tspan class="rose">reliably</tspan>.</text>`;
    g += `<text x="800" y="520" text-anchor="middle" font-size="30" class="mute rise" ${d(2)}>The bottleneck: self-evaluation and candidate selection</text>`;
    g += `<line x1="640" y1="600" x2="960" y2="600" stroke="var(--chalk)" stroke-opacity=".25" class="fade" ${d(2.8)}/>`;
    g += `<text x="800" y="680" text-anchor="middle" font-size="60" class="disp fade" ${d(3)}>SciLaws<tspan class="real">-</tspan>Bench</text>`;
    g += `<text x="800" y="740" text-anchor="middle" font-size="26" class="mono par fade" ${d(3.4)}>arXiv:2609.01552</text>`;
    scenes.push({ name: 'Closing', dur: 8, svg: svg(g),
      cap: 'The real world tests scientific validity; the parallel world tests genuine discovery. <b>SciLaws-Bench</b> measures both.' });
  })();


  // ---------- player ----------
  const root = document.getElementById('film');
  if (!root) return;
  const stage = root.querySelector('.fm-stage'), cap = root.querySelector('.fm-cap');
  const prog = root.querySelector('.fm-track i'), timeEl = root.querySelector('.fm-time');
  const playBtn = root.querySelector('.fm-play'), big = root.querySelector('.fm-big'), chapNav = root.querySelector('.fm-chapters');
  const total = scenes.reduce((s, x) => s + x.dur, 0);
  const starts = []; scenes.reduce((s, x, i) => (starts[i] = s, s + x.dur), 0);
  const els = scenes.map(s => { const el = document.createElement('div'); el.className = 'fm-scene'; el.innerHTML = s.svg; stage.appendChild(el); return el; });
  const chaps = scenes.map((s, i) => {
    const b = document.createElement('button'); b.type = 'button';
    b.innerHTML = `<span>${String(i + 1).padStart(2, '0')}</span>${s.name}`;
    b.onclick = () => { go(i); play(true); }; chapNav.appendChild(b); return b;
  });
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const icons = { play: '<svg viewBox="0 0 16 16"><path d="M4 2l10 6-10 6z"/></svg>', pause: '<svg viewBox="0 0 16 16"><path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z"/></svg>' };
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  let cur = -1, t = 0, playing = false, last = null, userPaused = false;

  function go(i) {
    if (cur >= 0) els[cur].classList.remove('on');
    cur = i; t = starts[i];
    const el = els[i];
    void el.offsetWidth;            // restart CSS animations
    el.classList.add('on');
    cap.innerHTML = scenes[i].cap;
    chaps.forEach((c, k) => c.classList.toggle('on', k === i));
    el.querySelectorAll('[data-count]').forEach(n => {
      const raw = n.dataset.count, m = raw.match(/[\d.]+/), target = +m[0];
      if (reduce) { n.textContent = raw; return; }
      const t0 = performance.now() + 2600;
      const step = now => {
        if (!playing) { requestAnimationFrame(step); return; }
        const p = Math.min(1, Math.max(0, (now - t0) / 1400));
        n.textContent = raw.replace(m[0], Math.round(target * (1 - (1 - p) ** 3)));
        if (p < 1 && cur === i) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }
  function play(on) {
    playing = on; last = null;
    root.classList.toggle('paused', !on);
    playBtn.innerHTML = on ? icons.pause : icons.play;
    playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
  }
  function tick(now) {
    if (playing && last != null) {
      t += (now - last) / 1000;
      if (t >= total) { t = total; play(false); root.classList.add('ended'); }
      else if (cur < scenes.length - 1 && t >= starts[cur + 1]) go(cur + 1), t = starts[cur];
    }
    last = playing ? now : null;
    if (scenes[cur].turns) { const n = document.getElementById('fm-turn'); if (n) n.textContent = Math.min(30, Math.max(1, Math.round((t - starts[cur] - 3) / 5.5 * 30))); }
    prog.style.width = (t / total * 100) + '%';
    timeEl.textContent = `${fmt(t)} / ${fmt(total)}`;
    requestAnimationFrame(tick);
  }
  function toggle() {
    if (!playing && t >= total) { root.classList.remove('ended'); go(0); }
    userPaused = playing; play(!playing);
  }
  playBtn.onclick = toggle;
  stage.onclick = toggle;
  big.onclick = e => { e.stopPropagation(); toggle(); };
  root.querySelector('.fm-restart').onclick = () => { root.classList.remove('ended'); userPaused = false; go(0); play(true); };
  stage.addEventListener('keydown', e => {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggle(); }
    if (e.key === 'ArrowRight' && cur < scenes.length - 1) go(cur + 1);
    if (e.key === 'ArrowLeft' && cur > 0) go(cur - 1);
  });

  go(0); play(false);
  // autoplay while the film is on screen; pause when scrolled away
  if (!reduce && 'IntersectionObserver' in window) {
    new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting && !userPaused && t < total) play(true);
      else if (!e.isIntersecting && playing) play(false);
    }), { threshold: 0.35 }).observe(stage);
  }
  requestAnimationFrame(tick);
})();
