"""HTML, CSS and JavaScript for the clinician-facing 3D case viewer."""

VIEWER_CSS = """
body { margin: 0; background: #10141c; color: #e8edf5; font: 15px Segoe UI, Arial, sans-serif; }
main { max-width: 1500px; margin: auto; padding: 18px 24px 40px; }
h1 { font-size: 1.5rem; margin: 0 0 6px; }
h2 { font-size: 1.15rem; margin: 26px 0 8px; }
p { line-height: 1.5; }
.note { color: #b8c5d8; }
.footnote { color: #8e9bb0; font-size: 13px; }
.disclaimer { border-left: 4px solid #e56b6f; padding: 8px 12px; background: #281d22; margin-top: 22px; }
.stage { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; align-items: start; }
.panel { background: #171d29; border: 1px solid #263044; border-radius: 8px; padding: 12px 14px; }
.panel h3 { font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; color: #8fa3c0; margin: 12px 0 6px; }
.panel h3:first-child { margin-top: 0; }
.layer { display: flex; align-items: flex-start; gap: 8px; padding: 3px 0; cursor: pointer; font-size: 14px; }
.layer input { margin-top: 3px; }
.swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.layer small { display: block; color: #8e9bb0; }
.slice-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
figure { margin: 0; background: #171d29; border: 1px solid #263044; border-radius: 8px; padding: 10px; }
figcaption { font-weight: 600; margin-bottom: 6px; }
.wrap { position: relative; background: #000; }
.wrap canvas { display: block; width: 100%; image-rendering: pixelated; cursor: crosshair; }
.o { position: absolute; color: #9fb4d3; font-size: 12px; font-weight: 700; pointer-events: none; text-shadow: 0 0 3px #000; }
.o.t { top: 3px; left: 50%; transform: translateX(-50%); }
.o.b { bottom: 3px; left: 50%; transform: translateX(-50%); }
.o.l { left: 5px; top: 50%; transform: translateY(-50%); }
.o.r { right: 5px; top: 50%; transform: translateY(-50%); }
input[type=range] { width: 100%; margin-top: 8px; }
.pos { color: #9fb4d3; font-size: 13px; }
table { border-collapse: collapse; width: 100%; background: #171d29; border: 1px solid #263044; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #263044; font-size: 14px; vertical-align: top; }
th { color: #8fa3c0; font-weight: 600; background: #1b2332; }
#slice-error { display: none; border-left: 4px solid #e6aa36; padding: 8px 12px; background: #252319; }
@media (max-width: 800px) { .stage { grid-template-columns: 1fr; } .slice-grid { grid-template-columns: 1fr; } }
"""

VIEWER_JS = """
(function () {
  var data = JSON.parse(document.getElementById('case-data').textContent);
  var gd = document.getElementById('viewer3d');
  var nx = data.dims[0], ny = data.dims[1], nz = data.dims[2];
  var sp = data.spacing;
  var state = { layers: {}, pos: data.center.slice() };
  data.layers.forEach(function (l) { state.layers[l.key] = l.visible; });

  document.querySelectorAll('.layer input').forEach(function (box) {
    box.addEventListener('change', function () {
      state.layers[box.dataset.key] = box.checked;
      var traces = box.dataset.traces ? box.dataset.traces.split(',').map(Number) : [];
      if (traces.length && window.Plotly) { Plotly.restyle(gd, { visible: box.checked }, traces); }
      if (vols) { drawAll(); }
    });
  });

  var vols = null;
  var planes = {
    axial:    { w: nx, h: ny, fixed: 2, size: nz, px: [sp[0], sp[1]],
                map: function (c, r, s) { return [c, ny - 1 - r, s]; },
                cross: function (p) { return [p[0], ny - 1 - p[1]]; },
                set: function (p, c, r) { p[0] = c; p[1] = ny - 1 - r; } },
    coronal:  { w: nx, h: nz, fixed: 1, size: ny, px: [sp[0], sp[2]],
                map: function (c, r, s) { return [c, s, nz - 1 - r]; },
                cross: function (p) { return [p[0], nz - 1 - p[2]]; },
                set: function (p, c, r) { p[0] = c; p[2] = nz - 1 - r; } },
    sagittal: { w: ny, h: nz, fixed: 0, size: nx, px: [sp[1], sp[2]],
                map: function (c, r, s) { return [s, c, nz - 1 - r]; },
                cross: function (p) { return [p[1], nz - 1 - p[2]]; },
                set: function (p, c, r) { p[1] = c; p[2] = nz - 1 - r; } }
  };

  function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
  var colors = {};
  data.corridors.forEach(function (c) { colors['c' + c.id] = hex(c.color); });
  data.structures.forEach(function (s) { colors['s' + s.id] = hex(s.color); });

  function turbo(x) {
    x = Math.max(0, Math.min(1, x));
    var r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    var g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    var b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    return [r, g, b].map(function (v) { return Math.max(0, Math.min(255, Math.round(v * 255))); });
  }
  var riskMin = data.risk_range[0], riskMax = data.risk_range[1];
  var lut = [];
  for (var q = 0; q < 256; q++) { lut.push(turbo((q / 255 - riskMin) / Math.max(riskMax - riskMin, 1e-6))); }

  function blend(px, color, a) {
    px[0] = px[0] * (1 - a) + color[0] * a;
    px[1] = px[1] * (1 - a) + color[1] * a;
    px[2] = px[2] * (1 - a) + color[2] * a;
  }

  var canvases = {};
  function draw(name) {
    var P = planes[name], cv = canvases[name], ctx = cv.getContext('2d');
    var img = ctx.createImageData(P.w, P.h), out = img.data, s = state.pos[P.fixed];
    var riskCut = Math.round(riskMin * 255);
    function idx(c, r) { var m = P.map(c, r, s); return (m[0] * ny + m[1]) * nz + m[2]; }
    for (var r = 0; r < P.h; r++) {
      for (var c = 0; c < P.w; c++) {
        var i = idx(c, r), g = vols.mri[i], px = [g, g, g];
        if (state.layers.risk) {
          var rv = vols.risk[i];
          if (rv >= riskCut && rv > 0) { blend(px, lut[rv], 0.18 + 0.4 * (rv - riskCut) / Math.max(255 - riskCut, 1)); }
        }
        if (vols.structures) {
          var sid = vols.structures[i];
          if (sid && state.layers['structure-' + sid]) { blend(px, colors['s' + sid], 0.45); }
        }
        if (vols.corridors) {
          var cid = vols.corridors[i];
          if (cid && state.layers['corridor-' + cid]) { blend(px, colors['c' + cid], 0.85); }
        }
        if (state.layers.tumor && vols.tumor[i]) {
          var edge = c === 0 || r === 0 || c === P.w - 1 || r === P.h - 1 ||
            !vols.tumor[idx(c - 1, r)] || !vols.tumor[idx(c + 1, r)] ||
            !vols.tumor[idx(c, r - 1)] || !vols.tumor[idx(c, r + 1)];
          blend(px, [255, 49, 88], edge ? 1 : 0.18);
        }
        var o = (r * P.w + c) * 4;
        out[o] = px[0]; out[o + 1] = px[1]; out[o + 2] = px[2]; out[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    var cross = P.cross(state.pos);
    ctx.strokeStyle = 'rgba(0,229,255,0.75)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cross[0] + 0.5, 0); ctx.lineTo(cross[0] + 0.5, P.h);
    ctx.moveTo(0, cross[1] + 0.5); ctx.lineTo(P.w, cross[1] + 0.5);
    ctx.stroke();
  }

  function describe(name) {
    var P = planes[name], p = state.pos, a = data.affine;
    var w = [0, 1, 2].map(function (row) { return a[row][0] * p[0] + a[row][1] * p[1] + a[row][2] * p[2] + a[row][3]; });
    var axis = P.fixed, v = w[axis];
    var tags = [['L', 'R'], ['P', 'A'], ['I', 'S']][axis];
    return 'Slice ' + (p[axis] + 1) + ' of ' + P.size + ' (' + tags[v >= 0 ? 1 : 0] + ' ' + Math.abs(v).toFixed(1) + ' mm)';
  }

  function drawAll() {
    Object.keys(planes).forEach(function (name) {
      draw(name);
      document.getElementById('pos-' + name).textContent = describe(name);
      document.getElementById('sl-' + name).value = state.pos[planes[name].fixed];
    });
  }

  async function inflate(b64) {
    var bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) { bytes[i] = bin.charCodeAt(i); }
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function init() {
    try {
      var loaded = {};
      for (var key in data.volumes) {
        loaded[key] = data.volumes[key] ? await inflate(data.volumes[key]) : null;
      }
      vols = loaded;
    } catch (err) {
      var box = document.getElementById('slice-error');
      box.style.display = 'block';
      box.textContent = 'Slice views could not be loaded in this browser. Use a current version of Chrome, Edge or Firefox. The 3D view is unaffected.';
      document.getElementById('slice-grid').style.display = 'none';
      return;
    }
    Object.keys(planes).forEach(function (name) {
      var P = planes[name], cv = document.getElementById('cv-' + name);
      cv.width = P.w; cv.height = P.h;
      cv.style.aspectRatio = (P.w * P.px[0]) + ' / ' + (P.h * P.px[1]);
      canvases[name] = cv;
      var slider = document.getElementById('sl-' + name);
      slider.max = P.size - 1;
      slider.addEventListener('input', function () { state.pos[P.fixed] = Number(slider.value); drawAll(); });
      function pick(ev) {
        var rect = cv.getBoundingClientRect();
        var c = Math.min(P.w - 1, Math.max(0, Math.floor((ev.clientX - rect.left) / rect.width * P.w)));
        var r = Math.min(P.h - 1, Math.max(0, Math.floor((ev.clientY - rect.top) / rect.height * P.h)));
        P.set(state.pos, c, r);
        drawAll();
      }
      var dragging = false;
      cv.addEventListener('mousedown', function (ev) { dragging = true; pick(ev); });
      cv.addEventListener('mousemove', function (ev) { if (dragging) { pick(ev); } });
      window.addEventListener('mouseup', function () { dragging = false; });
    });
    drawAll();
  }
  init();
})();
"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__ - 3D case viewer</title>
  <style>__CSS__</style>
</head>
<body>
  <main>
    <h1>__TITLE__ - 3D case visualization</h1>
    <p class="note">__SUMMARY__ Rotate and zoom the 3D view, show or hide
       items in the <strong>Layers</strong> panel, and click or drag in the slice views below to move through the scan.</p>
    <div class="stage">
      <div id="viewer">__PLOT__</div>
      <aside class="panel">__LAYERS__</aside>
    </div>
    <h2>Slice views</h2>
    <p class="note">Left (L) and right (R) of the patient are marked on each image. The same layers apply here as in the 3D view.</p>
    <div id="slice-error"></div>
    <div class="slice-grid" id="slice-grid">
      __SLICES__
    </div>
    __TABLES__
    <p class="disclaimer">__DISCLAIMER__</p>
  </main>
  <script id="case-data" type="application/json">__CASE_DATA__</script>
  <script>__JS__</script>
</body>
</html>
"""

SLICE_FIGURES = """
<figure><figcaption>Axial (from above)</figcaption>
  <div class="wrap"><canvas id="cv-axial"></canvas>
    <span class="o t">A</span><span class="o b">P</span><span class="o l">L</span><span class="o r">R</span></div>
  <input type="range" id="sl-axial" min="0" value="0"><div class="pos" id="pos-axial"></div></figure>
<figure><figcaption>Coronal (from the front)</figcaption>
  <div class="wrap"><canvas id="cv-coronal"></canvas>
    <span class="o t">S</span><span class="o b">I</span><span class="o l">L</span><span class="o r">R</span></div>
  <input type="range" id="sl-coronal" min="0" value="0"><div class="pos" id="pos-coronal"></div></figure>
<figure><figcaption>Sagittal (from the side)</figcaption>
  <div class="wrap"><canvas id="cv-sagittal"></canvas>
    <span class="o t">S</span><span class="o b">I</span><span class="o l">P</span><span class="o r">A</span></div>
  <input type="range" id="sl-sagittal" min="0" value="0"><div class="pos" id="pos-sagittal"></div></figure>
"""
