/* BL View quicklook renderer.
 *
 * The backscatter field arrives from /api/window already block-averaged to the
 * requested display size and quantised to one byte per cell in log10 space, so
 * this file never handles more numbers than there are pixels to draw. Two
 * stacked canvases: the heatmap is painted at the grid's native size through an
 * ImageData LUT and scaled up with smoothing off (so a cell is honestly a cell,
 * not an interpolation), and the overlay carries axes, layer lines, the
 * crosshair and the screened-period band.
 */

const PLOT = { left: 58, right: 12, top: 10, bottom: 34 };

/* Display colour-scale windows, log10(beta).
 * The API quantises over a deliberately wide range (-7.5 .. -3.5) so nothing is
 * lost in the payload, but showing that whole range spends most of the ramp on
 * far-field detector noise: above ~3 km the R^2-growing noise is comparable to
 * mixing-layer backscatter, and rendering it across half the ramp turns the top
 * of every plot into salt-and-pepper. The default display window starts at the
 * noise floor instead, so noise clamps flat at the bottom of the scale and the
 * ramp is spent on real structure. "Full" is kept for when the noise itself is
 * what you want to look at. */
const SCALES = {
  // Bottom sits at the far-field noise level so noise clamps flat instead of
  // shimmering across the ramp; top sits just above mixing-layer backscatter so
  // the whole ramp is spent separating mixing layer from residual layer from
  // haze. Cloud is two orders of magnitude higher and clamps to the darkest
  // step, which is the honest reading -- a ceilometer cannot resolve *how*
  // bright a cloud is once the return saturates.
  default: [-6.3, -5.0],
  full: null,       // whatever the payload was encoded with
  auto: null,       // 1st-99th percentile of this window
};

/** Layer identity: the four validated categorical slots, plus a dash pattern.
 *  The dash is secondary encoding -- identity never rests on hue alone. */
const LAYERS = [
  { id: "mixing_layer",   label: "Mixing layer top",   varName: "--layer-mixing",   dash: [],       field: "top"  },
  { id: "residual_layer", label: "Residual layer top", varName: "--layer-residual", dash: [9, 5],   field: "top"  },
  { id: "haze",           label: "Haze base",          varName: "--layer-haze",     dash: [],       field: "base" },
  { id: "haze_top",       label: "Haze top",           varName: "--layer-haze",     dash: [2, 4],   field: "top", source: "haze" },
  { id: "cloud",          label: "Cloud base",         varName: "--layer-cloud",    dash: [5, 3],   field: "base", width: 2.5 },
];

const QUALITY_BITS = [
  [1, "precipitation"], [2, "fog"], [4, "low SNR"], [8, "saturated"],
  [16, "instrument warning"], [32, "instrument alarm"],
  [64, "window contaminated"], [128, "detection skipped"],
];

const state = {
  window: null,
  grid: null,          // Uint8Array view over the payload
  hours: 24,
  maxHeight: 4000,
  scale: "default",
  minConfidence: 0,
  enabled: new Set(["mixing_layer", "residual_layer", "haze", "cloud"]),
  hover: null,
};

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* --------------------------------------------------------------- utilities */
function hhmm(seconds) {
  const d = new Date(seconds * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
}
function stamp(seconds) {
  const d = new Date(seconds * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} `
       + `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}Z`;
}
function metres(v) {
  return v === null || v === undefined || !Number.isFinite(v)
    ? null : `${Math.round(v).toLocaleString("en-GB")} m`;
}
/** Format log10(beta) back as a physical value, e.g. 2.4e-6. */
function betaLabel(log10) {
  if (!Number.isFinite(log10)) return "—";
  const v = Math.pow(10, log10);
  return `${v.toExponential(1)} m⁻¹sr⁻¹`;
}
function qualityNames(bits) {
  return QUALITY_BITS.filter(([mask]) => (bits & mask) !== 0).map(([, name]) => name);
}
function parseHex(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** 256-entry RGBA lookup built from the 13-step sequential ramp.
 *  Index 0 is the API's "no data" code and stays fully transparent, so masked
 *  gates show the card surface instead of pretending to be a low value. */
function buildLut() {
  const steps = [];
  for (let i = 0; i <= 12; i++) steps.push(parseHex(css(`--ramp-${i}`)));
  const lut = new Uint8ClampedArray(256 * 4);
  for (let code = 1; code < 256; code++) {
    const t = (code - 1) / 254 * (steps.length - 1);
    const i = Math.min(Math.floor(t), steps.length - 2);
    const f = t - i;
    for (let c = 0; c < 3; c++) {
      lut[code * 4 + c] = steps[i][c] + (steps[i + 1][c] - steps[i][c]) * f;
    }
    lut[code * 4 + 3] = 255;
  }
  return lut;
}

/* ------------------------------------------------------------------ fetch */
async function loadWindow() {
  const params = new URLSearchParams({
    hours: String(state.hours),
    max_time: "1400",
    max_range: "460",
    min_confidence: String(state.minConfidence),
  });
  if (state.maxHeight) params.set("max_height", String(state.maxHeight));

  const response = await fetch(`/api/window?${params}`);
  if (!response.ok) throw new Error(`window request failed: ${response.status}`);
  const payload = await response.json();

  const binary = atob(payload.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  payload.data = null;

  state.window = payload;
  state.grid = bytes;
  return payload;
}

async function loadStatus() {
  const response = await fetch("/api/status");
  return response.ok ? response.json() : null;
}

async function loadLatest() {
  const response = await fetch("/api/profile/latest");
  return response.ok ? response.json() : null;
}

/* ------------------------------------------------------------- colour scale */
function scaleLimits() {
  const w = state.window;
  if (state.scale === "auto"
      && Number.isFinite(w?.percentiles?.p01) && Number.isFinite(w?.percentiles?.p99)) {
    return [w.percentiles.p01, w.percentiles.p99];
  }
  if (state.scale === "default") return SCALES.default;
  return [w.encoding.vmin, w.encoding.vmax];
}

/** Byte code -> log10(beta), inverting the API's quantisation. */
function decodeCode(code) {
  const w = state.window;
  if (code === 0) return NaN;
  return w.encoding.vmin + (code - 1) / 254 * (w.encoding.vmax - w.encoding.vmin);
}

/* ------------------------------------------------------------- heatmap draw */
function drawHeatmap() {
  const canvas = $("heatmap");
  const w = state.window;
  const [nTime, nRange] = w.shape;
  const dpr = window.devicePixelRatio || 1;
  const box = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  if (!nTime || !nRange) return;

  // Re-quantise into the *display* range when auto-scaling, so the ramp spends
  // its whole 254 steps on the data actually present.
  const [lo, hi] = scaleLimits();
  const span = Math.max(hi - lo, 1e-9);
  const remap = new Uint8Array(256);
  for (let code = 1; code < 256; code++) {
    const value = decodeCode(code);
    remap[code] = 1 + Math.round(Math.min(Math.max((value - lo) / span, 0), 1) * 254);
  }

  const lut = buildLut();
  const image = new ImageData(nTime, nRange);
  const out = image.data;
  const grid = state.grid;
  for (let t = 0; t < nTime; t++) {
    const column = t * nRange;
    for (let j = 0; j < nRange; j++) {
      // Range index 0 is the lowest gate, but ImageData row 0 is the top of
      // the picture -- flip so height increases upward.
      const row = nRange - 1 - j;
      const src = remap[grid[column + j]] * 4;
      const dst = (row * nTime + t) * 4;
      out[dst] = lut[src];
      out[dst + 1] = lut[src + 1];
      out[dst + 2] = lut[src + 2];
      out[dst + 3] = grid[column + j] === 0 ? 0 : 255;
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = nTime;
  offscreen.height = nRange;
  offscreen.getContext("2d").putImageData(image, 0, 0);

  const area = plotArea(box);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, area.x, area.y, area.w, area.h);
}

function plotArea(box) {
  return {
    x: PLOT.left,
    y: PLOT.top,
    w: Math.max(1, box.width - PLOT.left - PLOT.right),
    h: Math.max(1, box.height - PLOT.top - PLOT.bottom),
  };
}

/* ------------------------------------------------------------ overlay draw */
function axisTicks(lo, hi, target) {
  const raw = (hi - lo) / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return ticks;
}

function timeTicks(t0, t1, width) {
  const span = t1 - t0;
  const target = Math.max(2, Math.floor(width / 110));
  const choices = [300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];
  const step = choices.find((s) => span / s <= target) ?? 86400;
  const ticks = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) ticks.push(t);
  return ticks;
}

function drawOverlay() {
  const canvas = $("overlay");
  const w = state.window;
  const dpr = window.devicePixelRatio || 1;
  const box = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  if (!w || !w.time.length) {
    ctx.fillStyle = css("--text-muted");
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No data in the store yet — run `blview ingest` or `blview demo`.",
                 box.width / 2, box.height / 2);
    return;
  }

  const area = plotArea(box);
  const t0 = w.time[0], t1 = w.time[w.time.length - 1];
  const r0 = 0, r1 = w.range[w.range.length - 1];
  const xOf = (t) => area.x + (t - t0) / Math.max(t1 - t0, 1e-9) * area.w;
  const yOf = (r) => area.y + area.h - (r - r0) / Math.max(r1 - r0, 1e-9) * area.h;

  // Gridlines: hairline, solid, recessive -- and drawn under everything.
  ctx.lineWidth = 1;
  ctx.strokeStyle = css("--gridline");
  ctx.fillStyle = css("--text-muted");
  ctx.font = "11px system-ui, sans-serif";

  const heightTicks = axisTicks(r0, r1, Math.max(3, Math.round(area.h / 70)));
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const value of heightTicks) {
    const y = Math.round(yOf(value)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(area.x, y);
    ctx.lineTo(area.x + area.w, y);
    ctx.stroke();
    ctx.fillText(value.toLocaleString("en-GB"), area.x - 8, y);
  }
  ctx.save();
  ctx.translate(14, area.y + area.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("height above instrument (m)", 0, 0);
  ctx.restore();

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of timeTicks(t0, t1, area.w)) {
    const x = Math.round(xOf(t)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, area.y);
    ctx.lineTo(x, area.y + area.h);
    ctx.stroke();
    ctx.fillText(hhmm(t), x, area.y + area.h + 14);
  }
  ctx.fillText(`UTC — ${stamp(t0)} to ${stamp(t1)}`, area.x + area.w / 2, area.y + area.h + 26);

  ctx.strokeStyle = css("--axis");
  ctx.strokeRect(area.x + 0.5, area.y + 0.5, area.w - 1, area.h - 1);

  drawScreenedBand(ctx, area, xOf);
  drawLayers(ctx, area, xOf, yOf);
  drawCrosshair(ctx, area, xOf, yOf);
}

/** Screened periods get a hatched strip at the plot foot: the backscatter is
 *  still shown, but no layer is claimed there, and the reader must be able to
 *  see which is which. */
function drawScreenedBand(ctx, area, xOf) {
  const w = state.window;
  const bandHeight = 7;
  const y = area.y + area.h - bandHeight;
  const dt = w.time.length > 1 ? w.time[1] - w.time[0] : 1;
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.x, y, area.w, bandHeight);
  ctx.clip();
  ctx.strokeStyle = css("--layer-mixing");
  ctx.lineWidth = 1;
  for (let i = 0; i < w.quality.length; i++) {
    const bits = w.quality[i];
    if (!(bits & (1 | 2 | 4 | 32))) continue;      // precip | fog | low SNR | alarm
    const x0 = xOf(w.time[i]);
    const x1 = xOf(w.time[i] + dt);
    for (let x = x0 - bandHeight; x < x1; x += 4) {
      ctx.beginPath();
      ctx.moveTo(x, y + bandHeight);
      ctx.lineTo(x + bandHeight, y);
      ctx.stroke();
    }
  }
  ctx.restore();
}

/** Group a columnar layer array into per-track polylines. */
function tracksOf(arrays, field) {
  if (!arrays) return [];
  const byTrack = new Map();
  for (let i = 0; i < arrays.time.length; i++) {
    const value = arrays[field][i];
    if (value === null || value === undefined) continue;
    const key = arrays.track_id[i] ?? `_${i}`;
    if (!byTrack.has(key)) byTrack.set(key, []);
    byTrack.get(key).push({
      t: arrays.time[i], v: value,
      confidence: arrays.confidence[i],
      interpolated: arrays.interpolated[i] === 1,
    });
  }
  for (const points of byTrack.values()) points.sort((a, b) => a.t - b.t);
  return [...byTrack.values()];
}

function drawLayers(ctx, area, xOf, yOf) {
  const w = state.window;
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.x, area.y, area.w, area.h);
  ctx.clip();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  for (const spec of LAYERS) {
    const source = spec.source ?? spec.id;
    if (!state.enabled.has(source)) continue;
    const arrays = w.layers[source];
    if (!arrays) continue;
    const colour = css(spec.varName);

    for (const points of tracksOf(arrays, spec.field)) {
      if (points.length < 2) continue;
      // Interpolated (gap-filled) points are drawn at reduced opacity so a
      // filled value never masquerades as a measured one.
      const runs = [];
      let run = [points[0]];
      for (let i = 1; i < points.length; i++) {
        if (points[i].interpolated !== run[0].interpolated) {
          run.push(points[i]);
          runs.push(run);
          run = [points[i]];
        } else {
          run.push(points[i]);
        }
      }
      runs.push(run);

      for (const segment of runs) {
        if (segment.length < 2) continue;
        const trace = () => {
          ctx.beginPath();
          segment.forEach((p, i) => {
            const x = xOf(p.t), y = yOf(p.v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          });
        };
        // Surface halo first: the line has to stay legible wherever it crosses
        // a dark part of the heatmap or another layer.
        ctx.globalAlpha = segment[0].interpolated ? 0.35 : 0.65;
        ctx.setLineDash([]);
        ctx.strokeStyle = css("--surface-1");
        ctx.lineWidth = 5;
        trace();
        ctx.stroke();

        ctx.globalAlpha = segment[0].interpolated ? 0.45 : 1;
        ctx.setLineDash(spec.dash);
        ctx.strokeStyle = colour;
        ctx.lineWidth = spec.width ?? 2;
        trace();
        ctx.stroke();
      }
    }
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawCrosshair(ctx, area, xOf, yOf) {
  const hover = state.hover;
  if (!hover) return;
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = css("--text-secondary");
  ctx.lineWidth = 1;
  const x = Math.round(xOf(hover.time)) + 0.5;
  const y = Math.round(yOf(hover.height)) + 0.5;
  ctx.beginPath();
  ctx.moveTo(x, area.y);
  ctx.lineTo(x, area.y + area.h);
  ctx.moveTo(area.x, y);
  ctx.lineTo(area.x + area.w, y);
  ctx.stroke();
  ctx.restore();
}

/* ------------------------------------------------------------------- hover */
function nearestIndex(values, target) {
  let lo = 0, hi = values.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < target) lo = mid; else hi = mid;
  }
  return Math.abs(values[lo] - target) <= Math.abs(values[hi] - target) ? lo : hi;
}

function onPointerMove(event) {
  const w = state.window;
  if (!w || !w.time.length) return;
  const box = $("overlay").parentElement.getBoundingClientRect();
  const area = plotArea(box);
  const px = event.clientX - box.left;
  const py = event.clientY - box.top;
  if (px < area.x || px > area.x + area.w || py < area.y || py > area.y + area.h) {
    hideTooltip();
    return;
  }
  const t0 = w.time[0], t1 = w.time[w.time.length - 1];
  const r1 = w.range[w.range.length - 1];
  const time = t0 + (px - area.x) / area.w * (t1 - t0);
  const height = (area.y + area.h - py) / area.h * r1;

  const ti = nearestIndex(w.time, time);
  const ri = nearestIndex(w.range, height);
  state.hover = { time: w.time[ti], height: w.range[ri], ti, ri };
  showTooltip(box, area, px, py);
  drawOverlay();
}

function layersAt(timeValue) {
  const w = state.window;
  const found = [];
  for (const spec of LAYERS) {
    const source = spec.source ?? spec.id;
    if (!state.enabled.has(source) || spec.source) continue;   // one row per layer
    const arrays = w.layers[source];
    if (!arrays || !arrays.time.length) continue;
    const i = nearestIndex(arrays.time, timeValue);
    // Only report a layer if it actually exists near this time.
    const dt = w.time.length > 1 ? (w.time[1] - w.time[0]) : 60;
    if (Math.abs(arrays.time[i] - timeValue) > Math.max(dt * 1.5, 90)) continue;
    found.push({
      label: spec.label.replace(/ (top|base)$/, ""),
      colour: css(spec.varName),
      base: arrays.base[i],
      top: arrays.top[i],
      confidence: arrays.confidence[i],
      interpolated: arrays.interpolated[i] === 1,
    });
  }
  return found;
}

function showTooltip(box, area, px, py) {
  const w = state.window;
  const { time, height, ti, ri } = state.hover;
  const tooltip = $("tooltip");
  tooltip.replaceChildren();

  const title = document.createElement("h3");
  title.textContent = `${stamp(time)} · ${metres(height)}`;
  tooltip.appendChild(title);

  const list = document.createElement("dl");
  const row = (name, value, colour) => {
    const dt = document.createElement("dt");
    if (colour) {
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = colour;
      dt.appendChild(swatch);
    }
    dt.appendChild(document.createTextNode(name));
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  };

  const code = state.grid[ti * w.shape[1] + ri];
  row("Backscatter", code === 0 ? "no data" : betaLabel(decodeCode(code)));

  for (const layer of layersAt(time)) {
    const value = layer.top !== null && layer.top !== undefined
      ? `${metres(layer.base)} – ${metres(layer.top)}`
      : `${metres(layer.base)} – top n/d`;
    row(layer.label + (layer.interpolated ? " (filled)" : ""), value, layer.colour);
  }
  tooltip.appendChild(list);

  const flags = qualityNames(w.quality[ti]);
  if (flags.length) {
    const note = document.createElement("p");
    note.className = "screened";
    note.textContent = `Screened: ${flags.join(", ")} — no layers claimed here.`;
    tooltip.appendChild(note);
  }

  tooltip.hidden = false;
  const rect = tooltip.getBoundingClientRect();
  const flipX = px + 16 + rect.width > box.width;
  tooltip.style.left = `${Math.max(4, flipX ? px - rect.width - 16 : px + 16)}px`;
  tooltip.style.top = `${Math.min(Math.max(4, py - rect.height / 2), box.height - rect.height - 4)}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
  if (state.hover) {
    state.hover = null;
    drawOverlay();
  }
}

/* ------------------------------------------------------------------ legend */
function drawLegend() {
  const legend = $("legend");
  legend.replaceChildren();
  const w = state.window;
  for (const spec of LAYERS) {
    const source = spec.source ?? spec.id;
    if (!state.enabled.has(source)) continue;
    const arrays = w?.layers?.[source];
    const count = arrays ? arrays.time.filter((_, i) => arrays[spec.field][i] !== null).length : 0;
    if (!count) continue;

    const item = document.createElement("div");
    item.className = "item";
    item.setAttribute("role", "listitem");

    const key = document.createElement("canvas");
    key.className = "key";
    key.width = 44; key.height = 20;
    const ctx = key.getContext("2d");
    ctx.scale(2, 2);
    ctx.strokeStyle = css(spec.varName);
    ctx.lineWidth = spec.width ?? 2;
    ctx.lineCap = "round";
    ctx.setLineDash(spec.dash);
    ctx.beginPath();
    ctx.moveTo(1, 5);
    ctx.lineTo(21, 5);
    ctx.stroke();
    item.appendChild(key);

    const label = document.createElement("span");
    label.textContent = spec.label;
    item.appendChild(label);

    const badge = document.createElement("span");
    badge.className = "count";
    badge.textContent = count.toLocaleString("en-GB");
    item.appendChild(badge);
    legend.appendChild(item);
  }

  if (!legend.children.length) {
    const empty = document.createElement("span");
    empty.className = "count";
    empty.textContent = "no layers in this window";
    legend.appendChild(empty);
  }
}

function drawColourbar() {
  const canvas = $("colourbar");
  const ctx = canvas.getContext("2d");
  const lut = buildLut();
  const image = ctx.createImageData(canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y++) {
    const code = 1 + Math.round((1 - y / (canvas.height - 1)) * 254);
    for (let x = 0; x < canvas.width; x++) {
      const dst = (y * canvas.width + x) * 4;
      image.data[dst] = lut[code * 4];
      image.data[dst + 1] = lut[code * 4 + 1];
      image.data[dst + 2] = lut[code * 4 + 2];
      image.data[dst + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);

  const ticks = $("colourbar-ticks");
  ticks.replaceChildren();
  if (!state.window) return;
  const [lo, hi] = scaleLimits();
  for (const value of axisTicks(lo, hi, 5)) {
    const span = document.createElement("span");
    const exponent = Math.round(value * 10) / 10;
    span.textContent = `10${String(exponent).replace(/-/g, "⁻")
      .replace(/[0-9]/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)])
      .replace(".", "·")}`;
    span.style.top = `${(1 - (value - lo) / Math.max(hi - lo, 1e-9)) * canvas.height}px`;
    ticks.appendChild(span);
  }
}

/* ------------------------------------------------------------ latest panel */
function renderLatest(profile) {
  const body = $("latest-table").querySelector("tbody");
  body.replaceChildren();
  if (!profile) {
    $("latest-time").textContent = "no data";
    return;
  }
  $("latest-time").textContent = stamp(profile.time);

  const specByType = Object.fromEntries(
    LAYERS.filter((s) => !s.source).map((s) => [s.id, s])
  );
  const rows = profile.layers.slice().sort((a, b) => a.base_height - b.base_height);
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "none";
    td.textContent = profile.screened
      ? `no layers — profile screened (${profile.quality_flags.join(", ")})`
      : "no layers detected in this profile";
    tr.appendChild(td);
    body.appendChild(tr);
  }
  for (const layer of rows) {
    const spec = specByType[layer.type];
    const tr = document.createElement("tr");

    const name = document.createElement("td");
    const wrap = document.createElement("span");
    wrap.className = "name";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = spec ? css(spec.varName) : css("--text-muted");
    wrap.appendChild(swatch);
    wrap.appendChild(document.createTextNode(layer.type.replace(/_/g, " ")));
    name.appendChild(wrap);

    const base = document.createElement("td");
    base.textContent = metres(layer.base_height) ?? "—";

    const top = document.createElement("td");
    if (layer.top_height === null || layer.top_height === undefined) {
      top.className = "none";
      top.textContent = layer.type === "cloud" ? "not determinable" : "—";
      top.title = layer.type === "cloud"
        ? "The beam was extinguished inside the cloud, so no top can be measured."
        : "";
    } else {
      top.textContent = metres(layer.top_height);
    }

    const confidence = document.createElement("td");
    confidence.textContent = layer.confidence.toFixed(2);

    const source = document.createElement("td");
    source.className = layer.interpolated ? "none" : "";
    source.textContent = layer.interpolated ? "gap-filled" : "measured";

    tr.append(name, base, top, confidence, source);
    body.appendChild(tr);
  }

  $("latest-note").textContent = profile.screened
    ? `Profile flagged: ${profile.quality_flags.join(", ")}. Layer detection is suppressed.`
    : "Confidence is a relative quality ranking, not a probability.";
}

/* -------------------------------------------------------------------- boot */
function render() {
  drawHeatmap();
  drawOverlay();
  drawLegend();
  drawColourbar();
}

async function refresh() {
  try {
    const [payload, status, latest] = await Promise.all([
      loadWindow(), loadStatus(), loadLatest(),
    ]);
    if (status) {
      const bits = [status.site_name, status.instrument];
      if (status.hours_available) bits.push(`${status.hours_available.toFixed(1)} h in store`);
      $("site-line").textContent = bits.filter(Boolean).join(" · ");
    }
    const counts = Object.entries(payload.layers || {})
      .map(([k, v]) => `${v.time.length.toLocaleString("en-GB")} ${k.replace(/_/g, " ")}`);
    $("chart-subtitle").textContent = payload.time.length
      ? `${payload.downsample.n_time_full.toLocaleString("en-GB")} profiles averaged into `
        + `${payload.shape[0]} columns · ${counts.join(", ")}`
      : "no data in the store";
    const screened = payload.quality.filter((q) => (q & (1 | 2 | 4 | 32)) !== 0).length;
    $("screen-note").textContent = screened
      ? `${screened} of ${payload.quality.length} columns screened for precipitation, `
        + "fog or low SNR (hatched at the plot foot) — backscatter is still shown there, "
        + "but no layer is claimed."
      : "No screened periods in this window.";
    renderLatest(latest);
    render();
  } catch (error) {
    $("chart-subtitle").textContent = `Could not load data: ${error.message}`;
  }
}

function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode);
  $("theme-label").textContent = mode === "auto" ? "Auto" : mode === "dark" ? "Dark" : "Light";
  try { localStorage.setItem("blview-theme", mode); } catch { /* private mode */ }
  if (state.window) render();
}

function init() {
  let saved = "auto";
  try { saved = localStorage.getItem("blview-theme") || "auto"; } catch { /* ignore */ }
  applyTheme(saved);

  $("theme-toggle").addEventListener("click", () => {
    const order = ["auto", "light", "dark"];
    const next = order[(order.indexOf(document.documentElement.getAttribute("data-theme")) + 1) % 3];
    applyTheme(next);
  });
  $("hours").addEventListener("change", (e) => { state.hours = Number(e.target.value); refresh(); });
  $("max-height").addEventListener("change", (e) => {
    state.maxHeight = e.target.value ? Number(e.target.value) : null;
    refresh();
  });
  $("scale").addEventListener("change", (e) => { state.scale = e.target.value; render(); });
  $("confidence").addEventListener("input", (e) => {
    state.minConfidence = Number(e.target.value);
    $("confidence-out").textContent = state.minConfidence.toFixed(2);
  });
  $("confidence").addEventListener("change", refresh);
  $("refresh").addEventListener("click", refresh);
  for (const box of document.querySelectorAll("[data-layer]")) {
    box.addEventListener("change", () => {
      box.checked ? state.enabled.add(box.dataset.layer) : state.enabled.delete(box.dataset.layer);
      render();
    });
  }
  const overlay = $("overlay");
  overlay.addEventListener("pointermove", onPointerMove);
  overlay.addEventListener("pointerleave", hideTooltip);
  window.addEventListener("resize", () => { if (state.window) render(); });

  refresh();
}

init();
