#!/usr/bin/env python3
"""
nn_visualizer.py
----------------
Génère un fichier HTML interactif de visualisation d'un réseau de neurones
à partir d'un fichier .npz contenant les poids et biais.

Architectures supportées :
  - 1 couche cachée  : W1, b1, W2, b2
  - 2 couches cachées: W1, b1, W2, b2, W3, b3

Usage :
  python3 nn_visualizer.py <fichier.npz> [-o sortie.html]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# 1. Chargement et détection de l'architecture
# ─────────────────────────────────────────────

def load_network(path: str) -> dict:
    """Charge un .npz et retourne un dict structuré avec weights, biases, arch."""
    data = np.load(path)
    keys = set(data.files)

    # Détecter le nombre de couches
    if "W3" in keys and "b3" in keys:
        n_layers = 2
        weights = [data["W1"], data["W2"], data["W3"]]
        biases  = [data["b1"].flatten(), data["b2"].flatten(), data["b3"].flatten()]
    elif "W2" in keys and "b2" in keys:
        n_layers = 1
        weights = [data["W1"], data["W2"]]
        biases  = [data["b1"].flatten(), data["b2"].flatten()]
    else:
        raise ValueError(
            f"Architecture non reconnue. Clés trouvées : {sorted(keys)}\n"
            "Attendu : W1/b1/W2/b2 (1 couche) ou W1/b1/W2/b2/W3/b3 (2 couches)."
        )

    # Déduire les tailles des couches
    n_input   = weights[0].shape[1]
    hidden    = [w.shape[0] for w in weights[:-1]]
    n_output  = weights[-1].shape[0]

    return {
        "n_layers": n_layers,
        "sizes": [n_input] + hidden + [n_output],
        "weights": [w.tolist() for w in weights],
        "biases":  [b.tolist() for b in biases],
    }


# ─────────────────────────────────────────────
# 2. Génération du HTML
# ─────────────────────────────────────────────

def generate_html(net: dict, source_name: str) -> str:
    nn_json = json.dumps(net)
    sizes   = net["sizes"]
    arch_label = " × ".join(str(s) for s in sizes)
    n_layers_label = f"{net['n_layers']} couche{'s' if net['n_layers'] > 1 else ''} cachée{'s' if net['n_layers'] > 1 else ''}"

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Réseau de neurones — {arch_label}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #ffffff;
    --bg2:       #f5f4f0;
    --text:      #1a1a1a;
    --text2:     #555;
    --border:    rgba(0,0,0,0.12);
    --border2:   rgba(0,0,0,0.22);
    --pos:       #185FA5;
    --neg:       #BA7517;
    --radius:    8px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:      #1a1a1a;
      --bg2:     #252525;
      --text:    #e8e6de;
      --text2:   #999;
      --border:  rgba(255,255,255,0.1);
      --border2: rgba(255,255,255,0.22);
      --pos:     #85B7EB;
      --neg:     #F9CB42;
    }}
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    padding: 24px 20px 40px;
    min-height: 100vh;
  }}

  header {{
    margin-bottom: 20px;
  }}
  header h1 {{
    font-size: 17px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 3px;
  }}
  header p {{
    font-size: 12px;
    color: var(--text2);
  }}

  #controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    margin-bottom: 14px;
    font-size: 13px;
    color: var(--text2);
  }}
  #controls label {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  #controls input[type=range] {{
    width: 130px;
    accent-color: var(--pos);
  }}

  #canvas-wrap {{
    width: 100%;
    border-radius: var(--radius);
    overflow: hidden;
    border: 0.5px solid var(--border);
  }}
  canvas {{
    display: block;
    width: 100%;
    cursor: default;
  }}

  #legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    margin-top: 10px;
    font-size: 12px;
    color: var(--text2);
  }}
  #legend span {{ display: flex; align-items: center; gap: 5px; }}
  .l-line {{ width: 22px; height: 3px; border-radius: 2px; }}
  .l-dot  {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  #tooltip {{
    position: fixed;
    pointer-events: none;
    opacity: 0;
    background: var(--bg);
    border: 0.5px solid var(--border2);
    border-radius: var(--radius);
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.7;
    color: var(--text);
    transition: opacity 0.1s;
    z-index: 999;
    max-width: 220px;
    white-space: nowrap;
  }}
</style>
</head>
<body>

<header>
  <h1>Réseau de neurones — {arch_label}</h1>
  <p>{source_name} &nbsp;·&nbsp; {n_layers_label} &nbsp;·&nbsp; {sizes[0]} entrées, {", ".join(str(s) for s in sizes[1:-1])} neurones cachés, {sizes[-1]} sorties</p>
</header>

<div id="controls">
  <label>
    Seuil de visibilité
    <input type="range" id="threshold" min="0" max="95" value="0" step="1">
    <span id="threshold-val">0 %</span>
  </label>
  <label><input type="checkbox" id="show-bias" checked> Biais</label>
  <label><input type="checkbox" id="show-labels" checked> Labels</label>
</div>

<div id="canvas-wrap">
  <canvas id="nn-canvas"></canvas>
</div>

<div id="legend">
  <span><span class="l-line" style="background:var(--pos)"></span> Poids positif</span>
  <span><span class="l-line" style="background:var(--neg)"></span> Poids négatif</span>
  <span><span class="l-dot" style="background:var(--pos)"></span> Biais &gt; 0</span>
  <span><span class="l-dot" style="background:var(--neg)"></span> Biais &lt; 0</span>
  <span><span class="l-dot" style="background:var(--text2);opacity:.5"></span> Biais ≈ 0</span>
</div>

<div id="tooltip"></div>

<script>
const NN = {nn_json};

const canvas  = document.getElementById('nn-canvas');
const ctx     = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

const SIZES   = NN.sizes;        // [n_in, ...hidden..., n_out]
const N_LAYER = SIZES.length;    // nombre total de couches (entrée incluse)
const MAX_STR = 5;               // épaisseur max d'un trait en px

let hoveredNode = null;
let hoveredEdge = null;
let threshold   = 0;
let showBias    = true;
let showLabels  = true;
let edgeList    = [];

// ── Helpers ──────────────────────────────────────────────────────────────────

function isDark() {{
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}}

function cssVar(name) {{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}}

function hexToRgb(hex) {{
  hex = hex.replace('#','');
  if (hex.length === 3) hex = hex.split('').map(c=>c+c).join('');
  return [parseInt(hex.slice(0,2),16), parseInt(hex.slice(2,4),16), parseInt(hex.slice(4,6),16)];
}}

function distToSegment(px, py, x1, y1, x2, y2) {{
  const dx = x2-x1, dy = y2-y1, len2 = dx*dx+dy*dy;
  if (len2===0) return Math.hypot(px-x1, py-y1);
  const t = Math.max(0, Math.min(1, ((px-x1)*dx+(py-y1)*dy)/len2));
  return Math.hypot(px-(x1+t*dx), py-(y1+t*dy));
}}

// ── Layout ───────────────────────────────────────────────────────────────────

function getDims() {{
  const maxN = Math.max(...SIZES);
  const w    = canvas.clientWidth || 900;
  const h    = Math.max(480, maxN * 28 + 100);
  return {{ w, h }};
}}

function setup() {{
  const dpr = window.devicePixelRatio || 1;
  const {{ w, h }} = getDims();
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  return {{ w, h }};
}}

function getPositions(w, h) {{
  const margin = 0.08;
  return SIZES.map((n, li) => {{
    const xFrac = N_LAYER === 1 ? 0.5 : margin + (li / (N_LAYER-1)) * (1 - 2*margin);
    const x     = w * xFrac;
    const spacing = h / (n + 1);
    return Array.from({{length: n}}, (_, i) => ({{ x, y: spacing*(i+1) }}));
  }});
}}

// ── Max absolute weight per layer transition ──────────────────────────────────

function getMaxWeights() {{
  return NN.weights.map(W =>
    Math.max(...W.map(row => Math.max(...row.map(Math.abs))))
  );
}}

// ── Edge list ─────────────────────────────────────────────────────────────────

function buildEdges(pos, maxW, thresholds) {{
  const edges = [];
  NN.weights.forEach((W, wi) => {{
    const fromLayer = pos[wi];
    const toLayer   = pos[wi+1];
    const nFrom = fromLayer.length, nTo = toLayer.length;
    const thresh = thresholds[wi];
    for (let j = 0; j < nTo; j++) {{
      for (let i = 0; i < nFrom; i++) {{
        const w = W[j][i];
        if (Math.abs(w) < thresh) continue;
        edges.push({{
          x1: fromLayer[i].x, y1: fromLayer[i].y,
          x2: toLayer[j].x,   y2: toLayer[j].y,
          w, maxW: maxW[wi],
          fromLayer: wi, fromIdx: i,
          toLayer: wi+1, toIdx: j,
          label: `${{layerName(wi, i)}} → ${{layerName(wi+1, j)}}`
        }});
      }}
    }}
  }});
  return edges;
}}

function layerName(li, ni) {{
  if (li === 0)          return `x${{ni+1}}`;
  if (li === N_LAYER-1)  return `y${{ni+1}}`;
  return `h${{li}}.${{ni+1}}`;
}}

// ── Draw ──────────────────────────────────────────────────────────────────────

function draw() {{
  const {{ w, h }} = setup();
  const pos     = getPositions(w, h);
  const maxW    = getMaxWeights();
  const thresh  = maxW.map(m => (threshold/100)*m);
  edgeList = buildEdges(pos, maxW, thresh);

  ctx.clearRect(0, 0, w, h);

  const dark   = isDark();
  const posRGB = hexToRgb(dark ? '#85B7EB' : '#185FA5');
  const negRGB = hexToRgb(dark ? '#F9CB42' : '#BA7517');

  // Dessiner les arêtes
  edgeList.forEach((e, idx) => {{
    const absW  = Math.abs(e.w);
    const norm  = absW / e.maxW;
    const [r,g,b] = e.w > 0 ? posRGB : negRGB;
    const isHovEdge = hoveredEdge === idx;

    let alpha, sw;
    if (hoveredEdge !== null) {{
      alpha = isHovEdge ? Math.min(0.05 + norm*0.85, 1)*1.3 : (0.05+norm*0.85)*0.06;
      sw    = isHovEdge ? (0.3+norm*MAX_STR)*1.8 : 0.3+norm*MAX_STR;
    }} else if (hoveredNode !== null) {{
      const {{ li, ni }} = hoveredNode;
      const conn = (li===e.fromLayer && ni===e.fromIdx) || (li===e.toLayer && ni===e.toIdx);
      alpha = conn ? Math.min(0.05+norm*0.85,1)*1.2 : (0.05+norm*0.85)*0.07;
      sw    = conn ? (0.3+norm*MAX_STR)*1.3 : 0.3+norm*MAX_STR;
    }} else {{
      alpha = 0.05 + norm*0.85;
      sw    = 0.3  + norm*MAX_STR;
    }}

    ctx.beginPath();
    ctx.moveTo(e.x1, e.y1);
    ctx.lineTo(e.x2, e.y2);
    ctx.strokeStyle = `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
    ctx.lineWidth   = sw;
    ctx.stroke();
  }});

  // Dessiner les neurones
  pos.forEach((layer, li) => {{
    const R = Math.min(10, Math.max(4, (h/(layer.length+1))*0.27));
    layer.forEach((node, ni) => {{
      const bias = li > 0 ? NN.biases[li-1][ni] : null;
      const isHov = hoveredNode && hoveredNode.li===li && hoveredNode.ni===ni;
      const fade  = (hoveredNode || hoveredEdge !== null) && !isHov;

      // Couleur du noeud selon biais
      let fill;
      if (bias === null) {{
        fill = dark ? '#666' : '#bbb';
      }} else if (Math.abs(bias) < 0.1) {{
        fill = dark ? '#777' : '#aaa';
      }} else {{
        fill = bias > 0
          ? (dark ? '#85B7EB' : '#185FA5')
          : (dark ? '#F9CB42' : '#BA7517');
      }}

      ctx.beginPath();
      ctx.arc(node.x, node.y, R+(isHov?3:0), 0, Math.PI*2);
      ctx.fillStyle   = fill;
      ctx.globalAlpha = fade ? 0.28 : 0.88;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.15)';
      ctx.lineWidth   = isHov ? 2 : 0.5;
      ctx.stroke();

      // Biais textuel à l'intérieur si assez grand
      if (showBias && bias !== null && R >= 8) {{
        ctx.font         = `${{Math.max(8, R-1)}}px system-ui,sans-serif`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle    = dark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.85)';
        ctx.globalAlpha  = fade ? 0.22 : 0.75;
        ctx.fillText(bias.toFixed(2), node.x, node.y);
        ctx.globalAlpha  = 1;
      }}

      // Label du neurone
      if (showLabels) {{
        const label = layerName(li, ni);
        const lx    = li === 0 ? node.x - R - 5 : node.x + R + 5;
        ctx.font         = '11px system-ui,sans-serif';
        ctx.textAlign    = li === 0 ? 'right' : 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle    = dark ? 'rgba(200,200,200,0.65)' : 'rgba(60,60,60,0.65)';
        ctx.globalAlpha  = fade ? 0.18 : 0.8;
        ctx.fillText(label, lx, node.y);
        ctx.globalAlpha  = 1;
      }}
    }});
  }});

  // Labels de couche en bas
  const lnames = ['Entrées', ...Array.from({{length:N_LAYER-2}}, (_,i)=>`Cachée ${{i+1}}`), 'Sorties'];
  const lsizes = SIZES;
  ctx.font      = '11px system-ui,sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = dark ? 'rgba(170,170,170,0.65)' : 'rgba(90,90,90,0.65)';
  ctx.globalAlpha = 1;
  const ly = h - 8;
  pos.forEach((layer, li) => {{
    ctx.fillText(`${{lnames[li]}} (${{lsizes[li]}})`, layer[Math.floor(layer.length/2)].x, ly);
  }});
}}

// ── Hit tests ─────────────────────────────────────────────────────────────────

function hitNode(mx, my) {{
  const {{ w, h }} = getDims();
  const pos = getPositions(w, h);
  const R = 14;
  for (let li=0; li<pos.length; li++)
    for (let ni=0; ni<pos[li].length; ni++) {{
      const n = pos[li][ni];
      if ((mx-n.x)**2 + (my-n.y)**2 <= R*R) return {{ li, ni }};
    }}
  return null;
}}

function hitEdge(mx, my) {{
  for (let i=edgeList.length-1; i>=0; i--) {{
    const e = edgeList[i];
    const sw  = 0.3 + (Math.abs(e.w)/e.maxW)*MAX_STR;
    const hit = Math.max(5, sw*0.5+2);
    if (distToSegment(mx, my, e.x1, e.y1, e.x2, e.y2) <= hit) return i;
  }}
  return null;
}}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTip(ev, html) {{
  tooltip.innerHTML = html;
  tooltip.style.opacity = '1';
  tooltip.style.left = (ev.clientX+14)+'px';
  tooltip.style.top  = (ev.clientY-10)+'px';
}}

function hideTip() {{
  tooltip.style.opacity = '0';
}}

canvas.addEventListener('mousemove', ev => {{
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const my = ev.clientY - rect.top;
  const nh = hitNode(mx, my);

  if (nh) {{
    hoveredNode = nh; hoveredEdge = null;
    canvas.style.cursor = 'pointer';
    const {{ li, ni }} = nh;
    const name  = layerName(li, ni);
    const lname = li===0 ? 'Entrée' : li===N_LAYER-1 ? 'Sortie' : `Cachée ${{li}}`;
    let html = `<strong>${{lname}} — ${{name}}</strong><br>`;
    if (li > 0) {{
      const b = NN.biases[li-1][ni];
      html += `Biais : ${{b > 0 ? '+' : ''}}${{b.toFixed(5)}}<br>`;
    }}
    // Top 3 poids entrants (sauf couche d'entrée)
    if (li > 0) {{
      const W = NN.weights[li-1];
      const top = W[ni].map((v,i) => ({{v,i}})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,3);
      html += `Top 3 entrants :<br>`;
      top.forEach(({{v,i}}) => {{
        const c = v > 0 ? 'var(--pos)' : 'var(--neg)';
        html += `&nbsp;${{layerName(li-1,i)}} : <span style="color:${{c}};font-weight:500">${{v>0?'+':''}}${{v.toFixed(4)}}</span><br>`;
      }});
    }}
    // Top 3 poids sortants (sauf couche de sortie)
    if (li < N_LAYER-1) {{
      const W = NN.weights[li];
      const top = W.map((row,j) => ({{v:row[ni],j}})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,3);
      html += `Top 3 sortants :<br>`;
      top.forEach(({{v,j}}) => {{
        const c = v > 0 ? 'var(--pos)' : 'var(--neg)';
        html += `&nbsp;→ ${{layerName(li+1,j)}} : <span style="color:${{c}};font-weight:500">${{v>0?'+':''}}${{v.toFixed(4)}}</span><br>`;
      }});
    }}
    showTip(ev, html);
  }} else {{
    const eh = hitEdge(mx, my);
    if (eh !== null) {{
      hoveredEdge = eh; hoveredNode = null;
      canvas.style.cursor = 'crosshair';
      const e = edgeList[eh];
      const c = e.w > 0 ? 'var(--pos)' : 'var(--neg)';
      showTip(ev,
        `<strong>${{e.label}}</strong><br>` +
        `Poids : <span style="color:${{c}};font-weight:500">${{e.w>0?'+':''}}${{e.w.toFixed(6)}}</span>`
      );
    }} else {{
      hoveredNode = null; hoveredEdge = null;
      canvas.style.cursor = 'default';
      hideTip();
    }}
  }}
  draw();
}});

canvas.addEventListener('mouseleave', () => {{
  hoveredNode = null; hoveredEdge = null;
  hideTip(); draw();
}});

// ── Contrôles ─────────────────────────────────────────────────────────────────

document.getElementById('threshold').addEventListener('input', function() {{
  threshold = +this.value;
  document.getElementById('threshold-val').textContent = threshold + ' %';
  draw();
}});
document.getElementById('show-bias').addEventListener('change',   function() {{ showBias   = this.checked; draw(); }});
document.getElementById('show-labels').addEventListener('change', function() {{ showLabels = this.checked; draw(); }});

window.addEventListener('resize', draw);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

draw();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# 3. CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Génère un HTML de visualisation d'un réseau de neurones depuis un .npz"
    )
    parser.add_argument("input",  help="Fichier .npz source (W1/b1/W2/b2[/W3/b3])")
    parser.add_argument("-o", "--output", default=None,
                        help="Fichier HTML de sortie (défaut : même nom que l'entrée, extension .html)")
    args = parser.parse_args()

    src  = Path(args.input)
    dest = Path(args.output) if args.output else src.with_suffix(".html")

    if not src.exists():
        print(f"Erreur : fichier introuvable — {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement de {src} …")
    net = load_network(str(src))

    sizes = net["sizes"]
    arch  = " × ".join(str(s) for s in sizes)
    print(f"Architecture détectée : {arch}  ({net['n_layers']} couche(s) cachée(s))")
    print(f"  Poids : {sum(len(W)*len(W[0]) for W in net['weights'])} paramètres")
    print(f"  Biais : {sum(len(b) for b in net['biases'])} paramètres")

    html = generate_html(net, src.name)
    dest.write_text(html, encoding="utf-8")
    print(f"HTML généré → {dest}")


if __name__ == "__main__":
    main()