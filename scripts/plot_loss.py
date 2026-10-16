"""Turns a `rustgpt train` log into an SVG loss curve (no dependencies)."""
import re
import sys

log, out = sys.argv[1], sys.argv[2]
train, val = [], []
for line in open(log):
    if m := re.match(r"step\s+(\d+)/\d+ \| loss ([\d.]+)", line):
        train.append((int(m[1]), float(m[2])))
    elif m := re.match(r"=== step (\d+): val loss ([\d.]+)", line):
        val.append((int(m[1]), float(m[2])))

# Smooth the noisy per-batch training loss with an EMA.
smooth, ema = [], None
for s, l in train:
    ema = l if ema is None else 0.8 * ema + 0.2 * l
    smooth.append((s, ema))

W, H = 720, 360
L, R, T, B = 56, 120, 48, 44
xmax = max(s for s, _ in train)
ymin, ymax = 1.0, 4.5
px = lambda s: L + (W - L - R) * s / xmax
py = lambda l: T + (H - T - B) * (ymax - min(max(l, ymin), ymax)) / (ymax - ymin)

def path(pts):
    return "M" + " L".join(f"{px(s):.1f},{py(l):.1f}" for s, l in pts)

parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">
<style>
  .bg {{ fill: #fcfcfb; }} .ink {{ fill: #0b0b0b; }} .ink2 {{ fill: #52514e; }}
  .grid {{ stroke: #e4e3df; }} .train {{ stroke: #2a78d6; }} .val {{ stroke: #eb6834; }} .dot {{ fill: #eb6834; }}
  .ring {{ stroke: #fcfcfb; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #1a1a19; }} .ink {{ fill: #f0efec; }} .ink2 {{ fill: #c3c2b7; }}
    .grid {{ stroke: #383835; }} .train {{ stroke: #3987e5; }} .val {{ stroke: #d95926; }} .dot {{ fill: #d95926; }}
    .ring {{ stroke: #1a1a19; }}
  }}
</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="ink" x="{L}" y="26" font-size="15" font-weight="600">Training a 0.8M-parameter GPT on Tiny Shakespeare</text>
<text class="ink2" x="{L}" y="42" font-size="12">Cross-entropy loss (nats per character), 4 CPU threads</text>''']
for y in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    parts.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke-width="1"/>')
    parts.append(f'<text class="ink2" x="{L-8}" y="{py(y)+4:.1f}" font-size="11" text-anchor="end">{y:.1f}</text>')
for s in range(0, xmax + 1, 500):
    parts.append(f'<text class="ink2" x="{px(s):.1f}" y="{H-B+18}" font-size="11" text-anchor="middle">{s}</text>')
parts.append(f'<text class="ink2" x="{(L+W-R)/2:.0f}" y="{H-8}" font-size="11" text-anchor="middle">training step</text>')
parts.append(f'<path class="train" d="{path(smooth)}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
parts.append(f'<path class="val" d="{path(val)}" fill="none" stroke-width="2" stroke-dasharray="5 4"/>')
for s, l in val:
    parts.append(f'<circle class="dot ring" cx="{px(s):.1f}" cy="{py(l):.1f}" r="4" stroke-width="2"/>')
# Direct labels at the line ends; text stays in ink colors.
ts, tl = smooth[-1]
vs, vl = val[-1]
ty, vy = py(tl), py(vl)
# Lower loss sits lower on screen; push the labels apart so they never overlap.
if abs(ty - vy) < 16:
    mid = (ty + vy) / 2
    ty, vy = (mid + 8, mid - 8) if tl < vl else (mid - 8, mid + 8)
parts.append(f'<text class="ink" x="{px(ts)+8:.1f}" y="{ty+4:.1f}" font-size="12">train {tl:.2f}</text>')
parts.append(f'<text class="ink" x="{px(vs)+8:.1f}" y="{vy+4:.1f}" font-size="12">val {vl:.2f}</text>')
# Legend.
lx = W - R - 230
parts.append(f'<line class="train" x1="{lx}" x2="{lx+20}" y1="70" y2="70" stroke-width="2"/>')
parts.append(f'<text class="ink2" x="{lx+26}" y="74" font-size="12">train (smoothed)</text>')
parts.append(f'<line class="val" x1="{lx+130}" x2="{lx+150}" y1="70" y2="70" stroke-width="2" stroke-dasharray="5 4"/>')
parts.append(f'<text class="ink2" x="{lx+156}" y="74" font-size="12">validation</text>')
parts.append("</svg>")
open(out, "w").write("\n".join(parts))
print(f"wrote {out}: {len(train)} train points, {len(val)} val points")
