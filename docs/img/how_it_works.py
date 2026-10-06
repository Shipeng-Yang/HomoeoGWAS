"""Generate the README 'How it works' schematic (light and dark SVG)."""
import math
from pathlib import Path

W, H = 1300, 680
DY = -70
SUB = {"A": "#2A9D8F", "B": "#E3A33B", "D": "#8467B5"}
THEMES = {
    "light": dict(text="#1F2328", muted="#59636E", card="#FFFFFF", border="#D1D9E0", lane1="#EEF6F5",
                  lane2="#F7F1FB", lane_border="#C9DDE0", accent="#1F6FEB", arrow="#8C959F", chip="#F6F8FA",
                  ok="#1A7F37", warn="#9A6700", info="#0969DA"),
    "dark": dict(text="#E6EDF3", muted="#9198A1", card="#161B22", border="#3D444D", lane1="#0F2A2A",
                 lane2="#211A2E", lane_border="#2F3B44", accent="#4493F8", arrow="#7D8590", chip="#21262D",
                 ok="#3FB950", warn="#D29922", info="#4493F8"),
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, t):
        self.t = t
        self.out = []

    def add(self, s):
        self.out.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, r=10, sw=1.2, dash=None):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{st}{da}/>')

    def text(self, x, y, s, size=13, fill=None, weight=400, anchor="start", italic=False):
        fill = fill or self.t["text"]
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                 f'text-anchor="{anchor}"{it}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color, sw=1.6, arrow=False):
        m = ' marker-end="url(#arr)"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{m}/>')

    def path(self, d, color, sw=1.6, arrow=False, fill="none"):
        m = ' marker-end="url(#arr)"' if arrow else ""
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{m}/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1.5):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"{st}/>')


def card(s, x, y, w, h, title, lines, title_size=14):
    t = s.t
    s.rect(x, y, w, h, t["card"], t["border"])
    s.text(x + 14, y + 24, title, size=title_size, weight=600)
    for i, ln in enumerate(lines):
        s.text(x + 14, y + 46 + i * 18, ln, size=12, fill=t["muted"])


def arrow(s, x1, y1, x2, y2):
    s.line(x1, y1, x2 - 2, y2, s.t["arrow"], arrow=True)


def chromosomes(s, x, y):
    for i, (k, c) in enumerate(SUB.items()):
        yy = y + i * 34
        s.text(x, yy + 13, k, size=13, weight=700, fill=c)
        s.rect(x + 18, yy, 92, 16, c, r=8)
        for j in range(5):
            s.rect(x + 30 + j * 16, yy + 3, 3, 10, s.t["card"], r=1)


def variance_bar(s, x, y, w):
    parts = [("A", 0.26, SUB["A"]), ("B", 0.37, SUB["B"]), ("D", 0.14, SUB["D"]), ("e", 0.23, s.t["border"])]
    cx = x
    for k, f, c in parts:
        ww = w * f
        s.rect(cx, y, ww, 18, c, r=0)
        s.text(cx + ww / 2, y + 13, k, size=11, weight=700, anchor="middle",
               fill="#FFFFFF" if k != "e" else s.t["muted"])
        cx += ww


def manhattan(s, x, y, w, h):
    import random
    rnd = random.Random(7)
    n = 90
    for i in range(n):
        sub = list(SUB.values())[(i * 3) // n]
        v = rnd.random() ** 3 * 0.55
        if i in (22, 58):
            v = 0.95
        s.circle(x + i * w / n, y + h - v * h, 1.9, sub)
    s.line(x, y + h + 3, x + w, y + h + 3, s.t["border"], sw=1)


def edge_glyph(s, cx, cy, k, r=17):
    cols = list(SUB.values()) + ["#D9534F"]
    if k == 2:
        pts = [(cx - r, cy), (cx + r, cy)]
    else:
        pts = [(cx + r * math.cos(2 * math.pi * i / k - math.pi / 2), cy + r * math.sin(2 * math.pi * i / k - math.pi / 2))
               for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            s.line(*pts[i], *pts[j], s.t["muted"], sw=1.4)
    for i, (px, py) in enumerate(pts):
        s.circle(px, py, 5.2, cols[i], s.t["card"], sw=1.5)


def build(theme):
    t = THEMES[theme]
    s = Svg(t)
    s.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">')
    s.add(f'<g transform="translate(0,{DY})">')
    s.add(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
          f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker></defs>')


    card(s, 30, 92, 200, 196, "Inputs", ["Genotypes (VCF / PLINK)", "Phenotypes", "Species YAML:", "  subgenome map",
                                         "Gene annotation +", "  homoeolog table", "  (interaction only)"])
    s.rect(30, 304, 200, 150, t["card"], t["border"])
    s.text(44, 328, "Split by subgenome", size=14, weight=600)
    chromosomes(s, 50, 344)
    s.text(44, 448, "homoeogwas split", size=11, fill=t["muted"], italic=True)
    arrow(s, 130, 288, 130, 304)

    lx, lw = 262, 988
    s.rect(lx, 92, lw, 228, t["lane1"], t["lane_border"], r=14)
    s.text(lx + 18, 118, "Workflow 1 · Subgenome-stratified mixed model", size=15, weight=700, fill=t["text"])
    s.text(lx + 18, 138, "How much of the trait does each subgenome explain, and where are the single-locus signals?",
           size=12, fill=t["muted"])

    s.rect(lx, 340, lw, 300, t["lane2"], t["lane_border"], r=14)
    s.text(lx + 18, 366, "Workflow 2 · Homoeolog interaction", size=15, weight=700)
    s.text(lx + 18, 386, "Do the copies of a gene on different subgenomes act together beyond their additive effects?",
           size=12, fill=t["muted"])

    s.path("M230,378 C246,378 246,206 262,206", t["arrow"], arrow=True)
    s.path("M230,410 C246,410 246,490 262,490", t["arrow"], arrow=True)

    y1 = 156
    card(s, 282, y1, 200, 140, "Per-subgenome GRMs", ["one kinship per subgenome", "K_A · K_B · K_D"])
    for i, c in enumerate(SUB.values()):
        for a in range(3):
            for b in range(3):
                op = 0.9 if a == b else 0.35
                s.add(f'<rect x="{300 + i * 56 + b * 13}" y="{y1 + 84 + a * 13}" width="12" height="12" fill="{c}" opacity="{op}"/>')
    arrow(s, 482, 226, 506, 226)
    card(s, 506, y1, 230, 140, "Multi-kernel REML", ["y = Xβ + u_A + u_B + u_D + ε", "u_S ~ N(0, σ²_S K_S)"])
    s.text(520, y1 + 104, "per-subgenome variance", size=12, fill=t["muted"])
    s.text(520, y1 + 122, "components", size=12, fill=t["muted"])
    arrow(s, 736, 226, 760, 226)
    card(s, 760, y1, 200, 140, "Per-SNP scan", ["leave-one-chromosome-out", "streaming CPU, parallel", "chunks; optional GPU"])
    arrow(s, 960, 200, 990, 200)
    arrow(s, 960, 262, 990, 262)
    s.rect(990, y1, 240, 64, t["card"], t["border"])
    s.text(1004, y1 + 22, "Variance partition (PVE)", size=13, weight=600)
    variance_bar(s, 1004, y1 + 34, 212)
    s.rect(990, y1 + 76, 240, 64, t["card"], t["border"])
    s.text(1004, y1 + 98, "Manhattan · QQ · λGC", size=13, weight=600)
    manhattan(s, 1004, y1 + 104, 212, 26)

    y2 = 404
    s.rect(282, y2, 200, 216, t["card"], t["border"])
    s.text(296, y2 + 24, "Homoeolog groups", size=14, weight=600)
    s.text(296, y2 + 44, "pair edges are the unit", size=12, fill=t["muted"])
    for i, (k, lab) in enumerate(((2, "2 copies → 1 edge"), (3, "3 copies → 3 edges"), (4, "4 copies → 6 edges"))):
        yy = y2 + 80 + i * 50
        edge_glyph(s, 320, yy, k)
        s.text(352, yy + 4, lab, size=12, fill=t["text"])
    arrow(s, 482, 512, 506, 512)

    s.rect(506, y2, 230, 216, t["card"], t["border"])
    s.text(520, y2 + 24, "Edge omniB", size=14, weight=600)
    s.text(520, y2 + 44, "three encodings per edge,", size=12, fill=t["muted"])
    s.text(520, y2 + 62, "combined by ACAT", size=12, fill=t["muted"])
    for i, lab in enumerate(("minor burden", "PC1", "kernel-Hadamard")):
        s.rect(520, y2 + 80 + i * 32, 140, 24, t["chip"], t["border"], r=12)
        s.text(590, y2 + 96 + i * 32, lab, size=12, anchor="middle")
    s.path(f"M660,{y2 + 92} C690,{y2 + 92} 690,{y2 + 124} 704,{y2 + 124}", t["arrow"])
    s.path(f"M660,{y2 + 156} C690,{y2 + 156} 690,{y2 + 124} 704,{y2 + 124}", t["arrow"])
    s.line(660, y2 + 124, 702, y2 + 124, t["arrow"], arrow=True)
    s.text(708, y2 + 128, "p", size=13, weight=700, italic=True)
    s.text(520, y2 + 196, "group p = ACAT over its edges", size=12, fill=t["muted"])
    arrow(s, 736, 512, 760, 512)

    s.rect(760, y2, 200, 216, t["card"], t["border"])
    s.text(774, y2 + 24, "One calibrated family", size=14, weight=600)
    for i, ln in enumerate(("all edges / groups in a", "single experiment-wide", "bootstrap-minP family", "",
                             "kinship-preserving null;", "smooth_pc4 residual", "variance (default)")):
        s.text(774, y2 + 46 + i * 18, ln, size=12, fill=t["muted"])
    arrow(s, 960, 512, 990, 512)
    s.rect(990, y2, 240, 216, t["card"], t["border"])
    s.text(1004, y2 + 24, "FWER-controlled discoveries", size=13, weight=600)
    for i, ln in enumerate(("edge or group, as declared", "components localise the", "signal; they are not extra", "discoveries")):
        s.text(1004, y2 + 46 + i * 18, ln, size=12, fill=t["muted"])
    yy = y2 + 150
    for i in range(14):
        hgt = [8, 12, 6, 10, 14, 7, 9, 44, 11, 6, 13, 8, 30, 9][i]
        c = t["accent"] if hgt > 25 else t["border"]
        s.rect(1010 + i * 15, yy + 44 - hgt, 10, hgt, c, r=2)
    s.line(1004, yy + 14, 1222, yy + 14, t["warn"], sw=1.2)
    s.text(1222, yy + 10, "FWER 5%", size=10, fill=t["warn"], anchor="end")

    yb = 662
    s.path(f"M1230,{y1 + 32} L1272,{y1 + 32} L1272,{yb - 2}", t["arrow"], arrow=True)
    s.line(1230, y1 + 108, 1272, y1 + 108, t["arrow"])
    s.line(1110, y2 + 216, 1110, yb - 2, t["arrow"], arrow=True)
    s.rect(30, yb, 1255, 76, t["card"], t["border"], r=14)
    s.text(48, yb + 28, "Evidence audit", size=15, weight=700)
    s.text(48, yb + 50, "homoeogwas audit labels every result", size=12, fill=t["muted"])
    chips = (("Computational validity", t["ok"]), ("Internal discovery", t["info"]), ("Replication required", t["warn"]))
    for i, (lab, c) in enumerate(chips):
        x = 330 + i * 200
        s.rect(x, yb + 22, 184, 32, t["chip"], c, r=16, sw=1.6)
        s.circle(x + 18, yb + 38, 5, c)
        s.text(x + 32, yb + 43, lab, size=12, weight=600)
    s.text(950, yb + 30, "Run it via", size=12, fill=t["muted"])
    s.text(950, yb + 50, "CLI + YAML  ·  MCP server  ·  AI-agent skill", size=12, weight=600)
    s.add("</g></svg>")
    return "\n".join(s.out)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for th in THEMES:
        (here / f"how_it_works_{th}.svg").write_text(build(th) + "\n")
