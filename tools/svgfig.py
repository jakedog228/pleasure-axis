"""Minimal inline-SVG chart helpers for the report (colors via CSS variables; <title> tooltips on every mark).

Roles used (defined in the report's CSS): --ink, --ink2, --muted, --grid, --axis, --s1..--s8 (categorical, fixed order),
--div-neg, --div-mid, --div-pos (diverging).
"""
import html
import math


def esc(s):
    return html.escape(str(s), quote=True)


class Fig:
    def __init__(self, w=640, h=360, ml=56, mr=16, mt=28, mb=44, title=None):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.parts = []
        self.title = title

    # coordinate helpers
    def set_scale(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def X(self, x):
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.ml - self.mr)

    def Y(self, y):
        return self.h - self.mb - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.mt - self.mb)

    def add(self, s):
        self.parts.append(s)

    def axes(self, xticks, yticks, xlabel="", ylabel="", xfmt=str, yfmt=str, zero=None):
        for t in yticks:
            y = self.Y(t)
            self.add(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
            self.add(f'<text x="{self.ml - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{esc(yfmt(t))}</text>')
        for t in xticks:
            x = self.X(t)
            self.add(f'<text x="{x:.1f}" y="{self.h - self.mb + 16}" class="tick" text-anchor="middle">{esc(xfmt(t))}</text>')
        self.add(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.h - self.mb}" y2="{self.h - self.mb}" class="axis"/>')
        if zero is not None and self.y0 < zero < self.y1:
            self.add(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.Y(zero):.1f}" y2="{self.Y(zero):.1f}" class="axis"/>')
        if xlabel:
            self.add(f'<text x="{(self.ml + self.w - self.mr) / 2:.1f}" y="{self.h - 8}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            self.add(f'<text transform="translate(14,{(self.mt + self.h - self.mb) / 2:.1f}) rotate(-90)" class="lab" text-anchor="middle">{esc(ylabel)}</text>')

    def line(self, xs, ys, cls, name="", tips=None, dots=True):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, ys) if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if len(pts) > 1:
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            self.add(f'<path d="{d}" class="ln {cls}"><title>{esc(name)}</title></path>')
        if dots:
            for i, (x, y) in enumerate(zip(xs, ys)):
                if y is None or (isinstance(y, float) and math.isnan(y)):
                    continue
                tip = tips[i] if tips else f"{name}: {x} → {y:.2f}"
                self.add(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="4" class="dot {cls}"><title>{esc(tip)}</title></circle>')

    def dot(self, x, y, cls, tip, r=4.5):
        self.add(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" class="dot {cls}"><title>{esc(tip)}</title></circle>')

    def errbar(self, x, lo, hi, cls):
        self.add(f'<line x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{self.Y(lo):.1f}" y2="{self.Y(hi):.1f}" class="err {cls}"/>')

    def text(self, x, y, s, cls="lab", anchor="start", dx=0, dy=0):
        self.add(f'<text x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>')

    def legend(self, items, x=None, y=None):
        """items: [(cls, label)] drawn as a row at the top."""
        x = self.ml if x is None else x
        y = 14 if y is None else y
        for cls, lab in items:
            self.add(f'<line x1="{x}" x2="{x + 16}" y1="{y - 4}" y2="{y - 4}" class="ln {cls}"/>'
                     f'<circle cx="{x + 8}" cy="{y - 4}" r="3.5" class="dot {cls}"/>'
                     f'<text x="{x + 22}" y="{y}" class="tick">{esc(lab)}</text>')
            x += 30 + 7 * len(lab)

    def svg(self, aria=""):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" class="fig" role="img" aria-label="{esc(aria or self.title or "")}">'
                + "".join(self.parts) + "</svg>")


def heatmap(M, rows, cols, title="", vmax=0.6, cell=56, fmt="{:+.2f}", tip=None):
    """Diverging heatmap (blue = positive, red = negative, gray midpoint) with values printed in each cell."""
    ml, mt = 150, 34
    w, h = ml + cell * len(cols) + 10, mt + cell * len(rows) + 10
    p = []
    for j, c in enumerate(cols):
        p.append(f'<text x="{ml + cell * j + cell / 2}" y="{mt - 10}" class="tick" text-anchor="middle">{esc(c)}</text>')
    for i, r in enumerate(rows):
        p.append(f'<text x="{ml - 8}" y="{mt + cell * i + cell / 2 + 4}" class="tick" text-anchor="end">{esc(r)}</text>')
        for j, c in enumerate(cols):
            v = M[i][j]
            t = max(-1.0, min(1.0, v / vmax))
            var = "--div-pos" if t >= 0 else "--div-neg"
            op = abs(t)
            p.append(f'<rect x="{ml + cell * j + 1}" y="{mt + cell * i + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" '
                     f'style="fill:var(--div-mid)"/>'
                     f'<rect x="{ml + cell * j + 1}" y="{mt + cell * i + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" '
                     f'style="fill:var({var});fill-opacity:{op:.2f}"><title>{esc(tip(r, c, v) if tip else f"{r} × {c}: {v:+.2f}")}</title></rect>'
                     f'<text x="{ml + cell * j + cell / 2}" y="{mt + cell * i + cell / 2 + 4}" class="cellv" text-anchor="middle">{esc(fmt.format(v))}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="fig" role="img" aria-label="{esc(title)}">' + "".join(p) + "</svg>"
