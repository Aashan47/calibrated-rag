"""Hand-written SVG charts (stdlib only) — a reliability diagram and an accuracy-vs-coverage
curve. SVG keeps the repo dependency-free and crisp on GitHub."""

from __future__ import annotations

W, H = 480, 360
L, R, T, B = 60, 20, 24, 48           # margins
PW, PH = W - L - R, H - T - B
INK, GRID, BAR, LINE, REF = "#1f2933", "#e4e7eb", "#4da3ff", "#2f855a", "#cbd2d9"


def _x(v):  # value in [0,1] -> px
    return L + v * PW


def _y(v):
    return T + (1 - v) * PH


def _axes(title: str, xlab: str, ylab: str) -> list[str]:
    s = [f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="16" text-anchor="middle" font-family="system-ui,sans-serif" '
         f'font-size="13" font-weight="600" fill="{INK}">{title}</text>']
    for i in range(6):
        g = i / 5
        s.append(f'<line x1="{_x(g):.1f}" y1="{_y(0):.1f}" x2="{_x(g):.1f}" y2="{_y(1):.1f}" stroke="{GRID}"/>')
        s.append(f'<line x1="{_x(0):.1f}" y1="{_y(g):.1f}" x2="{_x(1):.1f}" y2="{_y(g):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{_x(g):.1f}" y="{_y(0)+16:.1f}" text-anchor="middle" font-family="system-ui" font-size="10" fill="{INK}">{g:.1f}</text>')
        s.append(f'<text x="{L-8}" y="{_y(g)+3:.1f}" text-anchor="end" font-family="system-ui" font-size="10" fill="{INK}">{g:.1f}</text>')
    s.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-family="system-ui" font-size="11" fill="{INK}">{xlab}</text>')
    s.append(f'<text x="14" y="{T+PH/2}" text-anchor="middle" font-family="system-ui" font-size="11" fill="{INK}" transform="rotate(-90 14 {T+PH/2})">{ylab}</text>')
    return s


def reliability_svg(bins: list[tuple], ece: float) -> str:
    s = _axes(f"Reliability diagram (ECE = {ece:.3f})", "confidence", "accuracy")
    s.append(f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" stroke="{REF}" stroke-dasharray="4 3"/>')
    bw = PW / len(bins) * 0.72
    for mid, conf, acc, cnt in bins:
        if acc is None:
            continue
        x = _x(mid) - bw / 2
        s.append(f'<rect x="{x:.1f}" y="{_y(acc):.1f}" width="{bw:.1f}" height="{(_y(0)-_y(acc)):.1f}" '
                 f'fill="{BAR}" fill-opacity="0.85"/>')
    s.append(f'<text x="{_x(0.03):.1f}" y="{_y(0.95):.1f}" font-family="system-ui" font-size="10" fill="{REF}">dashed = perfect calibration</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d">%s</svg>' % (W, H, "".join(s))


def coverage_svg(points: list[tuple], chosen=None) -> str:
    """points: list of (coverage, selective_accuracy)."""
    s = _axes("Selective accuracy vs. coverage", "coverage (fraction answered)", "accuracy on answered")
    pts = sorted(points)
    path = " ".join(f"{_x(c):.1f},{_y(a):.1f}" for c, a in pts)
    s.append(f'<polyline points="{path}" fill="none" stroke="{LINE}" stroke-width="2"/>')
    for c, a in pts:
        s.append(f'<circle cx="{_x(c):.1f}" cy="{_y(a):.1f}" r="2.3" fill="{LINE}"/>')
    if chosen:
        c, a = chosen
        s.append(f'<circle cx="{_x(c):.1f}" cy="{_y(a):.1f}" r="5" fill="none" stroke="{BAR}" stroke-width="2"/>')
        s.append(f'<text x="{_x(c):.1f}" y="{_y(a)-9:.1f}" text-anchor="middle" font-family="system-ui" font-size="10" fill="{BAR}">calibrated</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d">%s</svg>' % (W, H, "".join(s))
