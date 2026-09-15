"""Regenerate Figure 1 (UMRO-5G architecture) and Figure 2 (3GPP/ETSI NFV entity
and reference-point mapping) for the revised Section 6.

SOURCE OF TRUTH: ``work/text_architecture.md``, blocks F1.0-F1.8 (Figure 1) and
F2.0-F2.4 (Figure 2).  That specification is binding: box identifiers, exact box
labels, provenance tags, connector reference points, the region partition, the
loop membership lists and the colour discipline are all taken from it verbatim.

The entity data below is declarative on purpose.  A reader checking the figure
against the specification should be able to read the BOXES / CONNS / LOOPS
tables and never look at the rendering code.  ``check_against_spec()`` (run at
the bottom of this module) parses the specification tables out of the markdown
file and asserts that every box identifier and every reference point named there
is present in the declarative data.

No title text is drawn inside either image: the titles live in the caption
paragraphs of the manuscript (REWRITE 9 and REWRITE 15).

Deliberate geometry, do not "correct" it:
  * the vertical order of the stack is Layer 1, Layer 2, Layer 4 (bottom to top);
  * Layer 3 is a lateral cross-cutting plane, not a tier of the stack;
  * the I24 corridor runs down the left margin, on the opposite side of the
    stack from the Layer 3 plane.

Requires only matplotlib (>= 3.8).  Run:  python make_architecture.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, PathPatch,
                                Rectangle, Polygon)
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC = os.path.join(ROOT, "work", "text_architecture.md")

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# --------------------------------------------------------------------------
# F1.7 / F2.4  provenance palette.  Framework hues are desaturated so that the
# three saturated control-loop hues (red / orange / blue) stay dominant.  The
# palette is an Okabe-Ito derivative and is safe under deuteranopia and
# protanopia; every framework is additionally distinguished by its badge text
# and every loop by its line style and marker shape, never by colour alone.
# --------------------------------------------------------------------------
PALETTE = {
    # key        : (fill,      edge,      badge text)
    "3gpp":       ("#DCE6F1", "#3C5A78", "3GPP"),
    "etsinfv":    ("#DDEDE4", "#3F7A60", "ETSI NFV"),
    "oran":       ("#EEE3F0", "#6A4E7C", "O-RAN"),
    "umro":       ("#EDEDED", "#6E6E6E", "UMRO-5G"),
    "external":   ("#FFFFFF", "#5A5A5A", "3GPP"),
}
BAND_FILL = {"L1": "#F3F5F8", "L2": "#F2F6F3", "L4": "#F6F3F8", "L3": "#F7F4EF"}
BAND_EDGE = "#9AA3AC"
INK = "#1A1A1A"
GREY = "#585858"

LOOPS = {
    # id      : (colour,   linestyle,            marker, name,        band)
    "fast":   ("#C1272D", "solid",                "^",   "Fast Loop",   "< 10 ms"),
    "medium": ("#E08214", (0, (7, 3.5)),         "s",   "Medium Loop", "10 ms - 1 s"),
    "slow":   ("#1F5FA9", (0, (1.6, 2.6)),       "o",   "Slow Loop",   "> 1 s to minutes"),
}
LOOP_TEXT = {
    "fast": ("Fast Loop (< 10 ms) - per-TTI PRB assignment, MCS selection, precoding, "
             "power control, HARQ. Entities: O-DU MAC scheduler, O-RU. "
             "Sets x_{b,k,n}, p_{b,k,n}, w_{b,k}. No management function participates."),
    "medium": ("Medium Loop (10 ms - 1 s) - xApp inter-cell coordination and slice-share "
               "re-adjustment over E2; NWDAF analytics driving PCF, SMF and AMF decisions "
               "over Nnwdaf; NFMF fault and performance management of NF instances; VNFM "
               "auto-scaling and healing. Sets z_s^(r) within slow-loop bounds, ABS and "
               "handover parameters, inference policy pi_s."),
    "slow": ("Slow Loop (> 1 s to minutes) - communication service and slice lifecycle "
             "(CSMF, NSMF, NSSMF); network service and VNF lifecycle (NFVO, VNFM, VIM); "
             "VNF placement and SFC embedding; rApp policy generation; ML model training "
             "and deployment. Sets y_{fn}, phi, xi_f, theta_s and the prices lambda^(r)."),
}

# Figure 1 canvas.  15 x 17.22 in at 300 dpi = 4500 x 5166 px.  Placed at 165 mm
# wide the print scale is 165 / (15 * 25.4) = 0.4331, so the smallest type used
# (14 pt) prints at 6.06 pt.  The canvas is taller than the 16:9 of F1.0 because
# the entity count of the respecified figure cannot be set legibly at 16:9: the
# stack is three bands deep and every box carries a label, a provenance
# identifier and, in Layer 3 and Layer 4, a timescale.  Widening the canvas at
# constant printed type size does not help, since the printed size of a glyph is
# fs * 165 mm / (FIG1_W * 25.4 mm); only moving content sideways would, and
# there is no free horizontal space left.  Height was recovered instead by
# setting the loop legend in three columns, by fitting the Layer 4 rows to their
# measured text height, and by widening compartment 4L until the provenance tag
# and the timescale of every 3GPP management function set on one line.
FIG1_W, FIG1_H = 15.0, 17.22
PT_W, PT_H = FIG1_W * 72.0, FIG1_H * 72.0

FS_TAG = 14.0      # provenance tags, connector labels, annotations   -> 6.06 pt
FS_BOX = 15.5      # box labels                                       -> 6.71 pt
FS_HDR = 18.5      # band headers                                     -> 8.01 pt
FS_TOK = 21.0      # region A layer tokens                            -> 9.10 pt
FS_CAP = 17.0      # interface capsule identifiers


# --------------------------------------------------------------------------
# declarative entity data
# --------------------------------------------------------------------------
@dataclass
class Box:
    id: str
    label: str
    tag: str
    region: str
    x: float
    y: float
    w: float
    h: float
    fw: str = "3gpp"
    loops: tuple = ()
    note: str = ""
    timescale: str = ""
    open_box: bool = False
    badge: str = ""
    fs: float = FS_BOX
    align: str = "center"


@dataclass
class Conn:
    id: str
    label: str
    src: str = ""          # "<box id>:<side>" or "" when pts is given
    dst: str = ""
    pts: tuple = ()        # explicit waypoints in axes coordinates
    style: str = "solid"
    arrow: str = "->"
    loop: str = ""
    label_xy: tuple | None = None
    rot: float = 0.0
    ha: str = "center"
    va: str = "center"
    chip: bool = True
    wrap: int = 0
    color: str = "#33424E"
    lw: float = 1.6
    fs: float = FS_TAG


# --------------------------------------------------------------------------
# generic drawing primitives
# --------------------------------------------------------------------------
_TP_CACHE: dict = {}


def text_pt(s: str, fs: float, bold: bool = False) -> float:
    """Rendered width of *s* in points, measured from the actual font."""
    key = (s, bold)
    if key not in _TP_CACHE:
        fp = FontProperties(family="Arial", size=100,
                            weight="bold" if bold else "normal")
        _TP_CACHE[key] = TextPath((0, 0), s, prop=fp).get_extents().width / 100.0
    return _TP_CACHE[key] * fs


def _wrap_lead_bold(text: str, width_frac: float, fs: float) -> list[str]:
    """Wrap *text* with its first line measured bold and the rest regular.  The
    loop legend sets the loop name in bold on the first line of its column; a
    plain wrap measures that line regular and lets it run past the column."""
    avail = width_frac * PT_W
    words, first = text.split(), ""
    while words and text_pt((first + " " + words[0]).strip(), fs, True) <= avail:
        first = (first + " " + words.pop(0)).strip()
    rest = _wrap(" ".join(words), width_frac, fs, 0) if words else []
    return [first] + rest


def _wrap(text: str, width_frac: float, fs: float, pad_pt: float = 10.0,
          bold: bool = False) -> list[str]:
    """Greedy wrap of *text* to a box width given in axes fractions."""
    avail = max(8.0, width_frac * PT_W - pad_pt)
    out: list[str] = []
    for para in text.split(chr(10)):
        words, line = para.split(), ""
        for w in words:
            cand = (line + " " + w).strip()
            if line and text_pt(cand, fs, bold) > avail:
                out.append(line)
                line = w
            else:
                line = cand
        out.append(line)
    return out or [""]


def draw_box(ax, b: Box):
    """Render one entity: rounded box, label, provenance tag (design principle
    5, never omitted), optional grey annotation, timescale, and one marker per
    control loop the entity belongs to."""
    fill, edge, _ = PALETTE[b.fw]
    if b.open_box:
        fill = "#FFFFFF"
    ax.add_patch(FancyBboxPatch(
        (b.x, b.y), b.w, b.h, boxstyle="round,pad=0,rounding_size=0.005",
        linewidth=1.6, edgecolor=edge, facecolor=fill,
        linestyle=(0, (5, 2.6)) if b.fw == "umro" else "-",
        mutation_aspect=PT_W / PT_H, zorder=6))

    rows: list[tuple] = [("lab", ln) for ln in _wrap(b.label, b.w, b.fs, 10, True)]
    rows += [("note", ln) for ln in _wrap(b.note, b.w, FS_TAG, 10)] if b.note else []
    tag_lines = _wrap(b.tag, b.w, FS_TAG, 10) if b.tag else []
    if b.timescale:
        one_line = (len(tag_lines) == 1 and
                    text_pt(tag_lines[0], FS_TAG) + text_pt(b.timescale, FS_TAG)
                    + 16.0 <= b.w * PT_W - 10.0)
        if one_line:
            rows.append(("tagts", (tag_lines[0], b.timescale)))
        else:
            rows += [("tag", ln) for ln in tag_lines]
            rows.append(("ts", b.timescale))
    else:
        rows += [("tag", ln) for ln in tag_lines]
    if b.badge:
        rows.append(("badge", b.badge))

    lh = {"lab": b.fs * 1.13, "note": FS_TAG * 1.13, "tag": FS_TAG * 1.13,
          "ts": FS_TAG * 1.13, "tagts": FS_TAG * 1.13,
          "badge": FS_TAG * 1.22}
    total = sum(lh[k] for k, _ in rows)
    y = b.y + b.h / 2 + total / 2 / PT_H
    xc = b.x + b.w / 2
    for kind, val in rows:
        y -= lh[kind] / PT_H
        yb = y + lh[kind] * 0.24 / PT_H
        if kind == "lab":
            ax.text(xc, yb, val, ha="center", va="baseline", fontsize=b.fs,
                    fontweight="bold", color=INK, zorder=8)
        elif kind == "note":
            ax.text(xc, yb, val, ha="center", va="baseline", fontsize=FS_TAG,
                    color=GREY, style="italic", zorder=8)
        elif kind == "badge":
            ax.text(xc, yb, val, ha="center", va="baseline", fontsize=FS_TAG,
                    color=edge, style="italic", fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=edge, linewidth=0.9))
        elif kind == "ts":
            ax.text(xc, yb, val, ha="center", va="baseline", fontsize=FS_TAG,
                    color="#2F2F2F", fontweight="bold", zorder=8)
        elif kind == "tagts":
            ax.text(b.x + 0.005, yb, val[0], ha="left", va="baseline",
                    fontsize=FS_TAG, color=edge, zorder=8)
            ax.text(b.x + b.w - 0.005, yb, val[1], ha="right", va="baseline",
                    fontsize=FS_TAG, color="#2F2F2F", fontweight="bold", zorder=8)
        else:
            ax.text(xc, yb, val, ha="center", va="baseline", fontsize=FS_TAG,
                    color=edge, zorder=8)

    for i, lp in enumerate(b.loops):
        col, _, mk, _, _ = LOOPS[lp]
        ax.plot([b.x + b.w - 0.006 - i * 0.010], [b.y + b.h - 0.0055],
                marker=mk, markersize=5.6, color=col, markeredgecolor="white",
                markeredgewidth=0.8, zorder=9, clip_on=False)



SIDES = {
    "l": lambda b: (b.x, b.y + b.h / 2),
    "r": lambda b: (b.x + b.w, b.y + b.h / 2),
    "t": lambda b: (b.x + b.w / 2, b.y + b.h),
    "b": lambda b: (b.x + b.w / 2, b.y),
}


def anchor(boxes: dict, ref: str) -> tuple[float, float]:
    bid, _, spec = ref.partition(":")
    b = boxes[bid]
    if "," in spec:                      # explicit fractional anchor "fx,fy"
        fx, fy = (float(v) for v in spec.split(","))
        return (b.x + fx * b.w, b.y + fy * b.h)
    return SIDES[spec or "r"](b)


def draw_conn(ax, boxes: dict, c: Conn):
    pts = list(c.pts)
    if c.src:
        pts.insert(0, anchor(boxes, c.src))
    if c.dst:
        pts.append(anchor(boxes, c.dst))

    col = LOOPS[c.loop][0] if c.loop else c.color
    ls = {"solid": "-", "dashed": (0, (6, 3)), "dotted": (0, (1.4, 2.4))}[c.style]
    path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle=c.arrow, mutation_scale=13, linewidth=c.lw,
        linestyle=ls, color=col, joinstyle="round", capstyle="round",
        shrinkA=0, shrinkB=0, zorder=7))

    if not c.label:
        return
    if c.label_xy is not None:
        lx, ly = c.label_xy
    else:
        mid = len(pts) // 2
        if len(pts) % 2 == 0:
            lx = (pts[mid - 1][0] + pts[mid][0]) / 2
            ly = (pts[mid - 1][1] + pts[mid][1]) / 2
        else:
            lx, ly = pts[mid]
    txt = "\n".join(_wrap(c.label, c.wrap / PT_W, c.fs, 0)) if c.wrap else c.label
    bbox = dict(boxstyle="round,pad=0.24", facecolor="white",
                edgecolor="none", alpha=0.92) if c.chip else None
    ax.text(lx, ly, txt, ha=c.ha, va=c.va, fontsize=c.fs, color=col,
            rotation=c.rot, rotation_mode="anchor", linespacing=1.15,
            bbox=bbox, zorder=10)


def compartment(ax, x0, y0, x1, y1, title, edge, ttl_lines=3):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.007",
        linewidth=1.7, edgecolor=edge, facecolor="white", alpha=0.75,
        mutation_aspect=PT_W / PT_H, zorder=4))
    lines = _wrap(title, x1 - x0, FS_TAG + 0.5, 14, bold=True)
    lh = (FS_TAG + 0.5) * 1.08 / PT_H
    y = y1 - 5.0 / PT_H
    for ln in lines:
        ax.text((x0 + x1) / 2, y, ln, ha="center", va="top",
                fontsize=FS_TAG + 0.5, fontweight="bold", color=edge, zorder=5)
        y -= lh
    return y


def rounded_loop(ax, pts, key, arrow_at=0.5, r=0.010):
    """F1.6 control-loop overlay: an open rounded contour routed through the
    gutters between entities, thin and unfilled so that it stays subordinate to
    the boxes it encloses.  A directional arrowhead marks the sense of the loop.
    """
    col, ls, _, _, _ = LOOPS[key]
    n = len(pts)
    verts, codes = [], []
    ar = PT_W / PT_H

    def _shift(p, q, d):
        dx, dy = q[0] - p[0], (q[1] - p[1]) / ar
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        f = min(d, L * 0.45) / L
        return (p[0] + dx * f, p[1] + dy * f * ar)

    for i in range(n):
        p_prev, p, p_next = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        a = _shift(p, p_prev, r)
        b = _shift(p, p_next, r)
        if i == 0:
            verts.append(a); codes.append(Path.MOVETO)
        else:
            verts.append(a); codes.append(Path.LINETO)
        verts.extend([p, b]); codes.extend([Path.CURVE3, Path.CURVE3])
    verts.append(verts[0]); codes.append(Path.LINETO)
    ax.add_patch(PathPatch(
        Path(verts, codes), facecolor="none", edgecolor=col, linewidth=2.2,
        linestyle=ls, zorder=5, capstyle="round"))

    # directional arrowhead on the segment holding index arrow_at
    i = int(arrow_at)
    p, q = pts[i], pts[(i + 1) % n]
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = (dx ** 2 + (dy / ar) ** 2) ** 0.5 or 1.0
    ux, uy = dx / L, dy / L
    ax.add_patch(FancyArrowPatch(
        (mx - ux * 0.004, my - uy * 0.004), (mx + ux * 0.004, my + uy * 0.004),
        arrowstyle="-|>", mutation_scale=19, linewidth=2.2, color=col,
        zorder=6, shrinkA=0, shrinkB=0))


# ==========================================================================
# FIGURE 1 -- geometry.  Vertical positions are given in points measured from
# the bottom of the canvas (PT_H points tall) and converted to axes fractions
# by YP(); horizontal positions are axes fractions and follow the region table
# of specification block F1.0:
#   A 0.024-0.060 layer name labels        B 0.064-0.128 I24 corridor
#   C 0.146-0.676 the Layer 1 / 2 / 4 stack
#   I23-I34 channel 0.676-0.716            D 0.716-0.900 Layer 3 plane
#   E 0.910-1.000 timescale ruler          margins: control-loop lanes
# ==========================================================================
# The vertical skeleton is written in points measured from the *base line* of
# the drawing, which sits Y_OFF points below the bottom edge of the canvas; the
# legend occupies the strip between the two.  YP() maps such a position to axes
# coordinates, YH() maps a length.  Keeping the two apart means the whole stack
# can be slid over the canvas by editing Y_OFF alone.
Y_OFF = 22.0


def YP(pt: float) -> float:
    return (pt - Y_OFF) / PT_H


def YH(pt: float) -> float:
    return pt / PT_H


# --- vertical skeleton, in points ----------------------------------------
# Every band height below is the measured height of its tallest text stack plus
# a padding of at least 7 pt; ``check_box_fit()`` at the bottom of this module
# re-derives the text heights from the font metrics and reports any box whose
# content comes within 5 pt of its border, so these numbers cannot silently rot.
PROV_Y, DIV_Y = 130.0, 140.0         # provenance key, footer rule
LEG_Y = 112.0                        # loop legend, first baseline of column 1
L1_B0, L1_B1 = 156.0, 298.0          # Layer 1 band
L1_Y, L1_H = 162.0, 130.0            # Layer 1 boxes
G1_0, G1_1 = 298.0, 332.0            # I12 capsule gutter
L2_B0, L2_B1 = 332.0, 691.0          # Layer 2 band
R2A_Y, R2A_H = 338.0, 60.0           # sub-row 2a, virtualization substrate
BRK_LBL, BRK_LINE = 409.0, 420.0     # user-plane SFC bracket
R2B_Y, R2B_H = 426.0, 76.0           # sub-row 2b, user-plane chain
N4_Y = 510.0                         # SMF-to-UPF N4 lane
NWD_LBL, NWD_FAN = 527.0, 539.0      # Nnwdaf label and fan
R2C_Y, R2C_H = 549.0, 74.0           # sub-row 2c, 5G core control plane
BUS_Y, BUS_LBL = 631.0, 642.0        # service-based bus line and its label
SLICE_TAB, SLICE_TAB2 = 674.0, 658.0  # slice-overlay tab
OVL_0, OVL_1 = 422.0, 687.0          # slice overlay, over sub-rows 2b and 2c
RIB_0, RIB_1 = 426.0, 621.0          # eMBB / URLLC / mMTC ribbons
# F1.2, third bullet: the Application Function is external to every band, so it
# is given a strip of its own between the Layer 2 and Layer 4 bands, tall enough
# for its label and its provenance tag; it overlapped both neighbours when it
# shared the 38 pt inter-band gutter.
G2_0, G2_1 = 691.0, 743.0            # Application Function strip
AF_Y, AF_H = 695.0, 44.0
L4_B0, L4_B1 = 743.0, 1165.0         # Layer 4 band
CMP_0, CMP_1 = 747.0, 1159.0         # compartments 4L / 4R
# Rows are shared by 4L and 4R so that the two cross-compartment reference
# points are horizontal inside the band (F1.3).  Row 4 is one line taller
# because NFMF carries the ETSI NFV "EM" equivalence badge.  The inter-row gaps
# are the label lanes: every gap is tall enough for the two-line reference-point
# chip it carries, so that no connector label lands on a box title.
R1_Y, R1_H = 1057.0, 60.0            # CSMF  and WIM
R2_Y, R2_H = 961.0, 60.0             # NSMF  and VIM
R3_Y, R3_H = 865.0, 60.0             # NSSMF and NFVO
R4_Y, R4_H = 753.0, 76.0             # NFMF  and VNFM
G_R1, G_R2, G_R3 = 36.0, 36.0, 36.0  # inter-row gaps, hosting label chips
G1_LBL, G2_LBL, G3_LBL = 1039.0, 943.0, 847.0
SMO_TOP, SMO_BRIDGE, SMO_BOT = 1201.0, 1171.0, 749.0
SMO_RIGHT_BOT = 877.0
L4X_Y, L4X_H = 1205.0, 44.0
PAN_0, PAN_1 = 156.0, 1233.0         # Layer 3 plane, vertical extent
PAN_ANN = 344.0                      # Layer 3 bottom annotation strip
# Layer 3 boxes: top edge and height of each, in points
L3_YH = {"L3-a": (1067.0, 76.0), "L3-b": (882.0, 178.0), "L3-c": (725.0, 146.0),
         "L3-d": (556.0, 76.0), "L3-e": (368.0, 166.0)}

# --- horizontal skeleton -------------------------------------------------
# Rotated marginal text is the scarcest resource in the drawing, so the two
# vertical corridors are laid out as explicit lanes: every rotated string owns
# one lane and the lane pitch (0.016 = 17.3 pt) is larger than the 14 pt type it
# carries.  Nothing is allowed to share a lane with anything else.
AX = 0.030                            # region A, rotated layer names
BX0, BX1 = 0.052, 0.124               # region B, I24 corridor
B_PRICE = 0.0575                      # lane 1: cross-layer price bus arrow
B_PRICE_L = 0.0655                    # lane 2: its lambda^(r) annotation
B_REF0, B_REF_D = 0.0815, 0.0160      # lanes 3-4: I24 reference-point list
B_ANN = 0.1135                        # lane 5: "does not traverse Layer 3"
BRANCH = 0.1395                       # I24 branch-arrow labels, inside the band
LOOP_ORANGE_L = 0.1270
CX0, CX1 = 0.132, 0.708               # region C, the stack
IX0 = 0.152                           # region C content, left edge
IX1 = 0.704                           # region C content, right edge
IX1_L2 = 0.650                        # Layer 2 content, right edge (ribbons beyond)
CHX0, CHX1 = 0.708, 0.746             # I23 / I34 channel
CH_CAP = 0.727                        # channel centre, interface capsules
CH_REF0, CH_REF_D = 0.7215, 0.0160    # channel lanes, reference-point lists
DX0, DX1 = 0.748, 0.888               # region D, Layer 3 plane
D_LANE = 0.7545                       # A1 route inside region D
D_LANE2, D_LANE3 = 0.7605, 0.7665     # R1 / model-artefact routes
D_LAB = 0.7625                        # rotated labels of the three routes
D_BX0, D_BX1 = 0.774, 0.884           # Layer 3 boxes
EX = 0.906                            # region E, timescale ruler axis
LOOP_BLUE_L = 0.010
LOOP_ORANGE_R, LOOP_BLUE_R = 0.894, 0.899

# sub-row 2b: the user-plane service function chain
B1X, B2X, B3X, B4X = 0.152, 0.272, 0.392, 0.560
B1W, B2W, B3W, B4W = 0.090, 0.090, 0.140, 0.090
# sub-row 2c: the 5G core control plane on the service-based bus
C2W, C2G = 0.0776, 0.0065
C2X = [IX0 + i * (C2W + C2G) for i in range(6)]
# Layer 4 compartments.  Compartment 4L is 322 pt wide because that is the
# width at which "NSSMF - Network Slice Subnet Management Function (RAN / CN /
# TN)" sets on two lines and at which the provenance tag and the timescale of
# NSMF - the widest pair - set on one line; both savings remove a text line from
# every row of the compartment and 48 pt from the height of the figure.
L4LX, L4LW = 0.142, 0.338            # compartment 4L
L4AX, L4AW = 0.176, 0.298            # 4L boxes (left inset holds the OSS/BSS brace)
BRACE_L, BRACE_T0, BRACE_TD = 0.1725, 0.1495, 0.0140   # OSS/BSS brace and label
MNS_X = 0.205                        # ProvMnS / PerfMnS / FaultMnS riser
JOIN_LBL_X = 0.350                   # 3GPP-to-ETSI join labels, own lane in 4L
GUT_X = 0.4955                       # gutter riser: AF service intent
GUT_LBL = 0.4885                     # its rotated label
L4RX, L4RW = 0.504, 0.202            # compartment 4R
L4BX, L4BW = 0.5215, 0.167           # 4R boxes, leaving rotated-label insets
L4R_LANE, L4R_RANE = 0.5115, 0.6985  # Or-Vi (to WIM) and Vi-Vnfm routing insets
ORVNFM_X, ORVI_X = 0.607, 0.560      # adjacent-row NFVO connectors inside 4R
RIB_X = 0.6555                       # first eMBB / URLLC / mMTC ribbon

BOXES1: list[Box] = [
    # ---------------- F1.1  Layer 1 -- Infrastructure ---------------------
    Box("L1-a", "O-RU Radio Units and Antenna Arrays (PRB pool)", "O-RAN.WG4",
        "C/Layer1", IX0, YP(L1_Y), 0.100, YH(L1_H), "oran", ("fast", "slow")),
    Box("L1-b", "Edge / Regional / Central Compute and Storage Hardware",
        "ETSI GS NFV 002 - NFVI hardware resources", "C/Layer1",
        0.260, YP(L1_Y), 0.212, YH(L1_H), "etsinfv", ("slow",),
        note="capacity C_n^c, C_n^m (Equations 27-28)"),
    Box("L1-c", "Transport Nodes: Fronthaul / Midhaul / Backhaul (optical, microwave)",
        "ETSI GS NFV 002 - NFVI network hardware", "C/Layer1",
        0.480, YP(L1_Y), 0.224, YH(L1_H), "etsinfv", ("slow",),
        note="capacity c_ij, delay d_ij (Equation 22)"),

    # ---------------- F1.2  Layer 2 -- Virtualization and Slicing ---------
    Box("L2-a", "NFVI Virtualization Layer - hypervisor / container runtime; "
        "virtual compute, storage and network", "ETSI GS NFV 002",
        "C/Layer2/2a", 0.204, YP(R2A_Y), 0.446, YH(R2A_H), "etsinfv", ("slow",)),

    Box("L2-b1", "O-DU", "O-RAN.WG1; 3GPP TS 38.401", "C/Layer2/2b",
        B1X, YP(R2B_Y), B1W, YH(R2B_H), "oran", ("fast", "medium")),
    Box("L2-b2", "O-CU-UP", "O-RAN.WG1; 3GPP TS 38.401", "C/Layer2/2b",
        B2X, YP(R2B_Y), B2W, YH(R2B_H), "oran", ("fast", "medium")),
    Box("L2-b3", "UPF - user-plane service function", "3GPP TS 23.501 clause 6.2",
        "C/Layer2/2b", B3X, YP(R2B_Y), B3W, YH(R2B_H), "3gpp", ("medium",)),
    Box("L2-b4", "Data Network", "3GPP TS 23.501", "C/Layer2/2b",
        B4X, YP(R2B_Y), B4W, YH(R2B_H), "external", (), open_box=True),

    Box("L2-c1", "AMF", "3GPP\nTS 23.501", "C/Layer2/2c",
        C2X[0], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",)),
    Box("L2-c2", "SMF", "3GPP\nTS 23.501", "C/Layer2/2c",
        C2X[1], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",)),
    Box("L2-c3", "PCF", "3GPP\nTS 23.501", "C/Layer2/2c",
        C2X[2], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",)),
    Box("L2-c5", "NWDAF", "3GPP\nTS 23.288", "C/Layer2/2c",
        C2X[3], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",)),
    Box("L2-c6", "NSSF / UDM / NRF", "3GPP\nTS 23.501", "C/Layer2/2c",
        C2X[4], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",), fs=FS_TAG),
    Box("L2-c4", "NEF", "3GPP\nTS 23.501", "C/Layer2/2c",
        C2X[5], YP(R2C_Y), C2W, YH(R2C_H), "3gpp", ("medium",)),

    # external consumer of Nnef (F1.2, third bullet).  Width 0.337 is the width
    # at which the label sets on one line, so that the box needs two text rows
    # and fits the strip it now owns.
    Box("AF", "Application Function / Vertical Tenant (external)",
        "3GPP TS 23.501, TS 29.522", "G2/external",
        0.150, YP(AF_Y), 0.337, YH(AF_H), "external", (), open_box=True),

    # ---------------- F1.3  Layer 4 -- Management and Orchestration -------
    Box("L4-x", "Communication Service Customer / Vertical", "3GPP TS 28.530",
        "above Layer4", 0.170, YP(L4X_Y), 0.310, YH(L4X_H), "external", ("slow",),
        open_box=True),

    Box("L4-a1", "CSMF - Communication Service Management Function",
        "3GPP TS 28.530, TS 28.533", "C/Layer4/4L",
        L4AX, YP(R1_Y), L4AW, YH(R1_H), "3gpp", ("slow",), timescale="minutes-hours"),
    Box("L4-a2", "NSMF - Network Slice Management Function",
        "3GPP TS 28.531, TS 28.533", "C/Layer4/4L",
        L4AX, YP(R2_Y), L4AW, YH(R2_H), "3gpp", ("slow",), timescale="tens of s - minutes"),
    Box("L4-a3", "NSSMF - Network Slice Subnet Management Function (RAN / CN / TN)",
        "3GPP TS 28.531, TS 28.541", "C/Layer4/4L",
        L4AX, YP(R3_Y), L4AW, YH(R3_H), "3gpp", ("slow",), timescale="seconds"),
    Box("L4-a4", "NFMF - Network Function Management Function",
        "3GPP TS 28.532, TS 28.533", "C/Layer4/4L",
        L4AX, YP(R4_Y), L4AW, YH(R4_H), "3gpp", ("medium",),
        timescale="100 ms - 1 s", badge='= ETSI NFV "EM"'),

    Box("L4-b4", "WIM - WAN Infrastructure Manager", "ETSI GS NFV-MAN 001",
        "C/Layer4/4R", L4BX, YP(R1_Y), L4BW, YH(R1_H), "etsinfv", ("slow",)),
    Box("L4-b3", "VIM - Virtualised Infrastructure Manager", "ETSI GS NFV-MAN 001",
        "C/Layer4/4R", L4BX, YP(R2_Y), L4BW, YH(R2_H), "etsinfv", ("slow",)),
    Box("L4-b1", "NFVO - NFV Orchestrator", "ETSI GS NFV-MAN 001",
        "C/Layer4/4R", L4BX, YP(R3_Y), L4BW, YH(R3_H), "etsinfv", ("slow",)),
    Box("L4-b2", "VNFM - VNF Manager", "ETSI GS NFV-MAN 001",
        "C/Layer4/4R", L4BX, YP(R4_Y), L4BW, YH(R4_H), "etsinfv", ("slow", "medium")),

    # ---------------- F1.4  Layer 3 -- Intelligence and Analytics ---------
    Box("L3-a", "Non-RT RIC + rApps", "O-RAN.WG2", "D/Layer3",
        D_BX0, YP(L3_YH["L3-a"][0]), D_BX1 - D_BX0, YH(L3_YH["L3-a"][1]),
        "oran", ("slow",), timescale="> 1 s"),
    Box("L3-b", "AI/ML Management: training, deployment, model lifecycle",
        "3GPP TS 28.105; O-RAN.WG2 AI/ML workflow", "D/Layer3",
        D_BX0, YP(L3_YH["L3-b"][0]), D_BX1 - D_BX0, YH(L3_YH["L3-b"][1]),
        "3gpp", ("slow",), timescale="> 1 s"),
    Box("L3-c", "MDAF - Management Data Analytics Function (MDAS producer)",
        "3GPP TS 28.104", "D/Layer3",
        D_BX0, YP(L3_YH["L3-c"][0]), D_BX1 - D_BX0, YH(L3_YH["L3-c"][1]),
        "3gpp", ("slow",), timescale="> 1 s"),
    Box("L3-d", "Near-RT RIC + xApps", "O-RAN.WG3", "D/Layer3",
        D_BX0, YP(L3_YH["L3-d"][0]), D_BX1 - D_BX0, YH(L3_YH["L3-d"][1]),
        "oran", ("medium",), timescale="10 ms - 1 s"),
    Box("L3-e", "Model Repository and Federated Learning Aggregator",
        "no standard counterpart (UMRO-5G)", "D/Layer3",
        D_BX0, YP(L3_YH["L3-e"][0]), D_BX1 - D_BX0, YH(L3_YH["L3-e"][1]),
        "umro", ("slow",), timescale="> 1 s"),
]
BOX1 = {b.id: b for b in BOXES1}


# --------------------------------------------------------------------------
# F1.2 / F1.3 / F1.4 / F1.5  connectors.  Every entry carries the reference
# point or interface that the specification puts on that connector; a
# connector with an empty label would be a defect, so `label` is mandatory.
# Entries with fs=0.0 are the further members of a bundle whose single label
# is drawn once on the bundle (Nnwdaf fan, Or-Vi pair, R1 pair, ProvMnS chain).
# --------------------------------------------------------------------------
X4L, X4R = 0.474, 0.5215               # facing edges of the two compartments
NNEF_X = 0.700                         # Nnef riser, clear of the slice ribbons
CH_RISER = 0.7115                      # I12-to-WIM riser inside the channel
FH_X = 0.197                           # Open Fronthaul riser, clear of L2-a
CONNS1: list[Conn] = [
    # ---- F1.2 user-plane service function chain --------------------------
    Conn("I12-fronthaul", "Open Fronthaul\nCUS-Plane (eCPRI)",
         pts=((FH_X, YP(L1_Y + L1_H)), (FH_X, YP(R2B_Y))), arrow="-|>",
         label_xy=(0.176, YP(358.0)), rot=90, chip=False, loop="fast"),
    Conn("F1-U", "F1-U", src="L2-b1:r", dst="L2-b2:l", arrow="-|>", loop="fast"),
    Conn("N3", "N3", src="L2-b2:r", dst="L2-b3:l", arrow="-|>"),
    Conn("N6", "N6", src="L2-b3:r", dst="L2-b4:l", arrow="-|>"),
    Conn("N4", "N4", src="L2-c2:b", dst="L2-b3:t", arrow="-|>",
         pts=((C2X[1] + C2W / 2, YP(N4_Y)), (B3X + B3W / 2, YP(N4_Y))),
         label_xy=(0.345, YP(N4_Y))),
    Conn("Nnwdaf", "Nnwdaf - analytics consumption (load, performance, service experience)",
         pts=((C2X[3] + C2W / 2, YP(R2C_Y)), (C2X[3] + C2W / 2, YP(NWD_FAN)),
              (C2X[0] + C2W / 2, YP(NWD_FAN)), (C2X[0] + C2W / 2, YP(R2C_Y))),
         arrow="-|>", label_xy=(0.400, YP(NWD_LBL))),
    Conn("Nnwdaf-b", "Nnwdaf", pts=((C2X[1] + C2W / 2, YP(NWD_FAN)),
                                    (C2X[1] + C2W / 2, YP(R2C_Y))),
         arrow="-|>", chip=False, fs=0.0),
    Conn("Nnwdaf-c", "Nnwdaf", pts=((C2X[2] + C2W / 2, YP(NWD_FAN)),
                                    (C2X[2] + C2W / 2, YP(R2C_Y))),
         arrow="-|>", chip=False, fs=0.0),
    Conn("Nnef", "Nnef - third-party / AF exposure\nand vertical requests",
         src="L2-c4:r", dst="AF:1,0.25",
         pts=((NNEF_X, YP(R2C_Y + R2C_H / 2)), (NNEF_X, YP(AF_Y + AF_H * 0.25))),
         arrow="-|>", label_xy=(0.600, YP(727.0))),
    Conn("service-intent", "service intent", src="AF:t", dst="L4-a1:r",
         pts=((GUT_X, YP(G2_1 - 2.0)), (GUT_X, YP(R1_Y + 13.0))),
         style="dotted", arrow="-|>", label_xy=(GUT_LBL, YP(1000.0)), rot=90),
    Conn("Vn-Nf", "Vn-Nf\n(ETSI GS NFV 002)",
         pts=((0.546, YP(R2A_Y + R2A_H)), (0.546, YP(R2B_Y))), arrow="<|-|>",
         label_xy=(0.615, YP(BRK_LBL)), chip=False),

    # ---- F1.3 Layer 4, 3GPP management hierarchy -------------------------
    Conn("service-order", "service order and intent (3GPP TS 28.530, TS 28.312)",
         src="L4-x:b", dst="L4-a1:t", arrow="-|>",
         label_xy=(0.345, YP(1183.0))),
    Conn("MnS-a1a2", "3GPP management services: ProvMnS, PerfMnS, FaultMnS "
         "(TS 28.532); legacy Itf-N",
         pts=((MNS_X, YP(R1_Y)), (MNS_X, YP(R2_Y + R2_H))), arrow="<|-|>",
         label_xy=(0.330, YP(G1_LBL)), wrap=330.0),
    Conn("MnS-a2a3", "ProvMnS / PerfMnS / FaultMnS (TS 28.532)",
         pts=((MNS_X, YP(R2_Y)), (MNS_X, YP(R3_Y + R3_H))), arrow="<|-|>", fs=0.0),
    Conn("MnS-a3a4", "ProvMnS / PerfMnS / FaultMnS (TS 28.532)",
         pts=((MNS_X, YP(R3_Y)), (MNS_X, YP(R4_Y + R4_H))), arrow="<|-|>", fs=0.0),

    # ---- F1.3 ETSI NFV MANO internal reference points --------------------
    # Adjacent rows are joined inside compartment 4R; the two connectors that
    # skip a row use the compartment's left and right insets, so that no
    # reference-point label has to cross the 4L / 4R gutter.
    Conn("Or-Vnfm", "Or-Vnfm (ETSI GS NFV-IFA 007)",
         pts=((ORVNFM_X, YP(R3_Y)), (ORVNFM_X, YP(R4_Y + R4_H))), arrow="<|-|>",
         label_xy=(ORVNFM_X, YP(G3_LBL)), wrap=170.0),
    Conn("Or-Vi-b3", "Or-Vi (ETSI GS NFV-IFA 005)",
         pts=((ORVI_X, YP(R3_Y + R3_H)), (ORVI_X, YP(R2_Y))), arrow="<|-|>",
         fs=0.0),
    Conn("Or-Vi-b4", "Or-Vi (ETSI GS NFV-IFA 005)",
         pts=((L4BX, YP(R3_Y + R3_H / 2)), (L4R_LANE, YP(R3_Y + R3_H / 2)),
              (L4R_LANE, YP(R1_Y + R1_H / 2)), (L4BX, YP(R1_Y + R1_H / 2))),
         arrow="<|-|>", label_xy=(0.600, YP(G1_LBL))),
    Conn("Vi-Vnfm", "Vi-Vnfm (ETSI GS NFV-IFA 006)",
         pts=((L4BX + L4BW, YP(R4_Y + R4_H / 2)),
              (L4R_RANE, YP(R4_Y + R4_H / 2)),
              (L4R_RANE, YP(R2_Y + R2_H / 2)),
              (L4BX + L4BW, YP(R2_Y + R2_H / 2))),
         arrow="<|-|>", label_xy=(L4R_RANE, YP(888.0)), rot=90, chip=True),

    # ---- F1.3 the 3GPP-to-ETSI join, drawn inside the Layer 4 band -------
    # F1.3 defect fix: both labels are set on two lines in the inter-row lane
    # of compartment 4L, clear of every box title, of the OSS/BSS brace and of
    # the gutter riser, instead of running as one line across the whole band.
    Conn("Os-Ma-nfvo", "Os-Ma-nfvo (ETSI GS NFV-IFA 013; RESTful binding "
         "ETSI GS NFV-SOL 005)",
         pts=((X4L, YP(R3_Y + R3_H / 2)), (X4R, YP(R3_Y + R3_H / 2))),
         arrow="<|-|>", label_xy=(JOIN_LBL_X, YP(G2_LBL)), wrap=265.0),
    Conn("Ve-Vnfm-em", "Ve-Vnfm-em (ETSI GS NFV-IFA 008)",
         pts=((X4L, YP(R4_Y + R4_H / 2)), (X4R, YP(R4_Y + R4_H / 2))),
         arrow="<|-|>", label_xy=(JOIN_LBL_X, YP(G3_LBL))),

    # ---- F1.4 intra-Layer-3 ---------------------------------------------
    Conn("A1", "A1 - policy and enrichment information (O-RAN.WG2)",
         pts=((D_BX0, YP(1110.0)), (D_LANE, YP(1110.0)),
              (D_LANE, YP(594.0)), (D_BX0, YP(594.0))), arrow="<|-|>",
         label_xy=(D_LAB, YP(735.0)), rot=90, chip=True),
    Conn("R1-ab", "R1 services (O-RAN.WG2)",
         pts=((0.829, YP(L3_YH["L3-a"][0])),
              (0.829, YP(L3_YH["L3-b"][0] + L3_YH["L3-b"][1]))),
         arrow="<|-|>", fs=0.0),
    Conn("R1-ac", "R1 services (O-RAN.WG2)",
         pts=((D_BX0, YP(1094.0)), (D_LANE2, YP(1094.0)),
              (D_LANE2, YP(860.0)), (D_BX0, YP(860.0))), arrow="<|-|>",
         label_xy=(D_LAB, YP(1010.0)), rot=90, chip=True),
    Conn("model-artefacts", "model artefacts",
         pts=((D_BX0, YP(934.0)), (D_LANE3, YP(934.0)),
              (D_LANE3, YP(440.0)), (D_BX0, YP(440.0))), arrow="<|-|>",
         style="solid", label_xy=(D_LAB, YP(460.0)), rot=90, chip=True),

    # ---- F1.5 interface bundles crossing the I23 / I34 channel -----------
    Conn("I23-d", "E2 (O-RAN.WG3)", pts=((CX1, YP(594.0)), (DX0, YP(594.0))),
         arrow="<|-|>", fs=0.0, lw=2.4),
    Conn("I23-a", "O1 (O-RAN.WG10); Nnwdaf analytics exposure (3GPP TS 23.288)",
         pts=((CX1, YP(470.0)), (DX0, YP(470.0))), arrow="<|-|>", fs=0.0, lw=2.4),
    Conn("I34-c", "MDA services (3GPP TS 28.104)",
         pts=((CX1, YP(829.0)), (DX0, YP(829.0))), arrow="<|-|>", fs=0.0, lw=2.4),
    Conn("I34-b", "R1 (O-RAN.WG2); AI/ML management services (3GPP TS 28.105)",
         pts=((CX1, YP(990.0)), (DX0, YP(990.0))), arrow="<|-|>", fs=0.0, lw=2.4),
    # I12 constituent that reaches the WIM of Layer 4 (F1.5, I12 endpoints)
    Conn("I12-wim", "I12",
         pts=((IX1, YP(232.0)), (CH_RISER, YP(232.0)),
              (CH_RISER, YP(R1_Y + 20)), (L4BX + L4BW, YP(R1_Y + 20))),
         arrow="-|>", style="dashed",
         label_xy=(CH_RISER, YP(710.0)), rot=90, fs=FS_TAG),
    # I12 constituents inside the stack: NFVI hardware to virtualization layer
    Conn("I12-vnnf-b", "Vn-Nf", pts=((0.366, YP(L1_Y + L1_H)), (0.366, YP(R2A_Y))),
         arrow="<|-|>", fs=0.0),
    Conn("I12-vnnf-c", "Vn-Nf", pts=((0.592, YP(L1_Y + L1_H)), (0.592, YP(R2A_Y))),
         arrow="<|-|>", fs=0.0),
]

# --------------------------------------------------------------------------
# F1.5  the four inter-layer interfaces.  `refs` is the reference-point list
# printed under the capsule; `endpoints` is the endpoint-entity list of the
# specification table, kept here so that the figure data can be checked
# against the specification without reading the drawing code.
# --------------------------------------------------------------------------
CAPSULES = {
    "I12": dict(
        endpoints="L1-a - L2-b1; L1-b / L1-c - L2-a; L1-c - L4-b4",
        refs="Open Fronthaul M-Plane and CUS-Plane (O-RAN.WG4); Vn-Nf "
             "(ETSI GS NFV 002); SDN southbound NETCONF/YANG, gNMI"),
    "I23": dict(
        endpoints="L2-b1 / L2-b2 - L3-d; L2-b* / L2-c* - L3-a; L2-c5 - L3-c",
        refs="E2 (O-RAN.WG3); O1 (O-RAN.WG10); Nnwdaf (3GPP TS 23.288)"),
    "I34": dict(
        endpoints="L3-a / L3-b / L3-c - L4-a2 / L4-a3 / L4-a4",
        refs="R1 (O-RAN.WG2); MDA services (3GPP TS 28.104); AI/ML management "
             "services (3GPP TS 28.105)"),
    "I24": dict(
        endpoints="L4-b2 - L2-b* / L2-c*; L4-b3 - L2-a and L1-b; L4-a4 - "
                  "L2-b* / L2-c*; SMO container - L1-b",
        refs="Ve-Vnfm-vnf (ETSI GS NFV-IFA 008); Nf-Vi (ETSI GS NFV-MAN 001); "
             "provisioning, performance and fault MnS over O1 "
             "(3GPP TS 28.532, O-RAN.WG10); O2 (O-RAN.WG6)"),
}

# --------------------------------------------------------------------------
# F1.6  control-loop overlays.  Each loop is an open rounded contour routed
# through the gutters between entities; the exact membership of each loop is
# carried by the per-box loop markers (draw_box), the contour carries the
# nesting.  Red is strictly inside orange, orange strictly inside blue.
# --------------------------------------------------------------------------
LOOP_PATHS = {
    "fast": [(0.1495, YP(154.0)), (0.2565, YP(154.0)), (0.2565, YP(296.0)),
             (0.196, YP(296.0)), (0.196, YP(422.0)), (0.372, YP(422.0)),
             (0.372, YP(506.0)), (0.1495, YP(506.0))],
    "medium": [(LOOP_ORANGE_L, YP(150.0)), (LOOP_ORANGE_L, YP(835.0)),
               (CHX0, YP(835.0)), (CHX0, YP(707.0)),
               (LOOP_ORANGE_R, YP(707.0)), (LOOP_ORANGE_R, YP(150.0))],
    "slow": [(LOOP_BLUE_L, YP(146.0)), (LOOP_BLUE_R, YP(146.0)),
             (LOOP_BLUE_R, YP(1255.0)), (LOOP_BLUE_L, YP(1255.0))],
}
LOOP_ARROW_SEG = {"fast": 5, "medium": 1, "slow": 2}

# F1.8  timescale ruler: (seconds, decade label, management function / loop)
RULER = [(1e-4, "100 us", ""), (1e-3, "1 ms", ""), (1e-2, "10 ms", "Fast"),
         (1e-1, "100 ms", ""), (1.0, "1 s", "NFMF"), (10.0, "10 s", "NSSMF"),
         (60.0, "1 min", "NSMF"), (600.0, "10 min", ""), (3600.0, "1 h", "CSMF")]
RULER_BANDS = [("fast", 1e-4, 1e-2), ("medium", 1e-2, 1.0), ("slow", 1.0, 3600.0)]


def ruler_y(t: float) -> float:
    import math
    lo, hi = -4.0, 3.5563
    f = (math.log10(t) - lo) / (hi - lo)
    return YP(200.0 + f * 950.0)


def draw_figure1(path: str) -> None:
    fig = plt.figure(figsize=(FIG1_W, FIG1_H), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    # ---------------- bands and region A layer names (F1.0 region A) ------
    for key, y0, y1, name in (
            ("L1", L1_B0, L1_B1, "Layer 1 - Infrastructure (hardware resources only)"),
            ("L2", L2_B0, L2_B1, "Layer 2 - Virtualization and Slicing"),
            ("L4", L4_B0, L4_B1, "Layer 4 - Management and Orchestration")):
        ax.add_patch(FancyBboxPatch(
            (CX0, YP(y0)), CX1 - CX0, YH(y1 - y0),
            boxstyle="round,pad=0,rounding_size=0.005", linewidth=1.4,
            edgecolor=BAND_EDGE, facecolor=BAND_FILL[key],
            mutation_aspect=PT_W / PT_H, zorder=2))
        # rotated band header: the run available is the band height, widened to
        # 200 pt so that the Layer 1 name sets on two lines
        lines = _wrap(name, max(y1 - y0, 200.0) / PT_W, 15.0, 8, True)
        lh = 15.0 * 1.12 / PT_W
        x = AX + (len(lines) - 1) * lh / 2
        yc = YP(max((y0 + y1) / 2, 250.0))
        for ln in lines:
            ax.text(x, yc, ln, rotation=90, ha="center", va="center",
                    fontsize=15.0, fontweight="bold", color="#20303C", zorder=4)
            x -= lh

    # ---------------- Layer 3 plane (F1.0 region D, F1.4) -----------------
    ax.add_patch(FancyBboxPatch(
        (DX0, YP(PAN_0)), DX1 - DX0, YH(PAN_1 - PAN_0),
        boxstyle="round,pad=0,rounding_size=0.005", linewidth=1.6,
        edgecolor="#8A7B5F", facecolor=BAND_FILL["L3"],
        mutation_aspect=PT_W / PT_H, zorder=2))
    y = YP(PAN_1 - 4.0)
    for ln in _wrap("Layer 3 - Intelligence and Analytics (cross-cutting plane)",
                    DX1 - DX0, 15.0, 12, True):
        ax.text((DX0 + DX1) / 2, y, ln, ha="center", va="top", fontsize=15.0,
                fontweight="bold", color="#5E5133", zorder=4)
        y -= 15.0 * 1.14 / PT_H
    for i, ln in enumerate(_wrap(
            "Layer 3 terminates no NFV-MANO reference point. It consumes and "
            "produces management and control services on either side of the "
            "MANO path.", DX1 - DX0 - 0.012, FS_TAG, 8)):
        ax.text((DX0 + DX1) / 2, YP(PAN_ANN) - i * FS_TAG * 1.14 / PT_H, ln,
                ha="center", va="top", fontsize=FS_TAG, color="#5E5133",
                style="italic", zorder=4)

    # ---------------- Layer 4 compartments and SMO container (F1.3) -------
    compartment(ax, L4LX, YP(CMP_0), L4LX + L4LW, YP(CMP_1),
                "3GPP Management System - Service-Based Management "
                "Architecture (3GPP TS 28.533)", "#3C5A78")
    compartment(ax, L4RX, YP(CMP_0), L4RX + L4RW, YP(CMP_1),
                "ETSI NFV MANO (ETSI GS NFV-MAN 001)", "#3F7A60")

    smo = [(0.136, YP(SMO_BOT)), (0.136, YP(SMO_TOP)), (DX1 + 0.004, YP(SMO_TOP)),
           (DX1 + 0.004, YP(SMO_RIGHT_BOT)), (DX0 - 0.001, YP(SMO_RIGHT_BOT)),
           (DX0 - 0.001, YP(SMO_BRIDGE)), (0.486, YP(SMO_BRIDGE)),
           (0.486, YP(SMO_BOT))]
    ax.add_patch(Polygon(smo, closed=True, facecolor="none", edgecolor="#6A4E7C",
                         linewidth=2.0, linestyle=(0, (6, 3)), zorder=5))
    ax.text(0.600, YP(SMO_TOP), "O-RAN SMO Framework "
            "(O-RAN.WG1)",
            ha="center", va="center", fontsize=FS_TAG + 1, fontweight="bold",
            color="#6A4E7C", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#6A4E7C", linewidth=1.2))

    # F1.3 equivalence badge: dotted brace over CSMF + NSMF + NSSMF
    ax.plot([BRACE_L] * 2, [YP(R3_Y), YP(R1_Y + R1_H)], color="#3C5A78",
            linestyle=(0, (1.4, 2.2)), linewidth=1.8, zorder=6)
    for yy in (YP(R3_Y), YP(R1_Y + R1_H)):
        ax.plot([BRACE_L, BRACE_L + 0.0045], [yy, yy], color="#3C5A78",
                linewidth=1.8, zorder=6)
    brace = '= ETSI NFV "OSS/BSS" block (ETSI GS NFV-MAN 001)'
    for i, ln in enumerate(_wrap(brace, (R1_Y + R1_H - R3_Y) / PT_W, FS_TAG, 6)):
        ax.text(BRACE_T0 + i * BRACE_TD, YP((R3_Y + R1_Y + R1_H) / 2), ln,
                rotation=90, ha="center", va="center", fontsize=FS_TAG,
                color="#3C5A78", zorder=6)

    # ---------------- Layer 2 slice overlay (F1.2) ------------------------
    ax.add_patch(FancyBboxPatch(
        (0.146, YP(OVL_0)), 0.558, YH(OVL_1 - OVL_0),
        boxstyle="round,pad=0,rounding_size=0.005", linewidth=1.6,
        edgecolor="#8C6E00", facecolor="#FFF6DD", alpha=0.55,
        linestyle=(0, (7, 3)), mutation_aspect=PT_W / PT_H, zorder=1.5))
    ax.text(0.152, YP(SLICE_TAB), "Network Slice Instances / Subnet Instances "
            "(NSI, NSSI)", ha="left", va="center", fontsize=FS_TAG,
            fontweight="bold", color="#7A5C00", zorder=6)
    ax.text(0.152, YP(SLICE_TAB2), "3GPP TS 28.530, TS 28.541  -  per-slice "
            "allocation z_s^(r) (Equations 25-26)", ha="left", va="center",
            fontsize=FS_TAG, color="#7A5C00", style="italic", zorder=6)
    for i, (nm, col) in enumerate((("eMBB", "#C9A227"), ("URLLC", "#B06A00"),
                                   ("mMTC", "#8C6E00"))):
        rx = RIB_X + i * 0.0155
        ax.add_patch(Rectangle((rx, YP(RIB_0)), 0.0140, YH(RIB_1 - RIB_0),
                               facecolor=col, alpha=0.20, edgecolor=col,
                               linewidth=0.9, zorder=1.6))
        ax.text(rx + 0.007, YP((RIB_0 + RIB_1) / 2), nm, rotation=90,
                ha="center", va="center", fontsize=FS_TAG, color="#5E4700",
                fontweight="bold", zorder=6)

    # service-based bus (F1.2 sub-row 2c)
    ax.plot([IX0, IX1_L2], [YP(BUS_Y)] * 2, color="#3C5A78", linewidth=3.0,
            zorder=4)
    for x in C2X:
        ax.plot([x + C2W / 2] * 2, [YP(R2C_Y + R2C_H), YP(BUS_Y)],
                color="#3C5A78", linewidth=1.4, zorder=4)
    ax.text(0.152, YP(BUS_LBL), "Service-Based Interfaces: Namf, Nsmf, Npcf, "
            "Nnef, Nnwdaf (3GPP TS 23.501 clause 7)", ha="left", va="center",
            fontsize=FS_TAG, color="#3C5A78", zorder=6)

    # user-plane SFC bracket (F1.2 sub-row 2b)
    ax.plot([B1X, B1X, B3X + B3W, B3X + B3W],
            [YP(R2B_Y - 6), YP(BRK_LINE), YP(BRK_LINE), YP(R2B_Y - 6)],
            color=GREY, linewidth=1.4, zorder=4)
    ax.text(0.370, YP(BRK_LBL), "user-plane SFC  -  SFC routing "
            "variables phi (Equation 19)", ha="center", va="center",
            fontsize=FS_TAG, color=GREY, zorder=8,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="none", alpha=0.94))

    # ---------------- F1.5  I24 corridor (region B) -----------------------
    # Five parallel lanes, listed left to right in the horizontal skeleton:
    # price-bus arrow, its annotation, two columns of the reference-point list,
    # and the "does not traverse Layer 3" note.  The lane pitch is 0.016
    # (17.3 pt), so no two rotated strings can touch.
    ax.add_patch(FancyBboxPatch(
        (BX0, YP(186.0)), BX1 - BX0, YH(940.0),
        boxstyle="round,pad=0,rounding_size=0.004", linewidth=1.6,
        edgecolor="#B06A00", facecolor="#FDEBD0", alpha=0.75,
        mutation_aspect=PT_W / PT_H, zorder=2))
    ax.text((BX0 + BX1) / 2, YP(1196.0), "I24", ha="center", va="center",
            fontsize=FS_CAP, fontweight="bold", color="#7A3E00", zorder=11,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#FDEBD0",
                      edgecolor="#B06A00", linewidth=1.5))
    for i, ln in enumerate(_wrap(CAPSULES["I24"]["refs"], 820.0 / PT_W,
                                 FS_TAG, 0)):
        ax.text(B_REF0 + i * B_REF_D, YP(656.0), ln, rotation=90, ha="center",
                va="center", fontsize=FS_TAG, color="#7A3E00", zorder=6)
    ax.text(B_ANN, YP(340.0), "NFV-MANO resource path - does not traverse Layer 3",
            rotation=90, ha="center", va="center", fontsize=FS_TAG,
            fontweight="bold", color="#7A3E00", zorder=6)
    # cross-layer price bus (F1.5)
    ax.add_patch(FancyArrowPatch(
        (B_PRICE, YP(1040.0)), (B_PRICE, YP(200.0)), arrowstyle="-|>",
        mutation_scale=15, linewidth=2.0, color=LOOPS["slow"][0], zorder=6,
        shrinkA=0, shrinkB=0))
    ax.text(B_PRICE_L, YP(620.0), "lambda^(r) - resource prices (Equation 34)",
            rotation=90, ha="center", va="center", fontsize=FS_TAG,
            color=LOOPS["slow"][0], zorder=6)
    # I24 branch arrows into Layer 1 and Layer 2, each with its reference point
    for ybr, lab in ((232.0, "Nf-Vi"), (376.0, "Nf-Vi / O2"),
                     (462.0, "Ve-Vnfm-vnf"), (590.0, "O1 MnS (TS 28.532)")):
        ax.add_patch(FancyArrowPatch(
            (BX1, YP(ybr)), (CX0 + 0.002, YP(ybr)), arrowstyle="-|>",
            mutation_scale=13, linewidth=1.6, color="#7A3E00", zorder=7,
            shrinkA=0, shrinkB=0))
        ax.text(BRANCH, YP(ybr + 8.0), lab, rotation=90, ha="left", va="center",
                fontsize=FS_TAG, color="#7A3E00", zorder=7)

    # ---------------- F1.5  I12 capsule -----------------------------------
    ax.text(0.240, YP(315.0), "I12", ha="center", va="center", fontsize=FS_CAP,
            fontweight="bold", color="#7A3E00", zorder=11,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#FDEBD0",
                      edgecolor="#B06A00", linewidth=1.5))
    for i, ln in enumerate(_wrap(CAPSULES["I12"]["refs"], 0.420, FS_TAG, 0)):
        ax.text(0.272, YP(323.0 - i * 16.0), ln, ha="left", va="center",
                fontsize=FS_TAG, color="#7A3E00", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.9))

    # ---------------- F1.5  I23 and I34 capsules in the channel -----------
    # Two lanes only, both kept clear of the Layer 3 panel edge at DX0: the
    # third lane of the previous layout ran inside the panel and over the
    # AI/ML Management box.
    for name, ycap, y0, y1 in (("I23", 668.0, 380.0, 640.0),
                               ("I34", 1150.0, 756.0, 1120.0)):
        refs = CAPSULES[name]["refs"]
        ax.text(CH_CAP, YP(ycap), name, ha="center", va="center",
                fontsize=FS_CAP, fontweight="bold", color="#7A3E00", zorder=11,
                bbox=dict(boxstyle="round,pad=0.42", facecolor="#FDEBD0",
                          edgecolor="#B06A00", linewidth=1.5))
        lanes = _wrap(refs, (y1 - y0) / PT_W, FS_TAG, 0)
        assert len(lanes) <= 2, (
            f'{name} reference list needs {len(lanes)} lanes; the channel holds '
            f'two before the third crosses DX0 into the Layer 3 panel. Shorten '
            f'the refs string or widen the y span.')
        for i, ln in enumerate(lanes):
            ax.text(CH_REF0 + i * CH_REF_D, YP((y0 + y1) / 2), ln, rotation=90,
                    ha="center", va="center", fontsize=FS_TAG, color="#7A3E00",
                    zorder=8, bbox=dict(boxstyle="round,pad=0.12",
                                        facecolor="white", edgecolor="none",
                                        alpha=0.9))

    # ---------------- entities and connectors -----------------------------
    for c in CONNS1:
        draw_conn(ax, BOX1, c)
    for b in BOXES1:
        draw_box(ax, b)

    # ---------------- F1.6  loop overlays ---------------------------------
    for key in ("slow", "medium", "fast"):
        rounded_loop(ax, LOOP_PATHS[key], key, LOOP_ARROW_SEG[key])

    # ---------------- F1.8  timescale ruler (region E) --------------------
    ax.plot([EX, EX], [ruler_y(1e-4), ruler_y(3600.0)], color="#333333",
            linewidth=1.6, zorder=6)
    for key, t0, t1 in RULER_BANDS:
        col, ls, _, _, _ = LOOPS[key]
        ax.plot([EX - 0.010] * 2, [ruler_y(t0), ruler_y(t1)], color=col,
                linewidth=4.0, linestyle=ls, solid_capstyle="butt", zorder=6)
    for t, lab, fn in RULER:
        yy = ruler_y(t)
        ax.plot([EX, EX + 0.005], [yy, yy], color="#333333", linewidth=1.4, zorder=6)
        ax.text(EX + 0.008, yy, lab, ha="left", va="center", fontsize=FS_TAG,
                color="#333333", zorder=6)
        if fn:
            ax.text(0.999, yy, fn, ha="right", va="center", fontsize=FS_TAG,
                    fontweight="bold", color="#333333", zorder=6)
    ax.text(EX - 0.004, ruler_y(2.0e-4), "timescale", rotation=90, ha="center",
            va="bottom", fontsize=FS_TAG, color="#333333", zorder=6)

    # ---------------- F1.7 provenance key and F1.6 loop legend ------------
    ax.plot([0.012, 0.992], [YP(DIV_Y)] * 2, color="#B9C0C7", linewidth=1.0,
            zorder=3)
    x = 0.012
    ax.text(x, YP(PROV_Y), "Provenance (F1.7):", ha="left", va="center",
            fontsize=FS_TAG, fontweight="bold", color=INK, zorder=6)
    x += text_pt("Provenance (F1.7):", FS_TAG, True) / PT_W + 0.014
    for key in ("3gpp", "etsinfv", "oran", "umro"):
        fill, edge, badge = PALETTE[key]
        lab = badge if key != "umro" else "UMRO-5G - no standard counterpart"
        ax.add_patch(FancyBboxPatch(
            (x, YP(PROV_Y - 8.0)), 0.028, YH(16.0),
            boxstyle="round,pad=0,rounding_size=0.004", linewidth=1.4,
            edgecolor=edge, facecolor=fill,
            linestyle=(0, (4, 2.4)) if key == "umro" else "-",
            mutation_aspect=PT_W / PT_H, zorder=6))
        ax.text(x + 0.032, YP(PROV_Y), lab, ha="left", va="center",
                fontsize=FS_TAG, color=edge, zorder=6)
        x += 0.032 + text_pt(lab, FS_TAG, False) / PT_W + 0.020
    ax.text(0.992, YP(PROV_Y), "identifier printed on every box",
            ha="right", va="center", fontsize=FS_TAG,
            color=GREY, style="italic", zorder=6)

    # The three loop captions are set as three columns rather than three
    # stacked full-width paragraphs: same type size, 32 pt less height, and the
    # nesting Fast - Medium - Slow reads left to right.
    keys = ("fast", "medium", "slow")
    ind, gap = 0.018, 0.006                      # sample indent, column gutter
    run = [text_pt(LOOP_TEXT[k], FS_TAG) for k in keys]
    free = 0.976 - 3 * ind - 2 * gap             # width left for the captions
    wid = [free * r / sum(run) for r in run]     # columns proportional to length
    x0 = 0.012
    for k, key in enumerate(keys):
        col, ls, mk, nm, band_ = LOOPS[key]
        ax.plot([x0, x0 + 0.013], [YP(LEG_Y)] * 2, color=col, linewidth=2.6,
                linestyle=ls, zorder=6)
        ax.plot([x0 + 0.0065], [YP(LEG_Y)], marker=mk, markersize=6.0, color=col,
                markeredgecolor="white", markeredgewidth=0.8, zorder=7)
        y = LEG_Y
        for j, ln in enumerate(_wrap_lead_bold(LOOP_TEXT[key], wid[k], FS_TAG)):
            ax.text(x0 + ind, YP(y), ln, ha="left", va="center",
                    fontsize=FS_TAG, color=col if j == 0 else INK,
                    fontweight="bold" if j == 0 else "normal", zorder=6)
            y -= FS_TAG * 1.15
        x0 += ind + wid[k] + gap

    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", path)




# ==========================================================================
# FIGURE 2 -- 3GPP and ETSI NFV entity and reference-point mapping.
# Specification blocks F2.0 to F2.4.  Landscape 3:2.  The provenance palette
# of Figure 1 is reused unchanged (F2.4): 3GPP blue-slate, ETSI NFV green,
# O-RAN violet; equivalence bands are neutral grey and the single
# non-equivalence band of R5 is drawn in the warning hue.
# ==========================================================================
FIG2_W, FIG2_H = 14.4, 10.4
PT2_W, PT2_H = FIG2_W * 72.0, FIG2_H * 72.0
FS2, FS2_BOX, FS2_HDR = 13.5, 15.0, 16.0

NRT_X0, NRT_X1 = 0.008, 0.100        # Non-RT RIC, just outside the left column
LC0, LC1 = 0.112, 0.352              # left column, 3GPP management framework
GC0, GC1 = 0.362, 0.610              # equivalence and reference-point gutter
RC0, RC1 = 0.622, 0.836              # right column, ETSI NFV framework
RL0, RL1 = 0.856, 0.992              # right rail, UMRO-5G layer / control loop
ORVI_X, NFVI_X, O2_X = 0.612, 0.846, 0.055   # routing lanes
EQ_GREY, WARN = "#8A8A8A", "#B06A00"

# vertical skeleton of Figure 2, in points from the bottom of its canvas
HDR2, HDR2_RULE = 716.0, 686.0
R1_BOXES = (642.0, 610.0, 578.0)     # CSMF, NSMF, NSSMF
R1_BAND, R1_BAND_LBL, R1_CONN_LBL = 638.0, 614.0, 568.0
R2_BOX, R2_BAND, R2_BAND_LBL, R2_CONN_LBL = 508.0, 522.0, 544.0, 494.0
R3_BOXES = (466.0, 422.0, 378.0)     # NFVO, VNFM, VIM
R4_BOX, R4_H2, R4_CAP = 304.0, 54.0, 292.0
R5_BOX, R5_BAND = 222.0, 236.0
R6_BOX, R6_CONN_LBL = 130.0, 186.0
SMO2 = (498.0, 678.0)                # O-RAN SMO container, bottom and top
OCLOUD_Y, NOTE_Y = 72.0, 20.0
BH = 28.0                            # standard row-box height


def Y2(pt: float) -> float:
    return pt / PT2_H


# F2.0 column headings, F2.1 rows top to bottom.  Declarative: the row table
# below is a transcription of the specification table in block F2.1.
HEADINGS2 = [
    (LC0, LC1, "3GPP management framework (3GPP TS 28.533)", "#3C5A78"),
    (GC0, GC1, "equivalence and reference point", "#3A3A3A"),
    (RC0, RC1, "ETSI NFV framework (ETSI GS NFV-MAN 001, ETSI GS NFV 002)",
     "#3F7A60"),
    (RL0, RL1, "UMRO-5G layer / control loop", "#6A4E7C"),
]

ROWS2 = [
    dict(id="R1", left=["CSMF", "NSMF", "NSSMF"], left_brace=True,
         band="=", band_label="same functional scope, different decomposition",
         conn="Os-Ma-nfvo (ETSI GS NFV-IFA 013; binding ETSI GS NFV-SOL 005)",
         right=["OSS/BSS"], rail="Layer 4 / Slow"),
    dict(id="R2", left=["NFMF"], band="=", band_label="identical role",
         conn="Ve-Vnfm-em (ETSI GS NFV-IFA 008)",
         right=["EM"], rail="Layer 4 / Medium"),
    dict(id="R3", left=[], left_note="no 3GPP counterpart; virtualised-resource "
         "orchestration is delegated",
         band="", band_label="",
         conn="Or-Vnfm (IFA 007) / Or-Vi (IFA 005) / Vi-Vnfm (IFA 006)",
         right=["NFVO", "VNFM", "VIM"],
         rail="Layer 4 / Slow (VNFM also Medium for scaling and healing)"),
    dict(id="R4", split=True,
         left=["3GPP NF - application and functional aspects"],
         left_owner="provisioning, performance, fault MnS (3GPP TS 28.532)",
         right=["VNF - virtualised-resource aspects"],
         right_owner="Ve-Vnfm-vnf (ETSI GS NFV-IFA 008)",
         caption="one instance, two management owners",
         rail="Layer 2 / Fast (O-DU) and Medium (control-plane NFs)"),
    dict(id="R5", left=["NSI / NSSI (3GPP TS 28.530, TS 28.541)"],
         band="not equal", band_label="composition, not equivalence - one NSI is "
         "realised by one or more NFV Network Services plus non-virtualised "
         "components (ETSI GR NFV-EVE 012)",
         conn="", right=["NS - Network Service"],
         rail="Layer 2 (instance), Layer 4 (management) / Slow"),
    dict(id="R6", left=["managed hardware, transport nodes"], band="=",
         band_label="", conn="Nf-Vi (ETSI GS NFV-MAN 001)",
         right=["NFVI - hardware resources + virtualisation layer"],
         right_tick="Vn-Nf (ETSI GS NFV 002)",
         rail="Layer 1 (hardware), Layer 2 (virtualisation layer) / Slow"),
]

PARADIGM = ("Interaction paradigms differ: the 3GPP framework is service-based "
            "(management service producers and consumers, 3GPP TS 28.533), the "
            "ETSI NFV framework is reference-point-based (ETSI GS NFV-MAN 001). "
            "The equivalences above are functional, not protocol-level.")



def _box2(ax, x0, x1, y, h, label, fw, fs=None, tag=""):
    fill, edge, _ = PALETTE[fw]
    ax.add_patch(FancyBboxPatch(
        (x0, Y2(y)), x1 - x0, Y2(h),
        boxstyle="round,pad=0,rounding_size=0.006", linewidth=1.6,
        edgecolor=edge, facecolor=fill, mutation_aspect=PT2_W / PT2_H, zorder=6))
    fs = fs or FS2_BOX
    lines = _wrap(label, x1 - x0, fs, 10, True)
    if tag:
        lines = lines + ["\x00" + t for t in _wrap(tag, x1 - x0, FS2, 10)]
    lh = fs * 1.14 / PT2_H
    yy = Y2(y + h / 2) + (len(lines) - 1) * lh / 2
    for ln in lines:
        small = ln.startswith("\x00")
        ax.text((x0 + x1) / 2, yy, ln.lstrip("\x00"), ha="center", va="center",
                fontsize=FS2 if small else fs,
                fontweight="normal" if small else "bold",
                color=edge if small else INK, zorder=8)
        yy -= lh


def _text2(ax, x, y, s, w, fs=FS2, ha="center", color=INK, bold=False,
           italic=False, zorder=8, bbox=None, va="center"):
    lines = _wrap(s, w, fs, 0, bold)
    lh = fs * 1.16 / PT2_H
    yy = Y2(y) + (len(lines) - 1) * lh / 2
    for ln in lines:
        ax.text(x, yy, ln, ha=ha, va=va, fontsize=fs, color=color,
                fontweight="bold" if bold else "normal",
                style="italic" if italic else "normal", zorder=zorder, bbox=bbox)
        yy -= lh
    return len(lines)


def _eq_band(ax, y, kind, label):
    """F2.1 / F2.4: a neutral grey '=' band, and for R5 a dashed band in the
    warning hue carrying its own label, because R5 is the one relationship in
    the figure that is a composition and not an equivalence."""
    col = EQ_GREY if kind == "=" else WARN
    xa, xb = GC0 + 0.006, GC1 - 0.006
    if kind == "=":
        for dy in (2.8, -2.8):
            ax.plot([xa, xb], [Y2(y + dy)] * 2, color=col, linewidth=3.2,
                    zorder=5, solid_capstyle="butt")
        if label:
            _text2(ax, (xa + xb) / 2, y - 20.0, label, xb - xa, FS2, color=col)
    else:
        for dy in (44.0, -44.0):
            ax.plot([xa, xb], [Y2(y + dy)] * 2, color=col, linewidth=2.6,
                    linestyle=(0, (5, 3)), zorder=5)
        ax.text(xa + 0.012, Y2(y + 44.0), "not an equivalence", ha="left",
                va="center", fontsize=FS2, fontweight="bold", color=col,
                zorder=7, bbox=dict(boxstyle="round,pad=0.24",
                                    facecolor="white", edgecolor="none"))
        _text2(ax, (xa + xb) / 2, y - 4.0, label, xb - xa - 0.02, FS2,
               color=col, italic=True)


def draw_figure2(path: str) -> None:
    global PT_W, PT_H
    keep = (PT_W, PT_H)
    PT_W, PT_H = PT2_W, PT2_H
    fig = plt.figure(figsize=(FIG2_W, FIG2_H), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- F2.0 column headings -------------------------------------------
    for x0, x1, text, col in HEADINGS2:
        _text2(ax, (x0 + x1) / 2, HDR2, text, x1 - x0, FS2_HDR, color=col,
               bold=True)
        ax.plot([x0, x1], [Y2(HDR2_RULE)] * 2, color=col, linewidth=1.6,
                zorder=4)

    # ---- F2.2 O-RAN SMO container (drawn first, behind the boxes) --------
    ax.add_patch(FancyBboxPatch(
        (NRT_X0 - 0.006, Y2(SMO2[0])), LC1 + 0.008 - NRT_X0 + 0.006,
        Y2(SMO2[1] - SMO2[0]), boxstyle="round,pad=0,rounding_size=0.006",
        linewidth=2.0, edgecolor="#6A4E7C", facecolor="#F7F2F9",
        linestyle=(0, (6, 3)), mutation_aspect=PT2_W / PT2_H, zorder=1.5))
    ax.text(0.180, Y2(SMO2[0]), "O-RAN SMO Framework (O-RAN.WG1)",
            ha="center", va="center", fontsize=FS2, fontweight="bold",
            color="#6A4E7C", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#6A4E7C", linewidth=1.2))
    _box2(ax, NRT_X0, NRT_X1, R1_BOXES[1] - 8.0, 44.0, "Non-RT RIC",
          "oran", FS2_BOX, "O-RAN.WG2")

    # ---- F2.1 rows -------------------------------------------------------
    for row in ROWS2:
        rid = row["id"]

        if rid == "R1":
            for lbl, yy in zip(row["left"], R1_BOXES):
                _box2(ax, LC0, LC1, yy, BH, lbl, "3gpp")
            ax.plot([LC1 + 0.008] * 2, [Y2(R1_BOXES[2]), Y2(R1_BOXES[0] + BH)],
                    color="#3C5A78", linewidth=2.0, zorder=6)
            for yy in (R1_BOXES[2], R1_BOXES[0] + BH):
                ax.plot([LC1, LC1 + 0.008], [Y2(yy)] * 2, color="#3C5A78",
                        linewidth=2.0, zorder=6)
            _eq_band(ax, R1_BAND, "=", row["band_label"])
            _box2(ax, RC0, RC1, R1_BAND - 20.0, 40.0, "OSS/BSS", "etsinfv")
            _text2(ax, (RL0 + RL1) / 2, R1_BAND, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)
            # Os-Ma-nfvo, from NSSMF down to the NFVO of row R3
            ax.add_patch(FancyArrowPatch(
                (LC1, Y2(R1_BOXES[2] + BH / 2)), (RC0, Y2(R3_BOXES[0] + BH / 2)),
                connectionstyle="angle,angleA=0,angleB=90,rad=6",
                arrowstyle="<|-|>", mutation_scale=13, linewidth=1.7,
                color="#3A3A3A", zorder=6, shrinkA=0, shrinkB=0))
            _text2(ax, (GC0 + GC1) / 2, R1_CONN_LBL, row["conn"],
                   GC1 - GC0, FS2, color="#3A3A3A",
                   bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                             edgecolor="none", alpha=0.94))

        elif rid == "R2":
            _box2(ax, LC0, LC1, R2_BOX, BH, "NFMF", "3gpp")
            _eq_band(ax, R2_BAND, "=", "")
            _text2(ax, (GC0 + GC1) / 2, R2_BAND_LBL, row["band_label"],
                   GC1 - GC0, FS2, color=EQ_GREY)
            _box2(ax, RC0, RC1, R2_BOX, BH, "EM", "etsinfv")
            ax.add_patch(FancyArrowPatch(
                (LC1, Y2(R2_BOX + BH / 2)), (RC0, Y2(R2_BOX + BH / 2)),
                arrowstyle="<|-|>", mutation_scale=13, linewidth=1.7,
                color="#3A3A3A", zorder=6, shrinkA=0, shrinkB=0))
            _text2(ax, (GC0 + GC1) / 2, R2_CONN_LBL, row["conn"], GC1 - GC0,
                   FS2, color="#3A3A3A",
                   bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                             edgecolor="none", alpha=0.94))
            _text2(ax, (RL0 + RL1) / 2, R2_BOX + BH / 2, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)

        elif rid == "R3":
            _text2(ax, (LC0 + LC1) / 2, R3_BOXES[1] + BH / 2, row["left_note"],
                   LC1 - LC0, FS2, color=GREY, italic=True)
            for lbl, yy in zip(row["right"], R3_BOXES):
                _box2(ax, RC0, RC1, yy, BH, lbl, "etsinfv")
            for ya, yb, lab in ((R3_BOXES[0], R3_BOXES[1], "Or-Vnfm (IFA 007)"),
                                (R3_BOXES[1], R3_BOXES[2], "Vi-Vnfm (IFA 006)")):
                ax.add_patch(FancyArrowPatch(
                    ((RC0 + RC1) / 2, Y2(ya)), ((RC0 + RC1) / 2, Y2(yb + BH)),
                    arrowstyle="<|-|>", mutation_scale=12, linewidth=1.6,
                    color="#3F7A60", zorder=7, shrinkA=0, shrinkB=0))
                _text2(ax, (RC0 + RC1) / 2, (ya + yb + BH) / 2, lab, RC1 - RC0,
                       FS2, color="#3F7A60",
                       bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                 edgecolor="none", alpha=0.95), zorder=9)
            xr = ORVI_X
            ax.add_patch(FancyArrowPatch(
                (RC0, Y2(R3_BOXES[0] + BH / 2)), (xr, Y2(R3_BOXES[0] + BH / 2)),
                arrowstyle="-", linewidth=1.6, color="#3F7A60", zorder=6,
                shrinkA=0, shrinkB=0))
            ax.add_patch(FancyArrowPatch(
                (xr, Y2(R3_BOXES[0] + BH / 2)), (xr, Y2(R3_BOXES[2] + BH / 2)),
                arrowstyle="-", linewidth=1.6, color="#3F7A60", zorder=6,
                shrinkA=0, shrinkB=0))
            ax.add_patch(FancyArrowPatch(
                (xr, Y2(R3_BOXES[2] + BH / 2)), (RC0, Y2(R3_BOXES[2] + BH / 2)),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.6,
                color="#3F7A60", zorder=6, shrinkA=0, shrinkB=0))
            ax.text(xr - 0.010, Y2((R3_BOXES[0] + R3_BOXES[2]) / 2 + BH / 2),
                    "Or-Vi (IFA 005)", rotation=90, ha="center", va="center",
                    fontsize=FS2, color="#3F7A60", zorder=7)
            _text2(ax, (RL0 + RL1) / 2, R3_BOXES[1] + BH / 2, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)

        elif rid == "R4":
            x0, x1 = LC0, RC1
            ax.add_patch(FancyBboxPatch(
                (x0, Y2(R4_BOX)), x1 - x0, Y2(R4_H2),
                boxstyle="round,pad=0,rounding_size=0.006", linewidth=1.8,
                edgecolor="#4A4A4A", facecolor="#F4F4F4",
                mutation_aspect=PT2_W / PT2_H, zorder=6))
            xm = (GC0 + GC1) / 2
            ax.plot([xm, xm], [Y2(R4_BOX + 2), Y2(R4_BOX + R4_H2 - 2)],
                    color="#4A4A4A", linewidth=1.8, linestyle=(0, (5, 3)),
                    zorder=8)
            _text2(ax, (x0 + xm) / 2, R4_BOX + R4_H2 - 16.0, row["left"][0],
                   xm - x0 - 0.02, FS2_BOX, color=PALETTE["3gpp"][1], bold=True)
            _text2(ax, (x0 + xm) / 2, R4_BOX + 14.0, row["left_owner"],
                   xm - x0 - 0.02, FS2, color=PALETTE["3gpp"][1])
            _text2(ax, (xm + x1) / 2, R4_BOX + R4_H2 - 16.0, row["right"][0],
                   x1 - xm - 0.02, FS2_BOX, color=PALETTE["etsinfv"][1],
                   bold=True)
            _text2(ax, (xm + x1) / 2, R4_BOX + 14.0, row["right_owner"],
                   x1 - xm - 0.02, FS2, color=PALETTE["etsinfv"][1])
            _text2(ax, (x0 + x1) / 2, R4_BOX - 14.0, row["caption"], 0.5, FS2,
                   color="#4A4A4A", italic=True)
            _text2(ax, (RL0 + RL1) / 2, R4_BOX + R4_H2 / 2, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)

        elif rid == "R5":
            _box2(ax, LC0, LC1, R5_BOX, BH, row["left"][0], "3gpp", FS2)
            _eq_band(ax, R5_BAND, "not equal", row["band_label"])
            _box2(ax, RC0, RC1, R5_BOX, BH, row["right"][0], "etsinfv", FS2)
            _text2(ax, (RL0 + RL1) / 2, R5_BOX + BH / 2, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)

        elif rid == "R6":
            _box2(ax, LC0, LC1, R6_BOX, BH, row["left"][0], "3gpp", FS2)
            _eq_band(ax, R6_BOX + 20.0, "=", "")
            _box2(ax, RC0, RC1, R6_BOX, 40.0, row["right"][0], "etsinfv", FS2,
                  row["right_tick"])
            for a, b in (((RC1, Y2(R3_BOXES[2] + BH / 2)),
                          (NFVI_X, Y2(R3_BOXES[2] + BH / 2))),
                         ((NFVI_X, Y2(R3_BOXES[2] + BH / 2)),
                          (NFVI_X, Y2(R6_BOX + 20.0))),
                         ((NFVI_X, Y2(R6_BOX + 20.0)), (RC1, Y2(R6_BOX + 20.0)))):
                ax.add_patch(FancyArrowPatch(
                    a, b, arrowstyle="-|>" if b[0] == RC1 and a[0] == NFVI_X
                    else "-", mutation_scale=12, linewidth=1.6,
                    color="#3F7A60", zorder=5, shrinkA=0, shrinkB=0))
            _text2(ax, (RC0 + RC1) / 2, R6_CONN_LBL,
                   row["conn"], RC1 - RC0, FS2, color="#3F7A60",
                   bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                             edgecolor="none", alpha=0.95), zorder=9)
            _text2(ax, (RL0 + RL1) / 2, R6_BOX + 20.0, row["rail"],
                   RL1 - RL0, FS2, color="#6A4E7C", bold=True)

    # ---- F2.2 O-Cloud ----------------------------------------------------
    _box2(ax, LC0, LC1, OCLOUD_Y, BH, "O-Cloud", "oran", FS2_BOX)
    for a, b, style in ((((LC0 + LC1) / 2, Y2(SMO2[0])), (O2_X, Y2(SMO2[0])), "-"),
                        ((O2_X, Y2(SMO2[0])), (O2_X, Y2(OCLOUD_Y + BH / 2)), "-"),
                        ((O2_X, Y2(OCLOUD_Y + BH / 2)), (LC0, Y2(OCLOUD_Y + BH / 2)),
                         "-|>")):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=13, linewidth=1.8,
            color="#6A4E7C", zorder=5, shrinkA=0, shrinkB=0))
    ax.text(O2_X - 0.010, Y2(320.0), "O2 (O-RAN.WG6)", rotation=90,
            ha="center", va="center", fontsize=FS2, fontweight="bold",
            color="#6A4E7C", zorder=9)
    for a, b in (((LC1, Y2(OCLOUD_Y + BH / 2)), (0.560, Y2(OCLOUD_Y + BH / 2))),
                 ((0.560, Y2(OCLOUD_Y + BH / 2)), (0.560, Y2(R6_BOX + 8.0))),
                 ((0.560, Y2(R6_BOX + 8.0)), (RC0, Y2(R6_BOX + 8.0)))):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-", linewidth=1.6, linestyle=(0, (1.4, 2.4)),
            color=EQ_GREY, zorder=4, shrinkA=0, shrinkB=0))
    ax.text(0.566, Y2(R6_BOX + 8.0), "=", ha="center", va="center",
            fontsize=FS2_HDR, fontweight="bold", color=EQ_GREY, zorder=9,
            bbox=dict(boxstyle="circle,pad=0.16", facecolor="white",
                      edgecolor=EQ_GREY, linewidth=1.2))
    _text2(ax, (GC0 + GC1) / 2 - 0.02, OCLOUD_Y + BH / 2 + 22.0,
           "an O-Cloud is an NFVI managed by a VIM", GC1 - GC0, FS2,
           color=EQ_GREY, italic=True, zorder=9)

    # ---- F2.3 paradigm note ---------------------------------------------
    n = _text2(ax, 0.5, NOTE_Y + 8.0, PARADIGM, 0.94, FS2, color="#3A3A3A")
    ax.add_patch(FancyBboxPatch(
        (0.012, Y2(NOTE_Y - 10.0)), 0.976, Y2(10.0 + n * FS2 * 1.16 + 6.0),
        boxstyle="round,pad=0,rounding_size=0.005", linewidth=1.4,
        edgecolor="#8A8A8A", facecolor="#FAFAFA",
        mutation_aspect=PT2_W / PT2_H, zorder=2))

    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    PT_W, PT_H = keep
    print("wrote", path)


# ==========================================================================
# Verification.  Reads the specification tables out of work/text_architecture.md
# and asserts that every box identifier and every reference point named there
# is present in the declarative data of this module.  Run as part of __main__.
# ==========================================================================
SPEC_BOX_RE = re.compile(r"^\|\s*(L[0-9]-[a-z][0-9]?|L4-x)\s*\|", re.M)

# Reference points and interfaces that the specification puts on a connector,
# a capsule or a bracket.  Each token is first checked to occur in the
# specification (so that this list cannot drift away from it) and then in the
# declarative data of the figures.
SPEC_REFS = [
    "Open Fronthaul", "eCPRI", "F1-U", "N3", "N6", "N4", "Nnwdaf", "Nnef",
    "Vn-Nf", "A1", "R1 services", "Or-Vnfm", "Or-Vi", "Vi-Vnfm", "Os-Ma-nfvo",
    "NFV-SOL 005", "Ve-Vnfm-em", "Ve-Vnfm-vnf", "Nf-Vi", "O1", "O2", "E2",
    "ProvMnS", "PerfMnS", "FaultMnS", "Itf-N", "MDA services",
    "AI/ML management services", "NETCONF", "gNMI", "TS 28.312",
]
FIG2_ENTITIES = [
    "CSMF", "NSMF", "NSSMF", "NFMF", "OSS/BSS", "EM", "NFVO", "VNFM", "VIM",
    "3GPP NF", "VNF", "NSI / NSSI", "NS - Network Service", "managed hardware",
    "NFVI", "Non-RT RIC", "O-Cloud",
]


def check_against_spec(verbose: bool = True) -> int:
    spec = open(SPEC, encoding="utf-8").read()
    f1 = spec[spec.index("## FIGURE 1 SPEC"):spec.index("## FIGURE 2 SPEC")]
    f2 = spec[spec.index("## FIGURE 2 SPEC"):spec.index("## EDIT AND REWRITE")]

    fig1_data = " || ".join(
        [b.id + " :: " + b.label + " :: " + b.tag + " :: " + b.note + " :: " +
         b.timescale + " :: " + b.badge for b in BOXES1] +
        [c.id + " :: " + c.label for c in CONNS1] +
        [k + " :: " + v["refs"] + " :: " + v["endpoints"]
         for k, v in CAPSULES.items()] +
        [LOOP_TEXT[k] for k in LOOP_TEXT])
    fig2_data = " || ".join(
        [str(r) for r in ROWS2] + [h[2] for h in HEADINGS2] + [PARADIGM,
         "Non-RT RIC (O-RAN.WG2)", "O-Cloud", "O2 (O-RAN.WG6)",
         "O-RAN SMO Framework (O-RAN.WG1)",
         "an O-Cloud is an NFVI managed by a VIM"])

    missing = []
    ids = sorted(set(SPEC_BOX_RE.findall(f1)))
    for bid in ids:
        if bid not in BOX1:
            missing.append("Figure 1 box " + bid)
    for cap in ("I12", "I23", "I34", "I24"):
        if cap not in CAPSULES:
            missing.append("Figure 1 interface capsule " + cap)
    for tok in SPEC_REFS:
        if tok not in f1:
            missing.append("token not found in the specification: " + tok)
        elif tok not in fig1_data:
            missing.append("Figure 1 reference point " + tok)
    for ent in FIG2_ENTITIES:
        if ent not in fig2_data:
            missing.append("Figure 2 entity " + ent)
    for tok in ("Os-Ma-nfvo", "Ve-Vnfm-em", "Or-Vnfm", "Or-Vi", "Vi-Vnfm",
                "Ve-Vnfm-vnf", "Nf-Vi", "Vn-Nf", "O2", "NFV-EVE 012"):
        if tok not in fig2_data:
            missing.append("Figure 2 reference point " + tok)
    # F1.7: every box carries exactly one provenance identifier, and F1.5:
    # every connector carries the reference point it draws.  A box without a tag
    # or a connector without a label is a defect, not a stylistic choice.
    for b in BOXES1:
        if not b.tag.strip():
            missing.append("Figure 1 box without a provenance tag: " + b.id)
    for c in CONNS1:
        if not c.label.strip():
            missing.append("Figure 1 connector without a reference point: " + c.id)
    # forbidden items (F1.3, F1.5): they must appear nowhere in the figures
    for bad in ("J12", "J23", "J34", "Service Orchestrator", "OSS/BSS Orchestrator",
                "Customer Portal", "SLA Enforcement Engine",
                "Cross-Domain Coordinator", "Core UPF", "TAPI"):
        if bad in fig1_data:
            missing.append("forbidden item present in Figure 1: " + bad)

    if verbose:
        print("spec check: %d Figure 1 box ids in the specification (%s)"
              % (len(ids), ", ".join(ids)))
        print("spec check: %d reference-point tokens, %d Figure 2 entities"
              % (len(SPEC_REFS), len(FIG2_ENTITIES)))
        if missing:
            print("spec check FAILED:")
            for m in missing:
                print("   -", m)
        else:
            print("spec check PASSED: every box id, every interface capsule and "
                  "every reference point named in the specification is present "
                  "in the declarative data; no forbidden item is present.")
    return len(missing)


def box_rows(b: Box) -> list:
    """The text rows draw_box() will lay out, in drawing order.  Kept beside
    draw_box so that check_box_fit() measures exactly what is rendered."""
    rows: list = [("lab", ln) for ln in _wrap(b.label, b.w, b.fs, 10, True)]
    rows += [("note", ln) for ln in _wrap(b.note, b.w, FS_TAG, 10)] if b.note else []
    tag_lines = _wrap(b.tag, b.w, FS_TAG, 10) if b.tag else []
    if b.timescale:
        one_line = (len(tag_lines) == 1 and
                    text_pt(tag_lines[0], FS_TAG) + text_pt(b.timescale, FS_TAG)
                    + 16.0 <= b.w * PT_W - 10.0)
        if one_line:
            rows.append(("tagts", (tag_lines[0], b.timescale)))
        else:
            rows += [("tag", ln) for ln in tag_lines]
            rows.append(("ts", b.timescale))
    else:
        rows += [("tag", ln) for ln in tag_lines]
    if b.badge:
        rows.append(("badge", b.badge))
    return rows


def check_box_fit(min_pad: float = 4.0, verbose: bool = True) -> int:
    """Report every Figure 1 box whose wrapped text stack comes within
    *min_pad* points of the top and bottom borders of the box.  This is the
    numeric form of the "text overflows its rectangle" defect: draw_box centres
    the stack, so the padding is (box height - text height) / 2."""
    bad = []
    for b in BOXES1:
        rows = box_rows(b)
        lh = {"lab": b.fs * 1.13, "note": FS_TAG * 1.13, "tag": FS_TAG * 1.13,
              "ts": FS_TAG * 1.13, "tagts": FS_TAG * 1.13, "badge": FS_TAG * 1.22}
        total = sum(lh[k] for k, _ in rows)
        pad = (b.h * PT_H - total) / 2.0
        if pad < min_pad:
            bad.append((b.id, b.h * PT_H, total, pad, len(rows)))
    if verbose:
        if bad:
            print("box fit FAILED (text within %.1f pt of the border):" % min_pad)
            for bid, hh, tt, pad, n in bad:
                print("   - %-6s box %.1f pt, %d text rows = %.1f pt, padding "
                      "%.1f pt" % (bid, hh, n, tt, pad))
        else:
            print("box fit PASSED: every one of the %d Figure 1 boxes clears its "
                  "border by at least %.1f pt on each side." % (len(BOXES1), min_pad))
    return len(bad)


if __name__ == "__main__":
    check_against_spec()
    check_box_fit()
    draw_figure1(os.path.join(HERE, "figure1_architecture.png"))
    draw_figure2(os.path.join(HERE, "figure2_entity_mapping.png"))
