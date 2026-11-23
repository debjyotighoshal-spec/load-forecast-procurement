"""Build the A1 portrait poster (poster/Poster_Load_Forecast_Procurement.pptx).

Numbers are read from results/tables via report/results_loader.py, the same
source the report uses. Run poster_figures.py first.
"""
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
from results_loader import load, REPO_URL  # noqa: E402

R = load()
A, C, K, Q, RB, B = R["acc_test"], R["cost"], R["risk"], R["q"], R["rob"], R["boot"]
ENS, REC, SX = R["ENS"], R["rec"], R["SX"]
cost_rec = C.loc[REC, "annual_cost_1pct_kEUR"]
cost_tso = C.loc["TSO public forecast", "annual_cost_1pct_kEUR"]
cost_nai = C.loc["Seasonal naive (last week)", "annual_cost_1pct_kEUR"]
sav_tso = cost_tso - cost_rec
real_sav = RB["TSO+buffer_test_cost_1pct_kEUR"] - RB["Blend_TSO_SARIMAX2_test_cost_1pct_kEUR"]
buf = K["buffer_ensemble_GW"]

NAVY = RGBColor(0x1F, 0x2A, 0x36)
RED = RGBColor(0x9E, 0x2A, 0x2B)
AMBER = RGBColor(0xD9, 0x8A, 0x1E)
BLUE = RGBColor(0x1F, 0x6F, 0xB0)
INK = RGBColor(0x1F, 0x2A, 0x36)
MUTED = RGBColor(0x5B, 0x65, 0x73)
TINT = RGBColor(0xF2, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"

W, H = 23.39, 33.11          # A1 portrait, inches
M = 0.9                       # outer margin
GAP = 0.6
COLW = (W - 2 * M - GAP) / 2

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6])


def box(x, y, w, h, fill=None, shape=MSO_SHAPE.RECTANGLE, name=None, radius=None):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.line.fill.background()
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    sh.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if name:
        sh.name = name
    return sh


def text(x, y, w, h, paras, name=None, anchor=MSO_ANCHOR.TOP, margin=0):
    """paras: list of dicts {t | runs, size, color, bold, font, align, bullet,
    space_after}. runs: list of (text, {bold, color, italic})."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        para.space_after = Pt(p.get("space_after", 6))
        para.line_spacing = p.get("line_spacing", 1.05)
        runs = p.get("runs", [(p.get("t", ""), {})])
        for rt, ro in runs:
            r = para.add_run()
            r.text = rt
            f = r.font
            f.size = Pt(ro.get("size", p.get("size", 22)))
            f.name = ro.get("font", p.get("font", BODY))
            f.bold = ro.get("bold", p.get("bold", False))
            f.italic = ro.get("italic", False)
            f.color.rgb = ro.get("color", p.get("color", INK))
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.32))))
            pPr.set("indent", str(-Emu(Inches(0.32))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                for el in pPr.findall(qn(tag)):
                    pPr.remove(el)
            bc = pPr.makeelement(qn("a:buClr"), {})
            clr = bc.makeelement(qn("a:srgbClr"), {"val": "9E2A2B"})
            bc.append(clr)
            pPr.append(bc)
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
    return tb


def picture(path, x, y, w, name):
    im = Image.open(path)
    h = w * im.height / im.width
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = name
    return h


def section(num, title, x, y, w):
    c = box(x, y, 0.78, 0.78, RED, MSO_SHAPE.OVAL, name=f"Section {num} badge")
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(num)
    r.font.size, r.font.bold, r.font.name = Pt(30), True, HEAD
    r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    text(x + 1.0, y - 0.02, w - 1.0, 0.85,
         [{"t": title, "size": 34, "bold": True, "font": HEAD, "color": NAVY}],
         name=f"Section {num} title", anchor=MSO_ANCHOR.MIDDLE)
    return y + 1.05


def kfmt(v):
    return f"EUR {v:,.0f}k"


# --------------------------------------------------------------- header ----
box(0, 0, W, 3.95, NAVY, name="Header background")
text(M, 0.4, W - 2 * M, 1.4,
     [{"t": "From Forecast to Purchase", "size": 84, "bold": True,
       "font": HEAD, "color": WHITE}], name="Title")
text(M, 1.75, W - 2 * M, 1.2,
     [{"t": "Forecasting German electricity demand one day ahead, and using "
            "the forecast to decide how much energy a supplier should buy "
            "for tomorrow", "size": 30, "color": RGBColor(0xDD, 0xE3, 0xEA)}],
     name="Subtitle")
text(M, 3.0, W - 2 * M, 0.7,
     [{"runs": [("Debjyoti Ghoshal  ·  Ajit Kumar", {"bold": True, "color": WHITE}),
                ("      Time-Series Forecasting, course project  ·  November 2026",
                 {"color": RGBColor(0xB8, 0xC2, 0xCE)})], "size": 26}],
     name="Authors")

# ---------------------------------------------------------- central idea ----
y = 4.35
text(M, y, W - 2 * M, 1.3,
     [{"runs": [("A forecast is worth what it saves. ", {"bold": True, "color": RED}),
                ("We predict tomorrow's demand, buy slightly more than the "
                 "prediction because running short costs twice as much as "
                 "running long, and measure the result in euros.", {})],
       "size": 32, "font": HEAD, "line_spacing": 1.0}], name="Central idea")
y = 5.95
steps =[("DATA", "2,100 days of German load and day-ahead prices, 2015-2020"),
         ("FORECAST", f"SARIMAX + LSTM ensemble: {A.loc[ENS, 'MAPE']:.2f} % error on a typical day"),
         ("DECIDE", f"Buy forecast + {buf:.2f} GW, the 67th percentile of demand"),
         ("SAVE", f"{kfmt(sav_tso)} a year vs. the public forecast (1 % supplier)")]
bw, aw = 4.6, 0.62
gapf = (W - 2 * M - 4 * bw - 3 * aw) / 6
x = M
for i, (h_, b_) in enumerate(steps):
    fill = RED if i == 3 else TINT
    bx = box(x, y, bw, 2.0, fill, MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Flow {h_}",
             radius=0.12)
    col = WHITE if i == 3 else INK
    text(x + 0.3, y + 0.18, bw - 0.6, 1.7,
         [{"t": f"{i + 1}  {h_}", "size": 24, "bold": True,
           "color": WHITE if i == 3 else RED, "space_after": 4},
          {"t": b_, "size": 22, "color": col, "line_spacing": 1.0}],
         name=f"Flow {h_} text")
    x += bw
    if i < 3:
        ar = box(x + gapf, y + 0.7, aw, 0.6, MUTED, MSO_SHAPE.RIGHT_ARROW,
                 name=f"Arrow {i + 1}")
        x += aw + 2 * gapf

# ------------------------------------------------------------ stat tiles ----
y = 8.35
tiles =[(f"{A.loc[ENS, 'RMSE']:.2f} GW", "test RMSE of our ensemble",
          f"TSO public forecast: {A.loc['TSO day-ahead forecast', 'RMSE']:.2f} GW"),
         (f"-{C.loc[REC, 'saving_vs_TSO_pct']:.0f} %", "imbalance cost vs. public forecast",
          f"{kfmt(cost_rec)} instead of {kfmt(cost_tso)} a year"),
         (f"EUR {real_sav:,.0f}k", "saving a year under real bidding timing",
          "the figure we would put in a budget"),
         (f"{Q['test_raw_PICP']:.0f} %", "of test days inside the 95 % interval",
          f"{Q['val_raw_PICP']:.0f} % in the calm validation year")]
tw = (W - 2 * M - 3 * 0.4) / 4
for i, (big, lab, sub) in enumerate(tiles):
    tx = M + i * (tw + 0.4)
    box(tx, y, tw, 2.4, TINT, MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Stat tile {i + 1}",
        radius=0.08)
    text(tx + 0.3, y + 0.12, tw - 0.6, 2.2,
         [{"t": big, "size": 60, "bold": True, "font": HEAD,
           "color": RED if i < 3 else BLUE, "space_after": 0, "line_spacing": 0.95},
          {"t": lab, "size": 21, "bold": True, "color": INK, "space_after": 2},
          {"t": sub, "size": 18, "color": MUTED}], name=f"Stat tile {i + 1} text")

# -------------------------------------------------------------- columns ----
FIG = HERE / "fig"
LX, RX = M, M + COLW + GAP
top = 11.2

# left column
y = section(1, "What the data say", LX, top, COLW)
y += picture(FIG / "p1_series.png", LX, y, COLW, "Figure: load series") + 0.15
y += picture(FIG / "p2_week_holiday.png", LX, y, COLW, "Figure: weekday profile") + 0.15
text(LX, y, COLW, 2.1,
     [{"t": "The week explains 69 % of the variance, the year 28 %. Sunday "
            "demand is 24 % below Wednesday.", "bullet": True},
      {"t": "A public holiday on a weekday looks like a Sunday (44.7 GW). All "
            "12 of the largest unexplained days were holidays. Adding a "
            "holiday flag was the biggest single improvement we made.",
       "bullet": True}], name="Data findings")
y += 2.35
y = section(2, "Which model forecasts best?", LX, y, COLW)
y += picture(FIG / "p3_rmse.png", LX, y, COLW, "Figure: RMSE ranking") + 0.15
text(LX, y, COLW, 2.4,
     [{"t": "MA(q) fits worst: the series remembers the same weekday for "
            "months, and an MA model only remembers q days of shocks.",
       "bullet": True},
      {"t": f"SARIMAX with holidays ({A.loc[SX, 'RMSE']:.2f} GW) and an LSTM "
            f"that sees the calendar ({A.loc['LSTM + calendar', 'RMSE']:.2f} GW) "
            f"tie. They make different errors, so averaging them wins "
            f"({A.loc[ENS, 'RMSE']:.2f} GW).", "bullet": True}],
     name="Model findings")
left_end = y + 2.4

# right column
y = section(3, "From forecast to purchase", RX, top, COLW)
text(RX, y, COLW, 1.2,
     [{"runs": [("Short costs the price + 50 %; long sells back at - 25 %. ",
                 {"bold": True, "color": NAVY}),
                ("So buy the ", {}),
                ("τ* = 0.5 / (0.5 + 0.25) = 0.67", {"bold": True, "color": RED}),
                (" quantile of tomorrow's demand, not the forecast.", {})],
       "size": 22}], name="Newsvendor rule")
y += 1.2
y += picture(FIG / "p4_tau.png", RX, y, COLW, "Figure: cost vs tau") + 0.3
y = section(4, "What it is worth", RX, y, COLW)
y += picture(FIG / "p5_cost.png", RX, y, COLW, "Figure: cost by strategy") + 0.15
text(RX, y, COLW, 1.5,
     [{"t": f"Supplier with 1 % of German demand, test year Oct 2019 - Sep 2020. "
            f"95 % bootstrap range of the saving vs. the public forecast: "
            f"{kfmt(B['test_saving_vs_TSO public forecast_1pct_kEUR_CI95'][0])} to "
            f"{kfmt(B['test_saving_vs_TSO public forecast_1pct_kEUR_CI95'][2])}. "
            f"Bidding by noon the day before cuts it to about {kfmt(real_sav)}.",
       "bullet": True}], name="Value findings")
y += 1.6
y = section(5, "How wrong can it be?", RX, y, COLW)
y += picture(FIG / "p6_interval.png", RX, y, COLW, "Figure: interval through COVID") + 0.15
text(RX, y, COLW, 3.0,
     [{"t": "Two quantile LSTMs, trained with the pinball loss, give a 95 % interval: "
            f"PICP {Q['val_raw_PICP']:.1f} % in validation, {Q['test_raw_PICP']:.1f} % "
            f"in the COVID test year (MPIW {Q['test_raw_MPIW']:.1f} GW).",
       "bullet": True},
      {"t": f"Worst day: EUR {max(K['worst_days_1pct_EUR'].values()):,}. One day "
            f"in 20 costs more than EUR {K['daily_cost_p95_1pct_EUR']:,.0f}. Three "
            f"of the 8 worst days were regional holidays, which are the next thing "
            f"to add to the model.", "bullet": True}], name="Risk findings")
right_end = y + 2.3

# -------------------------------------------------------------- footer -----
fy = H - 0.4 - 0.55 - 0.15 - 1.85      # anchored to the bottom edge
if max(left_end, right_end) > fy - 0.15:
    print(f"WARNING: columns end at {max(left_end, right_end):.2f} in, "
          f"footer starts at {fy:.2f} in")
box(M, fy, W - 2 * M, 1.85, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE, name="Recommendation box",
    radius=0.1)
text(M + 0.45, fy + 0.12, W - 2 * M - 0.9, 1.6,
     [{"runs": [("Recommendation  ", {"bold": True, "color": AMBER, "font": HEAD}),
                (f"Forecast with the SARIMAX + LSTM ensemble, buy {buf:.2f} GW "
                 f"above the forecast, and recalibrate that buffer every month. "
                 f"Feed in the public TSO forecast and regional holidays next, and "
                 f"keep a trader in the loop for shocks like a lockdown.",
                 {"color": WHITE})], "size": 25, "line_spacing": 1.05}],
     name="Recommendation text", anchor=MSO_ANCHOR.MIDDLE)
text(M, fy + 2.0, W - 2 * M, 0.55,
     [{"runs": [("Code and data: ", {"bold": True}), (REPO_URL + "     ", {}),
                ("Data: ", {"bold": True}),
                ("Open Power System Data / ENTSO-E (CC-BY 4.0).     ", {}),
                ("AI use: ", {"bold": True}),
                ("an AI assistant (Claude) helped with code and drafting; see the report.", {})],
       "size": 17, "color": MUTED}], name="Footer")
print(f"left column ends {left_end:.2f} in, right {right_end:.2f} in, "
      f"footer at {fy:.2f} in")

out = HERE / "Poster_Load_Forecast_Procurement.pptx"
prs.save(out)
print("saved", out)
