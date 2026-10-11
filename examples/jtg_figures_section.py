"""Section drawings with the results of the checks laid over them (illustrative values), one HTML page.

    uv run python examples/jtg_figures_section.py [page.html]

Seven sections, each drawn to scale with its bars, ducts and tendons (femotion.codes.jtg.figures.section_figure):
    reinforced-concrete T girder in sagging (compression zone, neutral axis, h0, ξb h0) → the same girder over a support
    in hogging (the compression zone at the bottom) → post-tensioned I girder (ducts and strands; flexure and the
    stresses in service as a diagram beside the section) → box girder with internal and external tendons (crack
    resistance of a segmental member) → precast girder with a cast-in-place topping (jtg.composite) → circular pier in
    eccentric compression (the 5.3.8 segment, the load at e) → a T girder whose detailing misses rules (rows ringed and
    numbered with their clauses) → the first figure again in English.
In Jupyter a section shows itself (RCSection._repr_svg_); section_figure(sec, check=…) returns a femotion.viz Figure
with .svg(), .save("x.svg" / "x.html") and _repr_svg_.
"""
import sys

from femotion import viz
from femotion.codes import jtg
from femotion.codes.jtg.figures import section_figure

strand = jtg.tendon("strand", 1860)
A1 = 139.0  # one φs15.2 strand (mm²)
side = [(2 * 113.1, z, 12) for z in (420, 680, 940)]  # web side bars 2Φ12 (area, depth, diameter)
figs = []

# 1 reinforced-concrete T girder at midspan, sagging
tee = jtg.tee(300, 1300, 1800, 150, concrete="C40", steel="HRB400", bottom=[(4, 28, 55), (4, 28, 115)],
              top=[(8, 16, 45)], bars=side)
figs.append(section_figure(tee, check=jtg.flexure(tee, 1700.0, gamma0=1.0), stirrup_d=10,
                           title="钢筋混凝土 T 梁 · 跨中正弯矩"))

# 2 the girder over an interior support, hogging: the top bars in tension, the web's bottom in compression
pier = jtg.tee(300, 1300, 1800, 150, concrete="C40", steel="HRB400", bottom=[(4, 28, 55)],
               top=[(12, 25, 55)], bars=side)
figs.append(section_figure(pier, check=jtg.flexure(pier, -1500.0, gamma0=1.0), stirrup_d=10,
                           title="钢筋混凝土 T 梁 · 支点负弯矩"))

# 3 post-tensioned I girder: two ducts of 12 strands in the bottom bulb, one of 9 above them
pc = jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", steel="HRB400",
             bottom=[(10, 16, 50)], top=[(16, 12, 45)],
             tendons=[(12 * A1, 1650, 1200, 90), (12 * A1, 1650, 1200, 90), (9 * A1, 1520, 1200, 80)],
             tendon=strand, sigma_pe=1080, sigma_l6=80)
figs.append(section_figure(pc, check=[jtg.flexure(pc, 9000.0, gamma0=1.0, pc_class="A"),
                                      jtg.service_stresses(pc, 6000.0, pc_class="A")],
                           title="后张法预应力 I 梁 · 承载力与使用阶段应力"))

# 4 box girder, segmental, fully prestressed: four ducts in the bottom slab, external tendons in the cell
box = jtg.box(3200, 2000, 250, 220, 900, b_top=6000, concrete="C50", steel="HRB400", bottom=[(30, 16, 60)],
              top=[(50, 16, 50)], tendons=[(12 * A1, 1890, 1200, 90)] * 4, tendon=strand, sigma_pe=1100,
              external=[(2 * 19 * A1, 1700, 1000)])
figs.append(section_figure(box, check=[jtg.flexure(box, 22000.0, gamma0=1.1, pc_class="full"),
                                       jtg.crack_resistance(box, 12500.0, 9000.0, pc_class="full", segmental=True)],
                           title="体内 + 体外预应力箱梁"))

# 5 precast I girder with a cast-in-place deck (8.1): the topping in its own colour
pre = jtg.tee(200, 1600, 1600, 160, bf_bottom=700, hf_bottom=300, concrete="C50", bottom=[(8, 16, 50)],
              top=[(10, 12, 40)], tendons=[(18 * A1, 1450, 1150, 100)], tendon=strand, sigma_pe=1050)
comp = jtg.composite(pre, b=2200, t=180, concrete="C40", top=[(14, 12, 50)])
figs.append(section_figure(comp, check=jtg.flexure(comp.ultimate, 5200.0, gamma0=1.0, pc_class="A"),
                           title="预制梁 + 现浇桥面板（组合式受弯构件）"))

# 6 circular pier, eccentric compression by 5.3.8
col = jtg.circle(1500, 28, 28, 75, concrete="C35")
figs.append(section_figure(col, check=jtg.eccentric(col, 9800.0, 3100.0, l0=18000.0, gamma0=1.0), stirrup_d=10,
                           title="圆形墩柱 · 偏心受压"))

# 7 detailing not met: two layers of five 28 mm bars in a 200 mm web, environment II
bad = jtg.tee(200, 1100, 1600, 160, concrete="C30", steel="HRB400", bottom=[(5, 28, 45), (5, 28, 100)],
              top=[(8, 20, 50)])
figs.append(section_figure(bad, check=jtg.section_detailing(bad, env="II", stirrup_d=10), stirrup_d=10,
                           title="构造检查：不满足的项目"))

# 8 the first one in English
figs.append(section_figure(tee, check=jtg.flexure(tee, 1700.0), stirrup_d=10, lang="en",
                           title="RC T girder, sagging"))

page = viz.html(figs, title="femotion · JTG section drawings")
out = sys.argv[1] if len(sys.argv) > 1 else "jtg_figures_section.html"
with open(out, "w", encoding="utf-8") as fh:
    fh.write(page)
print(f"{len(figs)} figures → {out}")
