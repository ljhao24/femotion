"""Durability and detailing of a reinforced-concrete T girder to JTG 3362-2018 4.5 and chapter 9 (illustrative values).

The T girder of examples/jtg_rc_girder.py (web 200 mm, depth 1100 mm, flange 1600 × 160 mm, C40, HRB400, ten 28 mm bars
at the bottom, eight 20 mm at the top), its deck slab and its stirrups:
    environment and concrete class (4.5.3) → cover (table 9.1.1) → bar layout: two rows of five do not fit in the web
    (9.3.3), the same ten bars as two welded frames of five layers (9.3.11) do, with side bars of 0.2 % (9.3.7, the
    upper value for thin webs with welded frames) → the same
    girder at the coast (environment III) → anchorage, laps and cut-off lengths (9.1.4, 9.1.9, 9.3.8) → stirrups
    (9.3.12, 9.1.6) → bars over the end support (9.3.9) → the deck slab (9.3.4 with 9.2.2, 9.2.4).
Detailing never changes a verdict: the reports list the unmet rules as warnings, the strength checks keep their `ok`.

    uv run python examples/jtg_detailing.py
"""
from femotion.codes import jtg
from femotion.codes.jtg import detailing as dt


def show(title, chk):
    print(f"\n{title}\n  {chk}")
    for n in chk.notes:
        print(f"    note: {n}")


env, life = "I", 100  # general environment (carbonation only), 100-year design life (JTG B01-2014)
side = [(2 * 50.27, z, 8) for z in (300, 440, 580, 720, 850)]  # side bars: 2 × 8 mm per level, depth from the top

# the bars as examples/jtg_rc_girder.py lumps them: two rows of five 28 mm bars, centres 54 and 110 mm above the bottom
rows = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
               bottom=[(5, 28, 54), (5, 28, 110)], top=[(8, 20, 50)], bars=side)
# the same ten bars as two welded frames, five layers of two in contact (centroid still 110 mm above the bottom)
frames = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
                 bottom=[(2, 28, 54 + 28 * k) for k in range(5)], top=[(8, 20, 50)], bars=side)

print(f"environment {env} ({jtg.ENVIRONMENTS[env]}), {life} years")
show("durability (4.5.3, 3.1.2)", jtg.durability(frames, env, life=life))
print(f"  lowest class C{jtg.min_concrete_class(env, 'superstructure', life)}, "
      f"cover of the outermost steel ≥ {jtg.cover_min(env, 'superstructure', life):g} mm (table 9.1.1)")

# stirrups of 10 mm around the main bars; on the top face the 12 mm transverse bars of the deck lie outside the
# longitudinal flange bars
kw = dict(env=env, life=life, stirrup_d=10, outer_d=(10, 12))
show("two rows of five in a 200 mm web", jtg.section_detailing(rows, **kw))
show("two welded frames of five layers", jtg.section_detailing(frames, welded_frame=True, main_rows=5, **kw))
show("the same girder at the coast (environment III)",
     jtg.section_detailing(frames, welded_frame=True, main_rows=5, **{**kw, "env": "III"}))
show("welded frame welds (9.3.11)", dt.welded_frame_detailing(5, 28, weld=140, short_weld=70))

# strength is checked as before; the detailing warnings sit beside it
print(f"\nflexure, frames: {jtg.flexure(frames, 2000.0)}")

# anchorage, laps, cut-off (lengths in mm)
d, h0 = 28, 1100 - 110
print("\nanchorage and laps of 28 mm HRB400 in C40")
la = {k: jtg.anchorage_length(d, "HRB400", "C40", **kw)
      for k, kw in (("straight", {}), ("hooked", {"hook": True}), ("in compression", {"compression": True}))}
print("  la " + ", ".join(f"{k} {v:g}" for k, v in la.items()) + " (table 9.1.4)")
for share in (0.25, 0.5, 1.0):
    print(f"  tension lap with {share:.0%} of the bars lapped: {jtg.lap_length(d, 'HRB400', 'C40', share=share):g} "
          "(table 9.1.9, 9.1.9 item 2)")
full, needed = dt.cutoff_lengths(d, h0, "HRB400", "C40")
print(f"  a bar cut off: {full:g} beyond the section that uses it fully, {needed:g} beyond the one that no longer "
      "needs it (9.3.8)")
show("a lap of the bottom bars (9.1.7, 9.1.9)",
     dt.splice_detailing("lap", d, concrete="C40", length=1500, share=0.25, joint_spacing=2000, clear=64))

# stirrups: 2 legs of 10 mm at 200 in the span, closed 135° hooks, laps of the bottom bars at mid-span
show("stirrups in the span", jtg.stirrup_detailing(frames, (2, 10, 200), hook=(50, 50, 135), support_spacing=100,
                                                   lap=("tension", d, 200)))
show("stirrups along the laps at 100 mm", jtg.stirrup_detailing(frames, (2, 10, 200), hook=(50, 50, 135),
                                                               support_spacing=100, lap=("tension", d, 100)))
show("bars over the end support (9.3.9)", dt.support_detailing(10, 4, d=d, extension=300))

# the deck slab between the girders (the flange, 160 mm): 12 mm main bars at 100, 10 mm distribution bars at 200
show("deck slab (9.3.4, 9.2.2, 9.2.4)", dt.flange_slab_detailing(160, main=(12, 100), distribution=(10, 200)))
