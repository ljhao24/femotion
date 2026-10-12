"""Hinges, bearings and expansion joints of JTG 3362-2018 8.6–8.8 with their detailing (9.7, 9.8): concrete hinges
(8.6.1 compression after Hertz, 8.6.2 transverse tension after Mörsch), laminated elastomeric bearings (8.7.3 area,
rubber thickness, compression, steel plates; 8.7.4 sliding; 8.7.5 PTFE sliding bearings), the movements of the beam at
a bearing or a joint (8.7.1 item 5, 8.8.2: temperature, shrinkage and creep with appendix C, prestress, braking,
rotation), joint selection and installation width (8.8.2, 8.8.3), lifting loops (9.8.2), culverts (9.8.1).

Units: mm, mm², MPa; forces kN; stiffness kN/mm; temperatures °C; angles rad. Clauses, formulas and the program's
conventions: docs/codes/JTG_3362-2018.md (chapter on 8.6–8.8). The braking force itself is JTG D60-2015 4.3.5
(jtg.braking_force).

Parameters of JT/T 4-2019 (laminated bearings) that the program takes as defaults, from its text: σc = 10 MPa, 8 MPa
when S < 7 (5.4.1); Eb = 2000 MPa (5.4.3; also the commentary to 8.7.3 of JTG 3362-2018); μ = 0.3 on concrete, 0.2 on
steel (5.4.4); tan α ≤ 0.5 without and ≤ 0.7 with braking (5.4.5); Ge = 1.0 MPa ("一般取值 1.0") and Ee = 5.4 Ge S² with
the shape factor S (within 5 … 12) of 5.4.7; covers 2.5 mm (top, bottom) and 5 mm (sides), rubber layers between plates
≥ 5 mm (5.3.1.2); plates Q235C or better (5.3.2.1), fy = 235 MPa. The friction coefficient μf of PTFE on stainless
steel has no design value in JT/T 4-2019 (table 6 has only the test limit 0.03) and must be given.

    from femotion.codes import jtg
    from femotion.codes.jtg import devices as dv
    b = dv.Laminated(300, 400, t1=8, n=4, ts=3)                                # 300 × 400, te = 37 mm
    mv = dv.movements(15_000, T_set=(10, 20), zone="cold", concrete="C50", RH=75, h=250, t0=60, sigma_pc=8.0)
    for chk in dv.laminated_bearing(b, 850, dl=mv.bearing_shear(), Fbk=20, theta=0.003):  # σc by JT/T 4-2019 5.4.1
        print(chk)
    dv.laminated_sliding(b, RGk=600, RQk=250, dl=mv.bearing_shear(), Fbk=20)              # μ = 0.3 on concrete
    jm = dv.movements(30_000, T_set=(10, 20), zone="cold", concrete="C50", RH=75, h=250, t0=90, sigma_pc=8.0,
                      brake=dv.braking_movement(20, b))
    dv.expansion_joint(jm, C=80, beta=1.3, Bmin=20)                              # C ≥ C⁺ + C⁻, installation width
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bearing import bearing_base_area, eta_s
from .materials import Concrete, Rebar, concrete as _concrete, rebar as _rebar
from .prestress import creep_coefficient, shrinkage_strain
from .rc import Check
from .temperature import effective_temperature

# JT/T 4-2019 (公路桥梁板式橡胶支座)
SIGMA_C = (10.0, 8.0)  # MPa, average compressive stress in service; the second when S < 7 (5.4.1)
SIGMA_C_S = 7.0  # shape factor below which σc is 8 MPa (5.4.1)
GE_DEFAULT = 1.0  # MPa, shear modulus Ge, "generally 1.0" (5.4.7)
EB_DEFAULT = 2000.0  # MPa, bulk modulus of the rubber (5.4.3; JTG 3362-2018 commentary to 8.7.3)
MU = {"concrete": 0.3, "steel": 0.2}  # friction of a laminated bearing on its contact face (5.4.4)
FY_PLATE = 235.0  # MPa, plates of Q235C or better (5.3.2.1)
T1_MIN = 5.0  # mm, rubber between two plates (5.3.1.2)
TAN_ALPHA = (0.5, 0.7)  # limits of the shear angle tan α without / with braking (5.4.5)
SHAPE_FACTOR_RANGE = (5.0, 12.0)  # S for E = 5.4 G S² (5.4.7)
COVER_MIN, EDGE_MIN, PLATE_MIN = 2.5, 5.0, 2.0  # mm, 8.7.3 item 4 (and JT/T 4-2019 figures 1, 2)
K_P = 1.3  # stress correction factor of the plates (8.7.3-10)
JOINT_BETA = (1.2, 1.4)  # amplification β of the joint movements (8.8.2)
ALPHA_C = 1e-5  # 1/°C, concrete of the beam (8.8.2-1, -2; D60 table 4.3.12-1)
ALPHA_S = 12e-6  # 1/°C, steel (D60 table 4.3.12-1): steel bridges ("steel deck", "concrete deck" of table 4.3.12-2)
RUBBERS = {"CR": (-25.0, 60.0), "EPDM": (-40.0, 60.0), "NR": (-40.0, 60.0)}  # site air temperatures (9.7.2; JT/T 4-2019
#                                                                                4.1.2 has CR and NR only)
BEARING_KINDS = ("laminated", "ptfe", "pot", "spherical")  # 9.7.1
JOINT_KINDS = ("modular", "finger", "seamless")  # 9.7.7 (模数式、梳齿板式、无缝式)


def _c(x) -> Concrete:
    return x if isinstance(x, Concrete) else _concrete(x)


def _s(x) -> Rebar:
    return x if isinstance(x, Rebar) else _rebar(x)


# ── 8.6 concrete hinges ─────────────────────────────────────────────────────
def _curvature(r1: float, r2: float) -> float:
    """1/r1 − 1/r2 of a convex cylinder r1 on a plane (r2 = inf) or in a concave seat of radius r2 > r1"""
    if not r1 > 0:
        raise ValueError("r1 > 0")
    k = 1.0 / r1 - (0.0 if math.isinf(r2) else 1.0 / r2)
    if k <= 0:
        raise ValueError("8.6.1 covers a convex cylinder r1 on a plane (r2 = inf) or in a concave seat r2 > r1")
    return k


def hinge_contact_width(Fhd: float, *, Ec: float, r1: float, r2: float = math.inf, l: float,
                        gamma0: float = 1.0) -> float:
    """width b of the pressure transfer area of a cylindrical hinge in line contact (8.6.1-3, mm):
    b = 2.74 √(γ0 Fhd / (Ec (1/r1 − 1/r2) l)). Fhd kN; Ec MPa; r1, r2 the radii of the upper (convex) and lower
    cylinders, r2 = inf on a plane (1/r2 = 0); l the hinge length (mm). The Hertz contact width at P = γ0 Fhd / 1.3
    with ν = 0.2 (commentary 8-13)."""
    return 2.74 * math.sqrt(gamma0 * Fhd * 1e3 / (Ec * _curvature(r1, r2) * l))


def hinge_compression(concrete, Fhd: float, *, l: float, r1: float, r2: float = math.inf, a: float | None = None,
                      Ab: float | None = None, gamma0: float = 1.0) -> Check:
    """compression of a cylindrical concrete hinge in line contact (8.6.1): γ0 Fhd ≤ 7.14 (ηs β fcd)² l / (Ec (1/r1 −
    1/r2)) (8.6.1-1), β = √(Ab / (b l)) (8.6.1-2), b by 8.6.1-3 (hinge_contact_width). Fhd: design force on the contact
    surface (kN); l, r1, r2 (mm) as hinge_contact_width; ηs, fcd, Ec of the hinge's concrete (5.7.1, tables 3.1.4,
    3.1.5). Ab (figure 5.7.1): given, or from the hinge width a (mm): the contact strip b × l runs the full length l and
    sits in the middle of a, so Ab = (b + 2 min(b, (a − b)/2)) l (bearing_base_area). Detailing (9.8.3, warnings): the
    convex radius r1 preferably 1.5 … 3.0 m; concrete at least C30."""
    c = _c(concrete)
    k = _curvature(r1, r2)
    b = hinge_contact_width(Fhd, Ec=c.Ec, r1=r1, r2=r2, l=l, gamma0=gamma0)
    failed = []
    if Ab is None:
        if a is None:
            raise ValueError("pass the hinge width a (Ab by figure 5.7.1) or Ab")
        if b >= a:
            failed.append(f"contact width b = {b:.1f} mm ≥ hinge width a = {a:g} mm")
        Ab = bearing_base_area(b, l, cx=max((a - b) / 2, 0.0), cy=0.0)
    beta = math.sqrt(Ab / (b * l))
    es = eta_s(c)
    R = 7.14 * (es * beta * c.fcd) ** 2 * l / (c.Ec * k) / 1e3
    d = {"b": b, "Ab": Ab, "beta": beta, "eta_s": es, "fcd": c.fcd, "Ec": c.Ec, "curvature": k,
         "p_mean": gamma0 * Fhd * 1e3 / 1.3 / (b * l)}
    chk = Check("concrete hinge, compression (8.6.1)", gamma0 * Fhd, R, "kN", "JTG 3362-2018 8.6.1", d, failed=failed)
    if not 1500 <= r1 <= 3000:
        chk.warnings.append(f"convex radius r1 = {r1 / 1e3:g} m outside 1.5 … 3.0 m (9.8.3)")
    if c.grade < 30:
        chk.warnings.append(f"hinge concrete C{c.grade} below C30 (9.8.3)")
    return chk


def hinge_transverse(Fhd: float, *, a: float, h: float, b: float, As: float, steel="HRB400", gamma0: float = 1.0,
                     As_side: float | None = None) -> Check:
    """transverse tension of a concrete hinge (8.6.2): γ0 Fhd ≤ h / (0.425 (a − b)) fsd As. a: hinge width, h: its
    height, 0.80 … 1.25 a (the formula's condition: failed outside), b: contact width (8.6.1-3), As: transverse bars
    (mm²) with fsd of `steel` (the 330 MPa cap of table 3.2.3-1 note 1 is for shear, torsion and punching bars and is
    not applied). Along the hinge (its side) bars of 0.4 As may be placed (details["As_side"]; a warning when the
    given As_side is less)."""
    st = _s(steel)
    if not b < a:
        raise ValueError("contact width b must be less than the hinge width a")
    R = h / (0.425 * (a - b)) * st.fsd * As / 1e3
    d = {"a": a, "h": h, "b": b, "As": As, "fsd": st.fsd, "h_over_a": h / a, "As_side": 0.4 * As,
         "z": 1.5 * (a - b) / (4 * h) * gamma0 * Fhd / 1.3}
    chk = Check("concrete hinge, transverse tension (8.6.2)", gamma0 * Fhd, R, "kN", "JTG 3362-2018 8.6.2", d)
    if not 0.80 * a - 1e-9 <= h <= 1.25 * a + 1e-9:
        chk.failed.append(f"h = {h:g} mm = {h / a:.2f} a outside 0.80 … 1.25 a (8.6.2)")
    if As_side is not None and As_side < 0.4 * As - 1e-9:
        chk.warnings.append(f"side bars {As_side:.0f} mm² < 0.4 As = {0.4 * As:.0f} mm² (8.6.2)")
    return chk


# ── 8.7 laminated elastomeric bearings ──────────────────────────────────────
@dataclass(frozen=True)
class Laminated:
    """a laminated elastomeric bearing (板式橡胶支座, JT/T 4-2019): rectangular la × lb (la the short side, mm) or
    circular (circle=True, diameter la); n inner rubber layers of t1 between n + 1 steel plates of ts; top and bottom
    cover `cover`, side cover of the plates `edge` (JT/T 4-2019 figures 1, 2: 2.5 and 5 mm). G: Ge (MPa); E: Ee (MPa),
    default 5.4 G S² (JT/T 4-2019 5.4.7, property Ee); Eb: bulk modulus (MPa). rubber: total rubber thickness te if it
    differs from n t1 + 2 cover (program convention: the covers shear with the inner layers)."""
    la: float
    lb: float | None = None
    t1: float = 0.0
    n: int = 0
    ts: float = 0.0
    cover: float = COVER_MIN
    edge: float = EDGE_MIN
    G: float = GE_DEFAULT
    E: float | None = None
    Eb: float = EB_DEFAULT
    rubber: float | None = None
    circle: bool = False

    def __post_init__(self):
        if self.circle:
            object.__setattr__(self, "lb", self.la)
        elif self.lb is None:
            raise ValueError("Laminated: pass lb (rectangle) or circle=True")
        elif self.la > self.lb:
            raise ValueError("Laminated: la is the short side (la ≤ lb)")
        if not (self.t1 > 0 and self.n >= 1):
            raise ValueError("Laminated: t1 > 0 and n ≥ 1 inner rubber layers")

    @property
    def te(self) -> float:
        """total rubber thickness (mm)"""
        return self.n * self.t1 + 2 * self.cover if self.rubber is None else self.rubber

    @property
    def Ee(self) -> float:
        """compressive modulus (MPa): E, or 5.4 G S² (JT/T 4-2019 5.4.7)"""
        return 5.4 * self.G * self.S ** 2 if self.E is None else self.E

    @property
    def l0a(self) -> float:
        """plate size along la (or plate diameter d0)"""
        return self.la - 2 * self.edge

    @property
    def l0b(self) -> float:
        return self.lb - 2 * self.edge

    @property
    def Ag(self) -> float:
        """gross plan area of the bearing (mm²)"""
        return math.pi * self.la ** 2 / 4 if self.circle else self.la * self.lb

    @property
    def Ae(self) -> float:
        """effective bearing area: the area of the plates (8.7.3-1)"""
        return math.pi * self.l0a ** 2 / 4 if self.circle else self.l0a * self.l0b

    @property
    def S(self) -> float:
        """shape factor (JT/T 4-2019 5.4.7): l0a l0b / (2 t1 (l0a + l0b)), circular d0 / (4 t1)"""
        if self.circle:
            return self.l0a / (4 * self.t1)
        return self.l0a * self.l0b / (2 * self.t1 * (self.l0a + self.l0b))

    @property
    def k(self) -> float:
        """horizontal (shear) stiffness Ge Ag / te (kN/mm)"""
        return self.G * self.Ag / self.te / 1e3

    @property
    def height(self) -> float:
        return self.te + (self.n + 1) * self.ts


def sigma_c_limit(b: Laminated) -> float:
    """average compressive stress limit σc of JT/T 4-2019 5.4.1: 10 MPa, 8 MPa when the shape factor S < 7"""
    return SIGMA_C[1] if b.S < SIGMA_C_S else SIGMA_C[0]


def slope_shear(b: Laminated, R: float, slope: float) -> float:
    """shear deformation (mm) from the component of the reaction R (kN) along a grade `slope` (tan) of the soffit or the
    cap under the bearing (8.7.3: Δl for soffits up to 1 %, Δt for cross slopes up to 2 %): R sin(atan slope) / k"""
    return R * math.sin(math.atan(abs(slope))) / b.k


def braking_movement(Fk: float, b: Laminated) -> float:
    """shear deformation of a laminated bearing under the braking force Fk (kN) given to it (8.8.2 item 4):
    Fk te / (Ge Ag), mm"""
    return Fk / b.k


def laminated_bearing(b: Laminated, Rck: float, *, dl: float, sigma_c: float | None = None, fy_plate: float = FY_PLATE,
                      Fbk: float | None = None, dl_brake: float | None = None, slope: float = 0.0, slope_t: float = 0.0,
                      theta: float = 0.0) -> list[Check]:
    """selection of a laminated elastomeric bearing (8.7.3), one bearing, serviceability (commentary). Rck: design
    reaction (kN), the characteristic combination of the vertical loads with impact (8.7.1 item 3). sigma_c: σc (MPa),
    default JT/T 4-2019 5.4.1 (10 MPa, 8 MPa when S < 7, sigma_c_limit); a larger value given fails (the code takes σc
    from JT/T 4). dl: shear deformation without braking (mm): temperature, shrinkage, creep and other longitudinal
    forces at characteristic values (Movements.bearing_shear). Braking: Fbk (kN) given to this bearing (Fbk te / (Ge
    Ag)) or dl_brake (mm). slope: grade of the soffit the bearing sits under (≤ 1 %): Rck along it adds to Δl and atan
    slope to θ; slope_t: cross slope of the cap the bearing is parallel to (≤ 2 %): Δt. theta: rotation at the top of
    the bearing from the deflection of the superstructure (rad). fy_plate: yield strength of the plates (MPa), default
    235 (Q235C or better, JT/T 4-2019 5.3.2.1), σs = 0.65 fy. Returns the checks:
      area 8.7.3-1 (Ae ≥ Rck/σc); shear 8.7.3-2 / -4 (te ≥ 2 Δl, with Δt: 2 √(Δl² + Δt²)) and, with braking, 8.7.3-3 /
      -5 (1.43); stability 8.7.3-6 / -7 (la/10 ≤ te ≤ la/5, la the short side or the diameter); compression 8.7.3-8,
      -9 (θ la/2 ≤ δc,m ≤ 0.07 te); plates 8.7.3-10 (ts ≥ Kp Rck (tes,u + tes,l) / (Ae σs), Kp = 1.3, the plate
      between two inner layers, or a cover and a layer when n = 1). Detailing (warnings): plates ≥ 2 mm, side cover
      ≥ 5 mm, top and bottom cover ≥ 2.5 mm (8.7.3 item 4), rubber between plates ≥ 5 mm (JT/T 4-2019 5.3.1.2).
    Rck ≥ 5000 kN is outside JT/T 4-2019 (its parameters): a warning."""
    out = []
    # 1 area
    sc = sigma_c_limit(b)
    sigma = sc if sigma_c is None else sigma_c
    sig = Rck * 1e3 / b.Ae
    ar = Check("bearing area (8.7.3-1)", sig, sigma, "MPa", "JTG 3362-2018 8.7.3-1",
               {"Ae": b.Ae, "Rck": Rck, "Ae_required": Rck * 1e3 / sigma, "sigma_c_jtt4": sc, "S": b.S},
               limit_check=True, symbol="Rck/Ae")
    if sigma > sc + 1e-9:
        ar.failed.append(f"σc = {sigma:g} MPa above {sc:g} MPa of JT/T 4-2019 5.4.1 (S = {b.S:.2f})")
    if Rck >= 5000:
        ar.warnings.append("Rck ≥ 5000 kN: outside JT/T 4-2019 (bearings below 5000 kN), whose σc, Ee, Eb, μ are used")
    out.append(ar)
    # 2 rubber thickness for shear
    d_slope = slope_shear(b, Rck, slope)
    d_l = dl + d_slope
    d_t = slope_shear(b, Rck, slope_t)
    geo_l, geo_t = [], []
    if abs(slope) > 0.01 + 1e-12:
        geo_l.append(f"soffit grade {abs(slope):.2%} > 1 %: 8.7.3 counts a bearing directly under a grade up to 1 %; "
                     "level the seat (9.7.4)")
    if abs(slope_t) > 0.02 + 1e-12:
        geo_t.append(f"cross slope {abs(slope_t):.2%} > 2 %: 8.7.3 counts a bearing on a cross slope up to 2 %")
    geo = geo_l + geo_t
    eq = ("8.7.3-4", "8.7.3-5") if slope_t else ("8.7.3-2", "8.7.3-3")
    d = {"dl": d_l, "dl_slope": d_slope, "dt": d_t, "te": b.te, "tan_alpha": math.hypot(d_l, d_t) / b.te}
    out.append(Check(f"rubber thickness, shear without braking ({eq[0]})", 2 * math.hypot(d_l, d_t), b.te, "mm",
                     f"JTG 3362-2018 {eq[0]}", d, failed=list(geo), limit_check=True, symbol="2Δl"))
    if Fbk is not None or dl_brake is not None:
        db = dl_brake if dl_brake is not None else braking_movement(Fbk, b)
        d = {"dl": d_l + db, "dl_brake": db, "dt": d_t, "te": b.te, "tan_alpha": math.hypot(d_l + db, d_t) / b.te}
        out.append(Check(f"rubber thickness, shear with braking ({eq[1]})", 1.43 * math.hypot(d_l + db, d_t), b.te,
                         "mm", f"JTG 3362-2018 {eq[1]}", d, failed=list(geo), limit_check=True, symbol="1.43Δl"))
    # 3 stability
    la = b.la
    eqs = "8.7.3-7" if b.circle else "8.7.3-6"
    st = Check(f"rubber thickness, stability ({eqs})", b.te, la / 5, "mm", f"JTG 3362-2018 {eqs}",
               {"te": b.te, "la": la, "te_min": la / 10, "te_max": la / 5}, limit_check=True, symbol="te")
    if b.te < la / 10 - 1e-9:
        st.failed.append(f"te = {b.te:g} mm < {'d' if b.circle else 'la'}/10 = {la / 10:g} mm")
    out.append(st)
    # 4 compression
    th = abs(theta) + math.atan(abs(slope))
    delta = Rck * 1e3 * b.te / (b.Ae * b.Ee) + Rck * 1e3 * b.te / (b.Ae * b.Eb)
    cd = Check("compression of the bearing (8.7.3-8, -9)", delta, 0.07 * b.te, "mm", "JTG 3362-2018 8.7.3-9",
               {"delta_cm": delta, "theta": th, "uplift_limit": th * la / 2, "S": b.S, "Ee": b.Ee, "Eb": b.Eb,
                "Rck": Rck, "te": b.te, "Ae": b.Ae, "la": la},
               failed=list(geo_l), limit_check=True, symbol="δc,m")
    if delta < th * la / 2 - 1e-12:
        cd.failed.append(f"δc,m = {delta:.3f} mm < θ la/2 = {th * la / 2:.3f} mm: the bearing lifts off at one edge")
    lo, hi = SHAPE_FACTOR_RANGE
    if not lo - 1e-9 <= b.S <= hi + 1e-9:
        msg = f"shape factor S = {b.S:.2f} outside {lo:g} … {hi:g} (JT/T 4-2019 5.4.7)"
        if b.E is None:
            cd.failed.append(msg + ": Ee = 5.4 G S² does not apply, pass E")
        else:
            cd.warnings.append(msg)
    out.append(cd)
    # 5 steel plates
    tes = 2 * b.t1 if b.n >= 2 else b.t1 + b.cover
    sigma_s = 0.65 * fy_plate
    ts_req = K_P * Rck * 1e3 * tes / (b.Ae * sigma_s)
    pl = Check("steel plates (8.7.3-10)", ts_req, b.ts, "mm", "JTG 3362-2018 8.7.3-10",
               {"tes_u_plus_l": tes, "sigma_s": sigma_s, "Kp": K_P, "Rck": Rck, "Ae": b.Ae, "fy": fy_plate},
               limit_check=True, symbol="ts,req")
    if b.ts < PLATE_MIN - 1e-9:
        pl.warnings.append(f"plates {b.ts:g} mm < {PLATE_MIN:g} mm (8.7.3 item 4)")
    if b.edge < EDGE_MIN - 1e-9:
        pl.warnings.append(f"side cover of the plates {b.edge:g} mm < {EDGE_MIN:g} mm (8.7.3 item 4)")
    if b.cover < COVER_MIN - 1e-9:
        pl.warnings.append(f"top / bottom cover {b.cover:g} mm < {COVER_MIN:g} mm (8.7.3 item 4)")
    if b.t1 < T1_MIN - 1e-9:
        pl.warnings.append(f"rubber layers {b.t1:g} mm < {T1_MIN:g} mm between plates (JT/T 4-2019 5.3.1.2)")
    out.append(pl)
    return out


def laminated_sliding(b: Laminated, *, RGk: float, dl: float, RQk: float = 0.0, Fbk: float | None = None,
                      mu: float | None = None, contact: str = "concrete", slope: float = 0.0,
                      R_slope: float | None = None) -> list[Check]:
    """sliding of a laminated bearing (8.7.4), in service. mu: friction of the bearing on its contact face, default by
    `contact` from JT/T 4-2019 5.4.4: 0.3 on concrete, 0.2 on steel (measured values may be given). RGk: reaction from
    self-weight (kN); RQk: reaction from the vehicle load with impact (kN, characteristic); dl: Δl of 8.7.3 without the
    braking part (mm); slope: grade of the soffit (≤ 1 %), adds slope_shear with the reaction of 8.7.3, R_slope
    (default RGk + RQk), as laminated_bearing does. Without braking: μ RGk ≥ 1.4 Ge Ag Δl/te (8.7.4-1); with braking
    Fbk (kN, given to this bearing): μ Rck ≥ 1.4 Ge Ag Δl/te + Fbk (8.7.4-2), Rck = RGk + 0.5 RQk."""
    if mu is None:
        if contact not in MU:
            raise ValueError(f"contact {contact!r}: one of {', '.join(MU)}, or pass mu")
        mu = MU[contact]
    d_slope = slope_shear(b, RGk + RQk if R_slope is None else R_slope, slope)
    dl = dl + d_slope
    H = 1.4 * b.G * b.Ag * dl / b.te / 1e3
    d = {"H_shear": H, "Ag": b.Ag, "te": b.te, "Ge": b.G, "mu": mu, "dl": dl, "dl_slope": d_slope}
    out = [Check("sliding without braking (8.7.4-1)", H, mu * RGk, "kN", "JTG 3362-2018 8.7.4-1", d | {"R": RGk},
                 limit_check=True, symbol="1.4GeAgΔl/te")]
    if Fbk is not None:
        Rck = RGk + 0.5 * RQk
        out.append(Check("sliding with braking (8.7.4-2)", H + Fbk, mu * Rck, "kN", "JTG 3362-2018 8.7.4-2",
                         d | {"R": Rck, "Fbk": Fbk}, limit_check=True, symbol="1.4GeAgΔl/te + Fbk"))
    if abs(slope) > 0.01 + 1e-12:
        for chk in out:
            chk.failed.append(f"soffit grade {abs(slope):.2%} > 1 % (8.7.3; level the seat, 9.7.4)")
    return out


def ptfe_sliding(b: Laminated, *, mu_f: float, RGk: float, RQk: float | None = None,
                 tan_alpha: tuple[float, float] = TAN_ALPHA) -> list[Check]:
    """friction of a PTFE sliding laminated bearing (8.7.5): the friction must not shear the rubber beyond tan α.
    Without braking μf RGk ≤ Ge Ag tan α (8.7.5-1), tan α = 0.5; with braking (when RQk, the vehicle reaction with
    impact, is given) μf (RGk + RQk) ≤ Ge Ag tan α (8.7.5-2), tan α = 0.7 (JT/T 4-2019 5.4.5). mu_f: PTFE on stainless
    steel (JT/T 4)."""
    out = []
    cases = [("without braking (8.7.5-1)", RGk, tan_alpha[0], "8.7.5-1")]
    if RQk is not None:
        cases.append(("with braking (8.7.5-2)", RGk + RQk, tan_alpha[1], "8.7.5-2"))
    for name, R, ta, eq in cases:
        out.append(Check(f"PTFE friction {name}", mu_f * R, b.G * b.Ag * ta / 1e3, "kN", f"JTG 3362-2018 {eq}",
                         {"R": R, "tan_alpha": ta, "shear_angle": mu_f * R * 1e3 / (b.G * b.Ag), "mu_f": mu_f,
                          "Ge": b.G, "Ag": b.Ag}, limit_check=True,
                         symbol="μf R"))
    return out


# ── movements at a bearing or a joint (8.7.1 item 5, 8.8.2) ────────────────
@dataclass
class Movements:
    """movements (mm, all ≥ 0) of the beam end at a bearing or an expansion joint, from l (mm) to the point that does
    not move: t_plus expansion from the temperature rise (8.8.2-1), t_minus contraction from the drop (8.8.2-2),
    shrinkage (8.8.2-3), creep (8.8.2-4), elastic shortening from prestress applied after the beam sits on the bearing
    (8.7.1 item 5-3), brake (8.8.2 item 4, either way), rotation θ y of the beam end at the bearing (8.7.1 item 5-2,
    the bottom moves away from the span, like an expansion)."""
    l: float
    t_plus: float
    t_minus: float
    shrinkage: float = 0.0
    creep: float = 0.0
    elastic: float = 0.0
    brake: float = 0.0
    rotation: float = 0.0
    details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def closing(self, beta: float = 1.3) -> float:
        """C⁺ = β (Δlt⁺ + Δlb⁺) (8.8.2-5)"""
        return beta * (self.t_plus + self.brake)

    def opening(self, beta: float = 1.3) -> float:
        """C⁻ = β (Δlt⁻ + Δls⁻ + Δlc⁻ + Δlb⁻) (8.8.2-6)"""
        return beta * (self.t_minus + self.shrinkage + self.creep + self.brake)

    def bearing_shear(self, braking: bool = False) -> float:
        """Δl of 8.7.3 (program convention): the larger of the expansion side (temperature rise, rotation) and the
        contraction side (temperature drop, shrinkage, creep, elastic shortening), each measured from the bearing's
        installation; braking=True adds `brake` to both (laminated_bearing takes the braking part separately)."""
        x = self.brake if braking else 0.0
        return max(self.t_plus + self.rotation + x, self.t_minus + self.shrinkage + self.creep + self.elastic + x)


def movements(l: float, *, T_set, T_max: float | None = None, T_min: float | None = None, zone: str | None = None,
              structure: str = "concrete", south: bool = False, alpha: float | None = None, concrete=None,
              RH: float | None = None, h: float | None = None, t0: float | None = None, tu: float | None = None,
              ts: float = 3.0, method: str = "table", fly_ash: float | None = None, eps_cs: float | None = None,
              phi: float | None = None, sigma_pc: float | None = None, elastic: bool = False, brake: float = 0.0,
              theta: float = 0.0, y: float = 0.0) -> Movements:
    """movements of the beam at a joint or a bearing (8.8.2 items 1–4; 8.7.1 item 5). l (mm): the beam length that
    moves the joint or bearing, from the point that does not move (stationary_point for continuous decks on flexible
    supports). T_set: (lower, upper) installation temperature (or one value); T_max, T_min: the highest and lowest
    effective temperatures, or from D60 table 4.3.12-2 by zone, structure, south (temperature.effective_temperature).
    Δlt⁺ = α l (Tmax − Tset,l), Δlt⁻ = α l (Tset,u − Tmin); α default 1e-5 for concrete bridges (8.8.2), 1.2e-5 for the
    steel bridges of table 4.3.12-2 (structure "steel deck" or "concrete deck", D60 table 4.3.12-1).
    Concrete bridges (structure="concrete") must state the shrinkage and the precompression: concrete (appendix C) or
    eps_cs (0 for none), and sigma_pc (0 for reinforced concrete); a ValueError otherwise, so that neither drops out of
    C⁻ unnoticed. Steel bridges take them as 0 unless given.
    Shrinkage Δls⁻ = εcs(tu, t0) l and creep Δlc⁻ = σpc / Ec φ(tu, t0) l (Ec of table 3.1.5) from appendix C
    (prestress.shrinkage_strain, creep_coefficient; concrete, RH %, notional size h mm, t0 the age at installation, d),
    or eps_cs, phi given; tu default t0 + 3650 d (program convention: the "final" values of commentary tables C-1, C-2
    run 10 years; for shrinkage table C-1 takes ts + 3650 d, t0 + 3650 is about 1 % more). sigma_pc: precompression
    at the centroid after the losses of that stage (MPa; mid-span and quarter points of simple spans, several sections
    of continuous ones). elastic=True: also σpc / Ec l (prestress applied with
    the beam on the bearing, 8.7.1 item 5-3). brake (mm): braking_movement. theta, y: end rotation (rad) and the depth
    from the centroid to the bearing (mm) → θ y (8.7.1 item 5-2)."""
    lo, hi = (T_set, T_set) if np.ndim(T_set) == 0 else T_set
    if lo > hi:
        raise ValueError("T_set = (lower, upper)")
    notes = []
    if T_max is None or T_min is None:
        if zone is None:
            raise ValueError("pass T_max and T_min, or the climate zone of D60 table 4.3.12-2")
        e_hi, e_lo = effective_temperature(zone, structure, south)
        T_max = e_hi if T_max is None else T_max
        T_min = e_lo if T_min is None else T_min
    if not T_min <= lo <= hi <= T_max:
        raise ValueError(f"installation range {lo:g} … {hi:g} °C outside the effective temperatures {T_min:g} … "
                         f"{T_max:g} °C")
    steel = structure in ("steel deck", "concrete deck")
    if alpha is None:
        alpha = ALPHA_S if steel else ALPHA_C
    if not steel:
        if concrete is None and eps_cs is None:
            raise ValueError("a concrete bridge: pass concrete (appendix C shrinkage) or eps_cs (0 for none), 8.8.2-3")
        if sigma_pc is None:
            raise ValueError("a concrete bridge: pass sigma_pc (0 for reinforced concrete), 8.8.2-4")
    sigma_pc = 0.0 if sigma_pc is None else sigma_pc
    d = {"T_max": T_max, "T_min": T_min, "T_set": (lo, hi), "alpha": alpha}
    t_plus = alpha * l * (T_max - lo)
    t_minus = alpha * l * (hi - T_min)
    c = _c(concrete) if concrete is not None else None
    if (eps_cs is None or (phi is None and sigma_pc)) and c is not None:
        if RH is None or h is None or t0 is None:
            raise ValueError("appendix C needs RH, h and t0 (or pass eps_cs and phi)")
        if tu is None:
            tu = t0 + 3650.0
            notes.append("tu = t0 + 3650 d (the 10-year final values of commentary tables C-1, C-2)")
        d.update(t0=t0, tu=tu, RH=RH, h=h)
    if eps_cs is None:
        if c is not None:
            eps_cs = shrinkage_strain(c, RH, h, tu, t0, ts, method)
        else:
            eps_cs = 0.0
            notes.append("no shrinkage (steel bridge, no concrete given)")
    Ec = c.Ec if c is not None else None
    if sigma_pc:
        if Ec is None:
            raise ValueError("creep and elastic shortening need the concrete (Ec)")
        if phi is None:
            phi = creep_coefficient(c, RH, h, tu, t0, method, fly_ash)
    phi = 0.0 if phi is None else phi
    d.update(eps_cs=eps_cs, phi=phi, sigma_pc=sigma_pc, Ec=Ec)
    creep = sigma_pc / Ec * phi * l if sigma_pc else 0.0
    el = sigma_pc / Ec * l if (sigma_pc and elastic) else 0.0
    return Movements(l, t_plus, t_minus, eps_cs * l, creep, el, brake, abs(theta) * y, d, notes)


def installation_width(C: float, Bmin: float, C_plus: float, C_minus: float) -> tuple[float, float]:
    """installation (or delivery) width of a joint (8.8.3): between Bmin + C⁺ and Bmin + (C − C⁻), C the movement range
    and Bmin the least working width of the chosen joint (mm). The joint then closes to no less than Bmin and opens to
    no more than Bmin + C."""
    a, b = Bmin + C_plus, Bmin + (C - C_minus)
    if a > b + 1e-9:
        raise ValueError(f"C = {C:g} mm < C⁺ + C⁻ = {C_plus + C_minus:.1f} mm: no installation width works")
    return a, b


def expansion_joint(mv: Movements, C: float, *, beta: float = 1.3, Bmin: float | None = None,
                    kind: str | None = None) -> Check:
    """selection of an expansion joint (8.8.2 item 5): C ≥ C⁺ + C⁻ (8.8.2-7), C⁺ = β (Δlt⁺ + Δlb⁺), C⁻ = β (Δlt⁻ +
    Δls⁻ + Δlc⁻ + Δlb⁻), β = 1.2 … 1.4 (below 1.2: failed; above 1.4: a note). C: movement range of the chosen joint
    (mm); Bmin: its least working width → details["installation"] (8.8.3). kind (9.7.7, warning otherwise): "modular",
    "finger", "seamless". Other movements (earthquake, wind, deflection, 8.8.2 note 1) are not included; an installation
    temperature outside the design range needs a new calculation (note 2: movements with T_set the actual one)."""
    Cp, Cm = mv.closing(beta), mv.opening(beta)
    d = {"C_plus": Cp, "C_minus": Cm, "beta": beta, "t_plus": mv.t_plus, "t_minus": mv.t_minus,
         "shrinkage": mv.shrinkage, "creep": mv.creep, "brake": mv.brake}
    chk = Check("expansion joint (8.8.2)", Cp + Cm, C, "mm", "JTG 3362-2018 8.8.2", d, list(mv.notes),
                limit_check=True, symbol="C⁺ + C⁻")
    if beta < JOINT_BETA[0] - 1e-12:
        chk.failed.append(f"β = {beta:g} < 1.2 (8.8.2)")
    elif beta > JOINT_BETA[1] + 1e-12:
        chk.notes.append(f"β = {beta:g} above the 1.2 … 1.4 of 8.8.2")
    if Bmin is not None and Cp + Cm <= C + 1e-9:
        chk.details["installation"] = installation_width(C, Bmin, Cp, Cm)
    if kind is not None and kind not in JOINT_KINDS:
        chk.warnings.append(f"joint type {kind!r}: 9.7.7 names modular, finger and seamless joints; others after "
                            "special study")
    return chk


# ── stiffness tools (8.7.1 item 5, D60 4.3.5 item 3; program tools) ─────────
def series(*k: float) -> float:
    """stiffness of springs in series, e.g. a laminated bearing on a flexible pier (commentary to 8.8.2): 1 / Σ 1/ki"""
    return 1.0 / sum(1.0 / v for v in k)


def distribute(F: float, k) -> np.ndarray:
    """a horizontal force (braking) shared by supports of stiffness ki on a deck rigid along its axis (D60 4.3.5
    item 3): Fi = F ki / Σ k"""
    k = np.asarray(k, float)
    return F * k / k.sum()


def stationary_point(x, k) -> float:
    """the point of a deck that does not move under a uniform strain (temperature, shrinkage, creep), with supports of
    horizontal stiffness ki at xi (bearing and pier in series): Σ ki (xi − x0) = 0, x0 = Σ ki xi / Σ ki"""
    x, k = np.asarray(x, float), np.asarray(k, float)
    return float((k * x).sum() / k.sum())


# ── 9.7, 9.8 detailing ──────────────────────────────────────────────────────
def bearing_detailing(*, kind: str = "laminated", rubber: str | None = None, site_T=None,
                      n_longitudinal: int | None = None, n_transverse: int | None = None) -> Check:
    """detailing of bearings (9.7.1–9.7.3). kind: "laminated", "ptfe", "pot", "spherical" (9.7.1; others after
    special study). rubber of laminated, PTFE and pot bearings: "CR" (chloroprene, site air temperatures −25 … +60 °C),
    "EPDM" or "NR" (−40 … +60 °C) against site_T = (lowest, highest) (9.7.2). n_longitudinal: bearings along the bridge
    at one support point of a beam, not two (9.7.3-1); n_transverse > 2: partial uplift to be considered (9.7.3-2;
    commentary: preferably no more than two). Returns a detailing report (Check.report)."""
    chk = Check.report("bearing detailing (9.7)", "JTG 3362-2018 9.7.1–9.7.3")
    if kind not in BEARING_KINDS:
        chk.warnings.append(f"bearing type {kind!r}: 9.7.1 names laminated, PTFE sliding, pot and spherical bearings; "
                            "others after special study")
    if rubber is not None:
        r = rubber.upper()
        if r not in RUBBERS:
            raise ValueError(f"rubber {rubber!r}: one of {', '.join(RUBBERS)}")
        lo, hi = RUBBERS[r]
        chk.details["rubber_range"] = (lo, hi)
        if site_T is None:
            chk.notes.append("site temperatures not given: 9.7.2 not checked")
        elif site_T[0] < lo or site_T[1] > hi:
            chk.warnings.append(f"{r} rubber serves {lo:g} … +{hi:g} °C, site {site_T[0]:g} … {site_T[1]:g} °C (9.7.2)")
    elif kind != "spherical":
        chk.notes.append("rubber not given: 9.7.2 not checked")
    if n_longitudinal is not None and n_longitudinal > 1:
        chk.warnings.append(f"{n_longitudinal} bearings along the bridge at one support point (9.7.3-1: not two)")
    if n_transverse is not None and n_transverse > 2:
        chk.warnings.append(f"{n_transverse} bearings across: consider partial uplift (9.7.3-2; commentary: preferably "
                            "no more than two)")
    return chk


def lifting_loop(Gk: float, d: float, *, n: int = 4, n_effective: int | None = None, steel: str = "HPB300",
                 cold_worked: bool = False, embed: float | None = None, inner_diameter: float | None = None) -> Check:
    """lifting loops of precast members (9.8.2): each loop counts with two legs; under the characteristic self-weight
    Gk (kN) the stress ≤ 65 MPa (a stress check: `ok`). With four loops only three act; up to three, all; more than
    four: pass n_effective. Detailing (warnings): HPB300, no cold-worked bars; embedment ≥ 35 d; inner diameter ≥ 3 d
    and ≥ 60 mm (the 180° hooks and their welding or tying to the member's bars are not checked). d: bar diameter
    (mm)."""
    if n_effective is None:
        if n > 4:
            raise ValueError("9.8.2 says three of four loops act; for more than four pass n_effective")
        n_effective = 3 if n == 4 else n
    A = 2 * math.pi * d * d / 4
    sigma = Gk * 1e3 / (n_effective * A)
    chk = Check("lifting loops (9.8.2)", sigma, 65.0, "MPa", "JTG 3362-2018 9.8.2",
                {"n": n, "n_effective": n_effective, "A_loop": A}, limit_check=True, symbol="σ")
    if str(steel).upper() != "HPB300":
        chk.warnings.append(f"loops of {steel}: 9.8.2 asks for HPB300")
    if cold_worked:
        chk.warnings.append("cold-worked bars must not be used for lifting loops (9.8.2)")
    if embed is not None and embed < 35 * d - 1e-9:
        chk.warnings.append(f"embedment {embed:g} mm < 35 d = {35 * d:g} mm (9.8.2)")
    if inner_diameter is not None and inner_diameter < max(3 * d, 60.0) - 1e-9:
        chk.warnings.append(f"inner diameter {inner_diameter:g} mm < max(3 d, 60) = {max(3 * d, 60.0):g} mm (9.8.2)")
    return chk


def culvert_detailing(diameter: float, *, layers: int) -> Check:
    """round culverts (9.8.1): two layers of bars from 1 m span up. The cover (9.1.1) and the handling and erection
    checks of precast culvert members are not part of this report."""
    chk = Check.report("culvert detailing (9.8.1)", "JTG 3362-2018 9.8.1", {"diameter": diameter, "layers": layers},
                       ["precast culvert members: check them for handling and erection (9.8.1)"])
    if diameter >= 1000 - 1e-9 and layers < 2:
        chk.warnings.append(f"round culvert of {diameter / 1e3:g} m with {layers} layer of bars: two layers from "
                            "1 m up (9.8.1)")
    return chk
