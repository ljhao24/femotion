"""The formulas of the calculation reports (femotion.codes.jtg.report), checked from the page.

For every case of tests/report_cases.py (one per branch of every template, the cases of the independent review
included) the check is written into a report and read back:
- every value substituted into a formula (tagged data-sym) is read as printed (rounded); the same symbol printed
  twice must read the same;
- the code's formula, written out here once more, recomputes the result from these values: it must equal the result
  printed and the program's result to the rounding of the printed values;
- every inequality of the page holds for the two numbers it compares (the row before and the row itself, or two
  numbers on one row);
- the template wrote the check (its guards ran), the verdict box states the condition not met.
Independent references (test_hand_calculations): results computed here from the code's formulas with the values of
its tables typed in (not from the program), one or more per kind of template.
Chinese reports: no English word in the names of the checks, the program's messages, the rules of the detailing
reports or the names of the values (a short whitelist of units, grades and symbols)."""
import html
import math
import re

import pytest

import report_cases as rc
from femotion.codes import jtg
from femotion.codes.jtg import _report_text as TX
from femotion.codes.jtg import report as jr
from femotion.report import Report

pi = math.pi


# ── reading a check back ────────────────────────────────────────────────────
def _num(t: str) -> float:
    t = t.replace("−", "-").replace("×10<sup>", "e").replace("</sup>", "")
    return float(re.sub(r"<[^>]+>", "", t))


def _write(chk, lang="zh"):
    rep = Report("t", lang=lang)
    ch = rep.chapter("c")
    jr.write_check(ch, chk, id="k", figure=False)
    h = rep.html()
    i = h.index(f'id="{rep.labels["k"][1]}"')
    j = re.compile(r"<h[1-4] id=").search(h, i + 10)
    seg = h[i:j.start() if j else None]
    vals = {}
    for k, t in re.findall(r'<span class="v" data-sym="([^"]+)">(.*?)</span>', seg):
        vals.setdefault(k, []).append((_num(t), t))
    return seg, vals, rep.store["jtg.checks"][-1]


def _digits_tol(t: str) -> float:
    """half a unit of the last digit printed"""
    t = re.sub(r"<[^>]+>", "", t.replace("×10<sup>", "e").replace("</sup>", ""))
    m = re.match(r"[−-]?(\d+)(?:\.(\d+))?(?:e(-?\d+))?", t)
    nd = len(m.group(2) or "")
    e = int(m.group(3) or 0)
    return 0.5 * 10 ** (e - nd)


def _close(a, b, rel=3e-3, abs_=1e-9):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


CASES = rc.templated()


@pytest.fixture(scope="module", params=CASES, ids=[n for n, _ in CASES])
def written(request):
    name, chk = request.param
    seg, vals, entry = _write(chk)
    return name, chk, seg, vals, entry


def test_template_used(written):
    name, chk, seg, vals, entry = written
    assert entry.guards >= 1, f"{name}: written in the general form"
    assert vals, name


def test_symbols_printed_alike(written):
    """a symbol printed twice reads the same"""
    name, chk, seg, vals, entry = written
    for k, vs in vals.items():
        a = [x for x, _ in vs]
        tol = max(_digits_tol(t) for _, t in vs)
        assert max(a) - min(a) <= 2 * tol + 1e-12, (name, k, vs)


# ── the code's formulas from the page ───────────────────────────────────────
def V(vals, k, default=None):
    if k not in vals:
        if default is None:
            raise KeyError(k)
        return default
    return vals[k][0][0]


def _flexure(c, v):
    out = []
    if "xb" in v:  # 5.2.7
        xb = V(v, "xib") * V(v, "h0")
        bw = V(v, "bf", V(v, "b", 0.0))
        Mu = V(v, "fcd") * bw * xb * (V(v, "h0") - xb / 2) + \
            V(v, "fsdc", 0) * V(v, "Asc", 0) * (V(v, "h0") - V(v, "asc", 0))
        return [("x_b", xb, "xb"), ("Mu,b", Mu / 1e6, "Mu", c.resistance)]
    if "ecu" in v:  # over-reinforced
        ss = V(v, "ecu") * V(v, "Es") * (V(v, "beta") * V(v, "h0") / V(v, "x") - 1)
        bw = V(v, "bf", V(v, "b", 0.0))
        C = V(v, "fcd") * bw * V(v, "x") + V(v, "sc", 0) * V(v, "Asc", 0)
        Mu = V(v, "fcd") * bw * V(v, "x") * (V(v, "h0") - V(v, "x") / 2) + \
            V(v, "sc", 0) * V(v, "Asc", 0) * (V(v, "h0") - V(v, "asc", 0))
        return [("σs", ss, "ss"), ("ΣN = 0", C, None, ss * V(v, "As")), ("Mu", Mu / 1e6, "Mu", c.resistance)]
    T = V(v, "fsd") * V(v, "As") + V(v, "fpd", 0) * V(v, "Ap", 0) + V(v, "sex", 0) * V(v, "Aex", 0)
    if "a" in v:
        num = V(v, "fsd") * V(v, "As") * V(v, "as") + V(v, "fpd", 0) * V(v, "Ap", 0) * V(v, "ap", 0) + \
            V(v, "sex", 0) * V(v, "Aex", 0) * V(v, "aex", 0)
        out.append(("a", num / T, "a"))
        out.append(("h0", V(v, "h") - V(v, "a"), "h0"))
    elif "as" in v and "h" in v and "Mu1" not in v:
        out.append(("h0", V(v, "h") - V(v, "as"), "h0"))
    Cs = V(v, "fsdc", 0) * V(v, "Asc", 0) + (V(v, "fpdc", 0) - V(v, "sp0c", 0)) * V(v, "Apc", 0)
    fcd, h0 = V(v, "fcd"), V(v, "h0")
    if "Mu1" in v:  # 5.2.4 (5.2.6)
        aref, h = V(v, "aref"), V(v, "h")
        M1 = V(v, "fpd", 0) * V(v, "Ap", 0) * (h - V(v, "ap", 0) - aref) + \
            V(v, "fsd") * V(v, "As") * (h - V(v, "as") - aref) + \
            V(v, "sex", 0) * V(v, "Aex", 0) * (h - V(v, "aex", 0) - aref)
        if "apc" in v and "asc" in v:  # A'p in tension
            M1 -= (V(v, "fpdc") - V(v, "sp0c")) * V(v, "Apc") * (V(v, "apc") - V(v, "asc"))
        out.append(("Mu1", M1 / 1e6, "Mu1"))
        Mu = M1
        if "Mu2" in v:
            bw = V(v, "bf", V(v, "b", 0.0))
            x1 = T / (fcd * bw)
            M2 = fcd * bw * x1 * (h0 - x1 / 2)
            out += [("x1", x1, "x1"), ("Mu2", M2 / 1e6, "Mu2")]
            Mu = max(M1, M2)
        out.append(("Mu", Mu / 1e6, "Mu", c.resistance))
        return out
    if "b" in v and "bf" in v and "hf" in v and V(v, "x") > V(v, "hf"):
        b, bf, hf = V(v, "b"), V(v, "bf"), V(v, "hf")
        x = (T - Cs - fcd * (bf - b) * hf) / (fcd * b)
        Mu = fcd * (b * x * (h0 - x / 2) + (bf - b) * hf * (h0 - hf / 2))
    else:
        bw = V(v, "bf", V(v, "b", 0.0))
        x = (T - Cs) / (fcd * bw)
        Mu = fcd * bw * x * (h0 - x / 2)
    Mu += V(v, "fsdc", 0) * V(v, "Asc", 0) * (h0 - V(v, "asc", 0)) + \
        (V(v, "fpdc", 0) - V(v, "sp0c", 0)) * V(v, "Apc", 0) * (h0 - V(v, "apc", 0))
    out += [("x", x, "x"), ("Mu", Mu / 1e6, "Mu", c.resistance)]
    if "Mcr" in v:
        g = 2 * V(v, "S0") / V(v, "W0")
        Mcr = (V(v, "spc") + V(v, "gamma") * V(v, "ftk")) * V(v, "W0") / 1e6
        out += [("γ", g, "gamma"), ("Mcr", Mcr, "Mcr", c.details["Mcr"])]
    return out


def _shear(c, v):
    if c.name.startswith("section limit"):
        return [("5.2.11", 0.51e-3 * math.sqrt(V(v, "fcuk")) * V(v, "b") * V(v, "h0"), "R", c.resistance)]
    if c.name.startswith("shear (slab"):
        return [("Vu", 1.25 * 0.5e-3 * V(v, "a2") * V(v, "ftd") * V(v, "b") * V(v, "h0"), "Vu", c.resistance)]
    out = []
    P = 100 * (V(v, "As") + V(v, "Ap", 0)) / (V(v, "b") * V(v, "h0"))
    out.append(("P", min(P, 2.5), "P"))
    if "Asv" in v:
        out.append(("ρsv", V(v, "Asv") / (V(v, "sv") * V(v, "b")), "rho_sv"))
    K = (2 + 0.6 * V(v, "P")) * math.sqrt(V(v, "fcuk")) * (V(v, "rho_sv") * V(v, "fsv") +
                                                        0.6 * V(v, "rho_pv", 0) * V(v, "fpv", 0))
    Vcs = 0.45e-3 * V(v, "a1") * V(v, "a2") * V(v, "a3") * V(v, "b") * V(v, "h0") * math.sqrt(K)
    Vu = Vcs + V(v, "Vsb", 0) + V(v, "Vpb", 0) + V(v, "Vpbex", 0)
    out += [("K", K, "K"), ("Vcs", Vcs, "Vcs"), ("Vu", Vu, "Vu", c.resistance),
            ("5.2.11", 0.51e-3 * math.sqrt(V(v, "fcuk")) * V(v, "b") * V(v, "h0"), "Vmax", c.details["V_section_max"]),
            ("5.2.12", 0.5e-3 * V(v, "a2") * V(v, "ftd") * V(v, "b") * V(v, "h0"), "Vmin", c.details["V_detailing"])]
    return out


def _axial(c, v):
    return [("Nu", 0.9 * V(v, "phi") * (V(v, "fcd") * V(v, "A") + V(v, "fsdc") * V(v, "As")) / 1e3, "Nu", c.resistance)]


def _eccentric(c, v):
    out = []
    if "alpha" in v:
        a, at = V(v, "alpha"), V(v, "alphat")
        assert at == max(1.25 - 2 * a, 0.0) or _close(at, 1.25 - 2 * a, 2e-3, 2e-4)
        N = a * V(v, "fcd") * V(v, "A") * (1 - math.sin(2 * pi * a) / (2 * pi * a)) + \
            (a - at) * V(v, "fsd") * V(v, "As")
        M = 2 / 3 * V(v, "fcd") * V(v, "A") * V(v, "r") * math.sin(pi * a) ** 3 / pi + \
            V(v, "fsd") * V(v, "As") * V(v, "rs") * (math.sin(pi * a) + math.sin(pi * at)) / pi
        return [("Nud", N / 1e3, "Nu", c.resistance), ("Mud = Nud ηe0", M / 1e6, "Mud", N * V(v, "e") / 1e6)]
    if "es1" in v:  # 5.3.6
        return [("Nu", V(v, "fsd") * V(v, "As") * (V(v, "h0") - V(v, "asc")) / V(v, "es1") / 1e3, "Nu", c.resistance)]
    x = V(v, "x")
    h = V(v, "h", c.section.h)
    xc = min(x, h)
    fcd = V(v, "fcd")
    if "hf" in v and "b" in v and "bf" in v and xc > V(v, "hf"):
        Cc = fcd * (V(v, "b") * xc + (V(v, "bf") - V(v, "b")) * V(v, "hf"))
    else:
        Cc = fcd * V(v, "bf", V(v, "b", 0.0)) * xc
    if "ecu" in v:
        ss = V(v, "ecu") * V(v, "Es") * (V(v, "beta") * V(v, "h0") / x - 1)
        out.append(("σs", ss, "ss"))
    else:
        ss = V(v, "ss")
    N1 = Cc + V(v, "fsdc", 0) * V(v, "Asc", 0) - ss * V(v, "As")
    out.append(("Nu (5.3.4-1)", N1 / 1e3, "Nu1"))
    Nu = N1
    if "Nr" in v:
        if "A" in v:  # T: concrete about A's
            Nr = (fcd * V(v, "A") * (V(v, "yc") - V(v, "asc")) + V(v, "fsdc") * V(v, "As") * V(v, "h0r")) / V(v, "er")
        else:
            Nr = (fcd * V(v, "b") * h * (V(v, "h01") - h / 2) +
                  V(v, "fsdc") * V(v, "As") * (V(v, "h01") - V(v, "as"))) / V(v, "er")
        out.append(("Nr", Nr / 1e3, "Nr"))
        Nu = min(N1, Nr)
    out.append(("Nu", Nu / 1e3, "Nu", c.resistance))
    if "eta" in v and "z1" in v:
        eta = 1 + (V(v, "l0") / V(v, "h")) ** 2 * V(v, "z1") * V(v, "z2") / (1300 * V(v, "e0") / V(v, "h0"))
        out.append(("η", eta, "eta"))
    return out


def _tension(c, v):
    if "e" in v and "ep" in v:  # small eccentricity
        N1 = V(v, "fsd") * V(v, "Asc") * (V(v, "h0") - V(v, "asc")) / V(v, "e")
        N2 = V(v, "fsd") * V(v, "As") * (V(v, "h01") - V(v, "as")) / V(v, "ep")
        return [("Nu1", N1 / 1e3, "N1"), ("Nu2", N2 / 1e3, "N2"), ("Nu", min(N1, N2) / 1e3, "Nu", c.resistance)]
    if "x" in v:  # large eccentricity
        if "esc" in v:
            N1 = V(v, "fsd") * V(v, "As") * (V(v, "h0") - V(v, "asc")) / V(v, "esc")
            N2 = V(v, "fsd") * V(v, "As") - V(v, "fcd") * V(v, "b") * V(v, "x1")
            return [("Nu1", N1 / 1e3, "N1"), ("Nu2", N2 / 1e3, "N2"), ("Nu", max(N1, N2) / 1e3, "Nu", c.resistance)]
        N = V(v, "fsd") * V(v, "As") - V(v, "fsdc", 0) * V(v, "Asc", 0) - V(v, "fcd") * V(v, "b") * V(v, "x")
        return [("Nu", N / 1e3, "Nu", c.resistance)]
    return [("Nu", V(v, "fsd") * V(v, "As") / 1e3, "Nu", c.resistance)]


def _torsion(c, v):
    out = [("ζ", V(v, "fsd") * V(v, "Ast") * V(v, "sv") / (V(v, "fsv") * V(v, "Asv1") * V(v, "Ucor")), "zeta")]
    bt = V(v, "beta_t", 1.0)
    Tu = bt * 0.35 * V(v, "beta_a") * V(v, "ftd") * V(v, "Wt") + \
        1.2 * math.sqrt(V(v, "zeta_u")) * V(v, "fsv") * V(v, "q") * V(v, "Acor")
    out.append(("Tu", Tu / 1e6, "Tu", c.resistance))
    if "Vu" in v:
        b, h0 = V(v, "b"), V(v, "h0")
        Vu = 0.5e-4 * V(v, "a1") * V(v, "a2") * V(v, "a3") * (10 - 2 * bt) * b * h0 * math.sqrt(
            (2 + 0.6 * V(v, "P")) * math.sqrt(V(v, "fcuk")) * V(v, "rho_sv") * V(v, "fsv"))
        bt_raw = 1.5 / (1 + 0.5 * V(v, "Vd") * 1e3 * V(v, "Wtb") / (V(v, "Td") * 1e6 * b * h0))
        out += [("βt", min(max(bt_raw, 0.5), 1.0), "beta_t"), ("Vu (5.5.4-1)", Vu, "Vu", c.details["Vu"]),
                ("Asv1/sv,T", max(V(v, "Asv1") / V(v, "sv") - V(v, "qs"), 0.0), "q")]
    tau = V(v, "gTd") * 1e6 / V(v, "Wtt")
    if "gVd" in v:
        tau += V(v, "gVd") * 1e3 / (V(v, "b") * V(v, "h0"))
    out.append(("5.5.3", tau, "tau", c.details["tau_section"]))
    return out


def _crack_width(c, v):
    d = c.details
    out = []
    if "Np0" in v:  # class B
        s = (V(v, "Ms") * 1e6 - V(v, "Np0") * (V(v, "z") - V(v, "ep"))) / (V(v, "At", 0) or d["At"]) / V(v, "z")
        out.append(("σss", s, "sssl"))
    elif "es" in v:
        out.append(("σss", V(v, "Ns") * 1e3 * (V(v, "es") - V(v, "z")) / (V(v, "As") * V(v, "z")), "sssl"))
    elif "esp" in v:
        out.append(("σss", V(v, "Ns") * 1e3 * V(v, "esp") / (V(v, "As") * (V(v, "h0") - V(v, "asc"))), "sssl"))
    elif "Ms" in v:
        out.append(("σss", V(v, "Ms") * 1e6 / (0.87 * V(v, "As") * V(v, "h0")), "sssl"))
    else:
        out.append(("σss", V(v, "Ns") * 1e3 / V(v, "As"), "sssl"))
    if "Ml" in v:
        out.append(("C2", 1 + 0.5 * V(v, "Ml") / V(v, "Ms"), "C2"))
    elif "Nl" in v:
        out.append(("C2", 1 + 0.5 * V(v, "Nl") / V(v, "Ns"), "C2"))
    rho = V(v, "As") / (V(v, "Ate") if "Ate" in v else V(v, "A"))
    out.append(("ρte", min(max(rho, 0.01), 0.1), "rho_te"))
    W = V(v, "C1") * V(v, "C2") * V(v, "C3") * V(v, "sss") / V(v, "Es") * (V(v, "c") + V(v, "d")) / \
        (0.36 + 1.7 * V(v, "rho_te"))
    out.append(("Wcr", W, "Wcr", c.demand))
    return out


def _crack_resistance(c, v):
    d = c.details
    g = d[d["edge"]]
    s = V(v, "Ms") * 1e6 / V(v, "W0") + V(v, "st_t", 0)
    out = [("|σst|", abs(s - V(v, "st_t", 0)) + V(v, "st_t", 0) * 0, None, abs(g["sigma_st"] - g.get("sigma_t", 0.0)))]
    if "sG" in v:
        out.append(("|σG|", V(v, "MG") * 1e6 / V(v, "W0"), None, abs(g["sigma_G"])))
    return out


def _principal(c, v):
    sx, sy, tau = V(v, "scx"), V(v, "scy"), V(v, "tau")
    out = []
    if "I0" in v:
        out.append(("σcx", V(v, "spc") + V(v, "M") * 1e6 * V(v, "y0") / V(v, "I0"), "scx"))
        if "S0" in v:
            out.append(("τ", V(v, "V") * 1e3 * V(v, "S0") / (V(v, "b") * V(v, "I0")), "tau"))
    r = math.hypot((sx - sy) / 2, tau)
    if "stp" in v:
        out.append(("σtp", r - (sx + sy) / 2, "stp", c.demand))
    else:
        out.append(("σcp", r + (sx + sy) / 2, "scp", c.demand))
    return out


def _service(c, v):
    d = c.details
    if "scc" in v:  # 7.1.4
        out = [("eN", V(v, "Mt") * 1e6 / V(v, "Np0") - V(v, "hps"), "eN"),
               ("σcc", V(v, "Np0") / V(v, "Acr") + V(v, "Np0") * V(v, "e0N") * V(v, "c") / V(v, "Icr"), "scc",
                c.demand)]
        if "sp_" in v:
            sp = V(v, "aEP") * (V(v, "Np0") / V(v, "Acr") -
                                V(v, "Np0") * V(v, "e0N") * (V(v, "hp") - V(v, "c")) / V(v, "Icr"))
            out += [("σp", sp, "sp_"), ("σp0 − σp", V(v, "sp0") - sp, "sp", max(d["tendon_stress"]))]
        return out
    out = [("σkc", V(v, "Mk") * 1e6 * V(v, "y0") / V(v, "I0"), "skc")]
    if "spe" in v:
        out.append(("σpe + αEPσkt", V(v, "spe") + V(v, "aEP") * V(v, "skt"), "sp", max(d["tendon_stress"])))
    return out


def _transient(c, v):
    if "x0" in v:
        out = [("σcc", V(v, "M") * 1e6 * V(v, "x0") / V(v, "Icr"), "scc", c.demand),
               ("0.80f'ck", 0.80 * V(v, "fckt"), None, c.resistance)]
        if "aES" in v:
            out.append(("σs", V(v, "aES") * V(v, "Ms") * 1e6 * (V(v, "h0i") - V(v, "x0s")) / V(v, "Icrs"), "ss",
                        max(c.details["sigma_s"])))
        return out
    return [("0.70f'ck", 0.70 * V(v, "fckt"), None, c.resistance), ("σcc", V(v, "scc"), None, c.demand)]


def _local(c, v):
    d = c.details
    out = [("β", math.sqrt(V(v, "Ab") / V(v, "Al")), "beta")]
    if "F2" not in v:
        out.append(("Fu", 0.9 * V(v, "eta_s") * V(v, "beta") * V(v, "fcd") * V(v, "Aln") / 1e3, "Fu", c.resistance))
        return out
    F1 = 1.3 * V(v, "eta_s") * V(v, "beta") * V(v, "fcd") * V(v, "Aln") / 1e3
    F2 = 0.9 * (V(v, "eta_s") * V(v, "beta") * V(v, "fcd") +
                V(v, "k") * V(v, "rho_v") * V(v, "beta_cor") * V(v, "fsd")) * V(v, "Aln") / 1e3
    out += [("βcor", math.sqrt(V(v, "Acor") / V(v, "Al")), "beta_cor"), ("Fu1", F1, "F1"), ("Fu2", F2, "F2"),
            ("Fu", min(F1, F2), "Fu", c.resistance)]
    if d.get("mesh") is not None:
        n1, As1, l1, n2, As2, l2, sp = d["mesh"]
        out.append(("ρv", (n1 * As1 * l1 + n2 * As2 * l2) / (V(v, "Acor0") * sp), "rho_v"))
    return out


def _punching(c, v):
    d = c.details
    if "bm" in v:
        return [("Fld", V(v, "ps") * V(v, "A") / 1e3, "Fld", d["Fld"]), ("bm", (V(v, "bt") + V(v, "bb")) / 2, "bm"),
                ("Fu", 0.7 * V(v, "beta_h") * V(v, "ftd") * V(v, "bm") * V(v, "h0") / 1e3, "Fu", c.resistance)]
    conc = V(v, "beta_h") * V(v, "ftd")
    sp = 0.15 * V(v, "spc", 0)
    if "Vs" not in v:
        return [("Fu", (0.7 * conc + sp) * V(v, "Um") * V(v, "h0") / 1e3, "Fu", c.resistance)]
    Vc = (0.35 * conc + sp) * V(v, "Um") * V(v, "h0") / 1e3
    Vs = 0.75 * V(v, "fsv") * V(v, "Asvu") / 1e3 if "Asvu" in v else \
        0.75 * V(v, "fsd") * V(v, "Asbu") * math.sin(math.radians(d["theta"])) / 1e3
    out = [("Vc", Vc, "Vc"), ("Vs", Vs, "Vs"), ("Fu", Vc + Vs, "Fu", c.resistance),
           ("5.6.2-1", 1.05 * conc * V(v, "Um") * V(v, "h0") / 1e3, "Vmax", d["V_section_max"])]
    if "Umo" in v:
        out.append(("outer", (0.7 * conc + sp) * V(v, "Umo") * V(v, "h0") / 1e3, "Vout", d["V_outer"]))
    return out


def _devices(c, v):
    nm, d = c.name, c.details
    if nm.startswith("bearing area"):
        return [("Rck/Ae", V(v, "Rck") * 1e3 / V(v, "Ae"), "sig", c.demand)]
    if nm.startswith("rubber thickness, shear"):
        k = 2.0 if "without" in nm else 1.43
        return [("Δ", k * math.hypot(V(v, "dl"), V(v, "dt", 0)), "req", c.demand)]
    if nm.startswith("rubber thickness, stability"):
        return [("te", V(v, "te"), None, d["te"]), ("la/5", V(v, "la") / 5, None, c.resistance)]
    if nm.startswith("compression of the bearing"):
        dl = V(v, "Rck") * 1e3 * V(v, "te") / V(v, "Ae") * (1 / V(v, "Ee") + 1 / V(v, "Eb"))
        return [("δc,m", dl, "dcm", d["delta_cm"])]
    if nm.startswith("steel plates"):
        return [("ts", V(v, "Kp") * V(v, "Rck") * 1e3 * V(v, "tes") / (V(v, "Ae") * V(v, "sigma_s")), "ts_req",
                 c.demand)]
    if nm.startswith("sliding"):
        H = 1.4 * V(v, "Ge") * V(v, "Ag") * V(v, "dl") / V(v, "te") / 1e3 + V(v, "Fbk", 0)
        return [("H", H, "H", c.demand), ("μR", V(v, "mu") * V(v, "R"), None, c.resistance)]
    if nm.startswith("PTFE"):
        return [("μfR", V(v, "muf") * V(v, "R"), "muR", c.demand),
                ("GeAg tanα", V(v, "Ge") * V(v, "Ag") * V(v, "tana") / 1e3, None, c.resistance)]
    Cp = V(v, "beta") * (V(v, "tp") + V(v, "br"))
    Cm = V(v, "beta") * (V(v, "tm") + V(v, "sh") + V(v, "cr") + V(v, "br"))
    return [("C+", Cp, "Cp"), ("C−", Cm, "Cm"), ("C", Cp + Cm, None, c.demand)]


def _recompute(c, v):
    n = c.name
    if n.startswith("flexure"):
        return _flexure(c, v)
    if n.startswith(("shear", "section limit")):
        return _shear(c, v)
    if n == "axial compression":
        return _axial(c, v)
    if n.startswith("eccentric"):
        return _eccentric(c, v)
    if n.startswith("tension"):
        return _tension(c, v)
    if n.startswith("torsion"):
        return _torsion(c, v)
    if n == "crack width":
        return _crack_width(c, v)
    if n.startswith("crack resistance (normal"):
        return _crack_resistance(c, v)
    if n.startswith(("crack resistance (inclined", "principal")):
        return _principal(c, v)
    if n.startswith("concrete compression in service"):
        return _service(c, v)
    if n.startswith("deflection"):
        k = 300 if c.details.get("cantilever") else 600
        return [("L/k", V(v, "L") / k, None, c.resistance)]
    if n.startswith("transient"):
        return _transient(c, v)
    if n.startswith("local compression"):
        return _local(c, v)
    if n.startswith("punching"):
        return _punching(c, v)
    return _devices(c, v)


def test_formulas_recompute_the_results(written):
    """page values → the code's formula → = the printed result = the program's result"""
    name, chk, seg, vals, entry = written
    rows = _recompute(chk, vals)
    assert rows
    for row in rows:
        label, calc, key = row[:3]
        if key is not None:
            shown = V(vals, key)
            tol = max(_digits_tol(t) for _, t in vals[key])
            assert abs(calc - shown) <= max(3e-3 * abs(shown), 2 * tol), (name, label, calc, shown)
        if len(row) > 3:
            assert _close(calc, row[3], 3e-3, 2e-3), (name, label, calc, row[3])


# ── inequalities ────────────────────────────────────────────────────────────
_NUM = re.compile(r"[-−]?\d+(?:\.\d+)?(?:×10\^-?\d+)?%?")


def _text(span: str) -> str:
    s = span.replace("<sup>", "^").replace("</sup>", "")
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return s.replace(" ", " ")


def _val(tok: str) -> tuple[float, float]:
    """(value, half a unit of the last digit)"""
    t = tok.replace("−", "-").rstrip("%")
    e = 0
    if "×10^" in t:
        t, e = t.split("×10^")
        e = int(e)
    nd = len(t.split(".")[1]) if "." in t else 0
    return float(t) * 10 ** e, 0.5 * 10 ** (e - nd)


def _last(s: str):
    s = re.split(r"取|, (?=[A-Za-zα-ωΑ-Ω])|（|\(cl\.", s)[0]
    toks = _NUM.findall(s)
    return _val(toks[-1]) if toks else None


def _holds(a, rel, b):
    (x, tx), (y, ty) = a, b
    t = tx + ty + 1e-12
    return {"≤": x <= y + t, "<": x < y + t, "≥": x >= y - t, ">": x > y - t}[rel]


def _inequalities(seg: str):
    """(left, relation, right, where) of every comparison on the page: formula rows (the previous row's final value
    against this row's), two numbers on one side of a row, and the verdict line"""
    out = []
    for blk in re.findall(r'<div class="eq[^"]*"[^>]*>(.*?)</div>', seg, re.S):
        rows = re.findall(r'<span class="lhs">(.*?)</span><span class="rel">(.*?)</span><span class="rhs">(.*?)'
                          r'</span><span class="tag">', blk, re.S)
        prev = None
        for lhs, rel, rhs in rows:
            lt, rt = _text(lhs), _text(rhs)
            rel = _text(rel).strip()
            if rel in "≤≥<>" and rel and not lt.strip() and prev is not None:
                left, right = _last(prev), _first_after_eq(rt)
                if left and right:
                    out.append((left, rel, right, prev + " " + rel + " " + rt))
            for a_, r_, b_ in re.findall(r"(" + _NUM.pattern + r")\s*(?:(?:mm|kN|MPa|kN·m)\s*)?([≤≥<>])\s*(" +
                                         _NUM.pattern + r")", rt):
                out.append((_val(a_), r_, _val(b_), rt))
            prev = rt if rel else prev
    for box in re.findall(r'<div class="box box-(?:ok|fail)">(.*?)</div></div>', seg, re.S):
        t = _text(box)
        mt = re.search(r"= (" + _NUM.pattern + r")\s*\S*\s*([≤≥<>]) [^=]*= (" + _NUM.pattern + ")", t)
        if mt:
            out.append((_val(mt.group(1)), mt.group(2), _val(mt.group(3)), t))
    return out


def _first_after_eq(s: str):
    """the value a comparison row ends with: the last number before 取 / a note"""
    return _last(s)


def test_inequalities_hold(written):
    name, chk, seg, vals, entry = written
    found = _inequalities(seg)
    assert found, name
    bad = [(a, r, b, w) for a, r, b, w in found if not _holds(a, r, b)]
    assert not bad, (name, bad)


def test_failed_condition_in_the_verdict(written):
    """a check not met says why in its verdict box; a box of a met check says no condition failed"""
    name, chk, seg, vals, entry = written
    box = re.search(r'<div class="box box-(ok|fail)">(.*?)</div></div>', seg, re.S)
    assert box and box.group(1) == ("ok" if chk.ok else "fail"), name
    if chk.failed:
        assert jr.message(chk.failed[0]) in _text(box.group(2)).replace("−", "-") or \
            jr.message(chk.failed[0]).replace("−", "-") in _text(box.group(2)).replace("−", "-"), name


# ── the H cases of the review ───────────────────────────────────────────────
def _case(prefix):
    return next(c for n, c in CASES if n.startswith(prefix))


def test_review_high_cases():
    # H1: torsion with shear, βt ≠ 0.75 and βt = 0.75: βt multiplies the concrete term
    for p in ("Q2", "Q4"):
        seg, v, e = _write(_case(p))
        assert "β<sub>t</sub>(0.35" in seg.replace("<i>", "").replace("</i>", "") and e.guards
    # H2: signed edge stresses, temperature term written, class B σG
    seg, v, _ = _write(_case("K1 class A"))
    assert V(v, "sst") < 0 and "−\\frac" not in seg
    seg, v, _ = _write(_case("K5"))
    assert "st_t" in v
    seg, v, _ = _write(_case("K3"))
    assert "sG" in v
    # H4: both sides of la/10 ≤ te ≤ la/5
    seg, v, _ = _write(next(c for n, c in CASES if n.startswith("B thin")))
    assert "&lt;" in seg or "<span class=\"rel\">&lt;</span>" in seg or ">&lt;<" in seg
    # H5: the first match names the check
    assert jr.check_name(_case("P1")) == "斜截面抗裂"
    # H6: no indirect reinforcement, 0.9ηsβfcdAln written
    seg, v, e = _write(_case("L2"))
    assert "Fu" in v and _close(V(v, "Fu"), _case("L2").resistance, 1e-4)
    # H7: 7.2.4 limit 0.80 f'ck
    seg, v, e = _write(_case("X2"))
    assert "0.80" in seg and e.guards
    # H8: the ξ of the design
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    des = jtg.stirrup_design(sec, 600.0, legs=2, dia=10, xi=0.8)
    rep = Report("s")
    jr.stirrup_design_block(rep.chapter("c"), des, where="x")
    h = rep.html()
    assert "ξ = 0.8" in h and "0.8 × 1 × 600" in h


# ── independent references: the code's formulas with its table values ──────
def A_(n, d):
    return n * pi * d * d / 4


def _hand():
    """(case, value, what): computed here from the code's formulas and tables 3.1.3–3.2.3 (fcd, ftd, fsd typed in)"""
    out = []
    fsd = 330.0
    # 5.2.2, rectangle 300×600 C30 (fcd 13.8), 4Φ25 at 45
    As = A_(4, 25)
    x = fsd * As / (13.8 * 300)
    out.append(("F1", 13.8 * 300 * x * (555 - x / 2) / 1e6, "resistance"))
    # 5.2.3, T in the web: C30, 8Φ28 at 100, A's 4Φ16 at 40
    As, Asc = A_(8, 28), A_(4, 16)
    x = (fsd * As - fsd * Asc - 13.8 * 600 * 100) / (13.8 * 300)
    out.append(("F2", (13.8 * (300 * x * (1100 - x / 2) + 600 * 100 * 1050) + fsd * Asc * 1060) / 1e6, "resistance"))
    # 5.2.4-1, A'p compressed, x < 2a' (C50 fpd 1260, f'pd 390)
    Fs, Fp = fsd * A_(6, 16), (390 - 200) * 4 * 139
    a_ = (Fs * 45 + Fp * 100) / (Fs + Fp)
    out.append(("F11", (1260 * 6 * 139 * (1200 - 100 - a_) + fsd * A_(4, 16) * (1200 - 50 - a_)) / 1e6, "resistance"))
    # 5.2.9: T C40 b 200, h0 1040, 4Φ22, 2Φ10@150, bent 2Φ25 + 2Φ22 at 45°
    P = 100 * A_(4, 22) / (200 * 1040)
    Vcs = 0.45e-3 * 1.1 * 200 * 1040 * math.sqrt((2 + 0.6 * P) * math.sqrt(40) * A_(2, 10) / (150 * 200) * 330)
    out.append(("S1", Vcs + 0.75e-3 * 330 * (2 * 490.9 + 2 * 380.1) * math.sin(pi / 4), "resistance"))
    # 5.2.12 × 1.25: slab C30 (ftd 1.39), h0 = 250 − 30
    out.append(("S3", 1.25 * 0.5e-3 * 1.39 * 1000 * 220, "resistance"))
    # 5.3.1: 500×500 C30, 6Φ25, l0/b = 10 → φ 0.98
    out.append(("C9", 0.9 * 0.98 * (13.8 * 250000 + 330 * A_(6, 25)) / 1e3, "resistance"))
    # 5.3.4: large eccentricity, symmetric 5Φ25, 500×800 C30, l0 6000
    e0, h0 = 750.0, 750.0
    eta = 1 + (6000 / 800) ** 2 / (1300 * e0 / h0)
    e = eta * e0 + 400 - 50
    aa, bb, cc = 13.8 * 500 / 2, 13.8 * 500 * (e - h0), -fsd * A_(5, 25) * (h0 - 50)
    x = (-bb + math.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
    out.append(("C1", 13.8 * 500 * x / 1e3, "resistance"))
    # 5.3.8: circle D1500 28Φ28 C35 (fcd 16.1), Nd 9800, Md 3100, l0 18000
    r, rs, Ac, As = 750, 690, pi * 750 ** 2, A_(28, 28)
    e0 = 3100 / 9800 * 1e3
    eta = 1 + (18000 / 1500) ** 2 * min(0.2 + 2.7 * e0 / 1440, 1) / (1300 * e0 / 1440)

    def NM(a):
        at = max(1.25 - 2 * a, 0)
        N = a * 16.1 * Ac * (1 - math.sin(2 * pi * a) / (2 * pi * a)) + (a - at) * fsd * As
        M = 2 / 3 * 16.1 * Ac * r * math.sin(pi * a) ** 3 / pi + \
            fsd * As * rs * (math.sin(pi * a) + math.sin(pi * at)) / pi
        return N, M
    lo, hi = 0.3, 0.9
    for _ in range(100):
        mid = (lo + hi) / 2
        N, M = NM(mid)
        lo, hi = (mid, hi) if M - eta * e0 * N > 0 else (lo, mid)
    out.append(("C4", NM(lo)[0] / 1e3, "resistance"))
    # 5.4.1 HRB500 at 330: 8Φ22
    out.append(("T1", 330 * A_(8, 22) / 1e3, "resistance"))
    # 5.4.2-1, -2: 4Φ22 each face, 400×600, e0 100
    As = A_(4, 22)
    out.append(("T2", min(330 * As * 510 / 155, 330 * As * 510 / 355) / 1e3, "resistance"))
    # 5.5.1: 400×800 C30 (ftd 1.39), 2Φ12@100, Ast 6×491, core 320×720
    Wt = 400 ** 2 / 6 * (3 * 800 - 400)
    zeta = min(330 * 2946 * 100 / (330 * A_(1, 12) * 2080), 1.7)
    out.append(("Q1", (0.35 * 1.39 * Wt + 1.2 * math.sqrt(zeta) * 330 * A_(1, 12) / 100 * 320 * 720) / 1e6,
                "resistance"))
    # 5.5.4-1, -2: Vd 300, Td 95 (βt = min(1.5/(1 + 0.5 Vd Wt/(Td b h0)), 1) = 1)
    bt = min(1.5 / (1 + 0.5 * 300e3 * Wt / (95e6 * 400 * 750)), 1.0)
    K = (2 + 0.6 * 100 * A_(4, 22) / (400 * 750)) * math.sqrt(30) * 330
    coef = 0.5e-4 * (10 - 2 * bt) * 400 * 750
    q = A_(1, 12) / 100 - (300 / coef) ** 2 / K * 400 / 2
    out.append(("Q2", (bt * 0.35 * 1.39 * Wt + 1.2 * math.sqrt(zeta) * 330 * q * 320 * 720) / 1e6, "resistance"))
    # 6.4.3: example span, C40 T, 10Φ28 at 110, Ms 769.3, Ml 567.7
    As = A_(10, 28)
    sss = 769.3e6 / (0.87 * As * 990)
    out.append(("W0 flexural", (1 + 0.5 * 567.7 / 769.3) * sss / 2e5 * (50 + 28) / (0.36 + 1.7 * 0.1), "demand"))
    # 6.4.4-1, C3 1.2: 6Φ20, 400×400, Ns 300, Nl 200
    As = A_(6, 20)
    rte = max(As / 160000, 0.01)
    out.append(("W2", 1.2 * (1 + 0.5 * 200 / 300) * 300e3 / As / 2e5 * (35 + 20) / (0.36 + 1.7 * rte), "demand"))
    # 6.5.3: L/600
    out.append(("D1", 16000 / 600, "resistance"))
    # 7.2.4-1 limit, f'cu 25 → f'ck 16.7
    out.append(("X2", 0.80 * 16.7, "resistance"))
    # 7.2.8 limit, f'cu 45 → f'ck = 29.6 (between C40 26.8 and C50 32.4)
    out.append(("X1", 0.70 * (26.8 + (32.4 - 26.8) / 2), "resistance"))
    # 5.7.1, 5.7.2: C50 at f'cu 45 (fcd 20.5), β = 3, mesh 5Φ12 (113) at 280, s 50
    rv = 2 * 5 * 113 * 280 / (280 * 280 * 50)
    F1 = 1.3 * 3 * 20.5 * 1.8e5 / 1e3
    F2 = 0.9 * (3 * 20.5 + 2 * rv * math.sqrt(280 * 280 / 2e5) * 330) * 1.8e5 / 1e3
    out.append(("L1", min(F1, F2), "resistance"))
    # 5.6.1, 5.6.3
    out.append(("U1", 0.7 * 1.65 * 3220 * 205 / 1e3, "resistance"))
    out.append(("U4", 0.7 * (1 - 0.15 * 450 / 500) * 1.39 * (800 + 2200) / 2 * 700 / 1e3, "resistance"))
    # 8.8.2-5…-7: β = 1.3
    return out


@pytest.mark.parametrize("case, value, what", _hand(), ids=[c for c, _, _ in _hand()])
def test_hand_calculations(case, value, what):
    chk = _case(case)
    prog = getattr(chk, what)
    assert _close(value, prog, 2e-3), (case, value, prog)
    seg, vals, entry = _write(chk)
    shown = {"resistance": ("Mu", "Tu", "Vu", "Nu", "Fu"), "demand": ("Wcr",)}[what]
    keys = [k for k in shown if k in vals]
    if keys:  # the result printed
        assert _close(V(vals, keys[0]), value, 2e-3)


# ── no English left in Chinese reports ──────────────────────────────────────
_ALLOWED = re.compile(
    r"^(mm|kN|MPa|PTFE|JTG|GB|JT|HRB\d+|HPB\d+|RRB\d+|C\d+|CR|NR|Φ|IRHD|max|min|sin|cos|tan|ln|lg|SIMP|ID|dB|"
    r"[A-Za-z]{1,2}|[A-Z][a-z]{0,3}|[a-z]{1,3}|[A-Za-z]+\d+[A-Za-z]*)$")
_ENGLISH = re.compile(r"[A-Za-z]{4,}|\b(?:the|and|not|for|are|bar|see|top|use|all|web|per|one|two|end|off|out|its|"
                      r"with|from|than|bars|less|more|is|of|to|in|on|at|by|or|as|be)\b")


def _english(s: str) -> list[str]:
    s = re.sub(r"\$[^$]*\$", "", s)  # inline math
    words = re.findall(r"[A-Za-z][A-Za-z']*", s)
    bad = [w for w in words if _ENGLISH.fullmatch(w) and w not in ("Acor", "Wcr", "Mcr", "Mud", "Nud", "Ucor", "Asvu",
                                                                    "Asbu", "Asv", "Ast", "fcuk", "Acr", "Icr", "Nl",
                                                                    "Ate", "IRHD", "PTFE", "SIMP", "Mudb", "fpk")]
    return bad


@pytest.fixture(scope="module")
def every_check():
    return rc.all_checks()


def test_chinese_names(every_check):
    bad = sorted({(c.name, jr.check_name(c)) for c in every_check if _english(jr.check_name(c))})
    assert not bad, bad


def test_chinese_messages(every_check):
    bad = set()
    for c in every_check:
        for x in list(c.failed) + list(c.warnings) + list(c.notes):
            zh = jr.message(x)
            if _english(zh):
                bad.add((x, zh))
    assert not bad, sorted(bad)


def test_chinese_detail_tables(every_check):
    """the general form and the detailing reports: the names of the values and the rules in Chinese"""
    bad = set()
    for c in every_check:
        rep = Report("t")
        jr.write_check(rep.chapter("c"), c, figure=False, formulas=False)
        h = rep.html()
        for cell in re.findall(r"<tr><td[^>]*>(.*?)</td>", h, re.S):
            t = re.sub(r'<span class="m">.*?</span>', "", cell, flags=re.S)
            t = re.sub(r"<[^>]+>", "", t)
            if _english(t):
                bad.add((c.name, t))
    assert not bad, sorted(bad)[:40]


def test_translation_tables_cover_the_library():
    """every literal check name and message pattern of the library has a translation (the catalogue in
    _report_text is checked against the harvested texts in test_chinese_*)"""
    assert len(TX.NAMES) > 100 and len(TX.MESSAGES) > 100 and len(TX.KEYS) > 200
