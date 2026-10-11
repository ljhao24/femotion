"""Design-code checks on top of femotion's analyses.

    jtg   Chinese highway bridge codes: actions and combinations (JTG D60-2015), reinforced-concrete ultimate limit
          state section resistance (JTG 3362-2018)

Codes only add the code-specific parts (loads, factors, resistance formulas). The generic parts they rely on live in
the analysis modules, e.g. influence lines and moving-load envelopes in femotion.frame.influence.

docs/jtg.md lists every clause used and the review against the official texts. femotion is not a substitute for a
qualified engineer.
"""
