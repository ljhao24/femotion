"""Design-code checks on top of femotion's analyses.

    jtg   Chinese highway bridge codes: actions and combinations (JTG D60-2015); reinforced and prestressed concrete
          to JTG 3362-2018: ultimate limit state (with torsion, local compression), serviceability (cracking, crack width,
          deflection, temperature gradients), stresses, composite members, strut-and-tie models and disturbed regions

Codes only add the code-specific parts (loads, factors, resistance formulas). The generic parts they rely on live in
the analysis modules, e.g. influence lines and moving-load envelopes in femotion.frame.influence, imposed
strains in femotion.frame.thermal.

One reference file per code in docs/codes/ (JTG_D60-2015.md, JTG_3362-2018.md, in Chinese): every clause, formula, table
and parameter the program uses, checked against the official texts, with the function that uses it. femotion is not
a substitute for a qualified engineer.
"""
