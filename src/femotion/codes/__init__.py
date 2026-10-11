"""Design-code checks on top of femotion's analyses.

    jtg   Chinese highway bridge codes: actions and combinations (JTG D60-2015), reinforced-concrete ultimate limit
          state section resistance (JTG 3362-2018)

Codes only add the code-specific parts (loads, factors, resistance formulas). The generic parts they rely on live in
the analysis modules, e.g. influence lines and moving-load envelopes in femotion.frame.influence.

One reference file per code in docs/codes/ (JTG_D60-2015.md, JTG_3362-2018.md, in Chinese): every clause, formula, table
and parameter the program uses, checked against the official texts, with the function that uses it. femotion is not
a substitute for a qualified engineer.
"""
