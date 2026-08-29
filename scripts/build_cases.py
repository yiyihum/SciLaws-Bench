#!/usr/bin/env python3
"""Convert the paper's four reference trajectories (appendix_case_studies.tex,
trajbox/trajlisting blocks) into docs/assets/cases.js for the homepage.

The traces are verbatim agent logs; the only transformation here is markup:
  |B|x|B|   -> <b>            our turn markers / header labels
  |EL|x|EL| -> <i class=el>   our elisions and margin notes
  |HL|x|HL| -> <span class=hl> interaction-design differences across the 2x2
  |N|x|N|   -> <span class=n>  our own words (task summary, score)
Everything else is HTML-escaped as-is.
"""
import html, json, re, sys
from pathlib import Path

TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "path/to/appendix_case_studies.tex")
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "cases.js"

src = TEX.read_text()
boxes = re.findall(r"\\begin\{trajbox\}\{(.*?)\}\n\\begin\{trajlisting\}\n(.*?)\\end\{trajlisting\}",
                   src, re.S)
assert len(boxes) == 4, f"expected 4 trajboxes, found {len(boxes)}"

def detex_title(t):
    m = re.search(r"\\texttt\{(.*?)\}", t); task = m.group(1).replace("\\_", "_")
    model = re.search(r"\}, (.*?) \\quad", t).group(1)
    note = re.search(r"\\quad\((.*?)\)", t).group(1)
    return task, model, note

def convert(body):
    # protect markup tokens, escape the rest, then re-insert tags
    parts = re.split(r"(\|B\||\|EL\||\|HL\||\|N\|)", body)
    out, open_tag = [], None
    TAG = {"|B|": "b", "|EL|": 'i class="el"', "|HL|": 'span class="hl"', "|N|": 'span class="n"'}
    for p in parts:
        if p in TAG:
            if open_tag is None:
                out.append(f"<{TAG[p]}>"); open_tag = p
            else:
                out.append(f"</{TAG[p].split()[0]}>"); open_tag = None
        else:
            out.append(html.escape(p, quote=False))
    return "".join(out).rstrip("\n")

# hand-written task cards: every string below is copied from the task's
# metadata.yaml / eval/reference_metrics.json in SciLaws-Bench
CARDS = {
 "proton_em_form_factor": dict(
   task_id="proton_em_form_factor__GE_over_GD", setting="Real", group="single",
   discipline="Earth & Physics", domain="physics / nuclear_particle_physics", license="CC-BY-4.0",
   context="The proton's internal charge distribution is characterized by its electromagnetic form factors, with the dipole form factor serving as a standard reference shape, a subject in nuclear and hadronic physics.",
   target=("GE_over_GD", "G_E / G_D", "—", "Ratio of the proton electric Sachs form factor to the standard dipole form factor, measured as a function of four-momentum transfer squared."),
   inputs=[("Q2_GeV2", "Q²", "GeV²", "Four-momentum transfer squared in electron-proton elastic scattering.")],
   data="37 train rows, 10 test rows. Train Q² ∈ [0.007, 1.83]; test Q² ∈ [2.07, 5.85] — the split extrapolates.",
   baselines=["arrington_2007 (reference, 8 constants)", "bradford_2006", "hofstadter_1956"],
   caps="≤ 8 global constants · no per-group parameters · must not define fit()",
   scored="S_N: RMSE on the held-out test set, normalized against the Arrington 2007 reference · S_V: frozen validity rubric",
   outcome="S_N = 0.436 (best of nine; 7/9 score 0.000) · S_V = 1.0",
   lesson="All nine models submit physically valid forms; seven still lose to the published reference. Validity and fit are separate questions."),
 "co2_adsorption_toth": dict(
   task_id="co2_adsorption_zeolite_isodb_toth__n_ads", setting="Real", group="multi",
   discipline="Materials & Engineering", domain="materials_science / adsorption", license="NIST-PD",
   context="Microporous solid adsorbents such as zeolites and activated carbons bind CO2 molecules at internal surface sites, a system studied in adsorption science and gas separation.",
   target=("n_ads", "n", "mmol/g", "Amount of CO2 adsorbed per gram of adsorbent at equilibrium, measured by volumetric or gravimetric isotherm experiment."),
   inputs=[("P_bar", "P", "bar", "Equilibrium CO2 gas pressure in the measurement cell.")],
   data="799 train rows across 42 adsorbent groups; 10 unseen test groups, each split into a 108-row calibration window (test_fit) and a 71-row scoring window (test_test).",
   baselines=["kim_2023 (Tóth isotherm, reference)", "swenson_stadie_2019"],
   caps="0 global fitted constants · ≤ 3 per-group parameters · fit() must run < 10 s per group",
   scored="S_N: RMSE on each unseen group's scoring window after the harness re-fits the per-group parameters, averaged over groups",
   outcome="S_N = 0.534 (GPT-5.5)",
   lesson="One shared Tóth form with three per-group parameters transfers to adsorbents the model never saw — the multi-group setting rewards a law, not a lookup table."),
 "keeling_curve": dict(
   task_id="mauna_loa_co2_keeling_curve_noaa__co2_ppm", setting="Parallel", group="single",
   discipline="Earth & Physics", domain="earth_science / atmosphere", license="Public-Domain",
   context="This is the atmospheric carbon dioxide record from Mauna Loa Observatory in Hawaii, a foundational dataset in atmospheric science and climatology.",
   target=("co2_ppm", "c", "ppm", "Monthly mean mole fraction of CO2 in dry air at Mauna Loa Observatory on the WMO X2019 calibration scale."),
   inputs=[("year_decimal", "t", "yr", "Decimal year of the monthly observation (midpoint of the calendar month, NOAA convention).")],
   data="No rows preloaded. The agent designs its own measurements with <experiment> inside t ∈ [1958, 2026]; the simulator returns residual-calibrated noisy observations.",
   baselines=["noaa_curve_fit_k4 (reference, 12 constants)", "noaa_curve_fit_1989", "bacastow_1985_quadratic", "keeling_1960_linear"],
   caps="≤ 12 global constants · no fit() · no group_id",
   scored="S_S: does the submitted form match the hidden simulator's structure (0 / 0.25 / 0.5 / 0.75 / 1.0)",
   outcome="S_S = 0.25 (GPT-5.5)",
   lesson="The model fits the hidden exponential-accumulation family in its own analysis, then submits a quadratic. On the Real version of the same task it beats the 12-constant NOAA reference (S_N = 0.78)."),
 "optical_dispersion_sellmeier": dict(
   task_id="optical_dispersion_sellmeier__refractive_index", setting="Parallel", group="multi",
   discipline="Earth & Physics", domain="physics / crystal optics", license="CC0-1.0",
   context="The refractive index of a transparent crystal is the factor by which it slows light, and its variation with wavelength is the optical dispersion studied in crystal optics and condensed-matter physics.",
   target=("refractive_index", "n", "dimensionless", "Real part of the refractive index at the specified vacuum wavelength, measured by the minimum-deviation goniometer method on single-crystal prisms at room temperature."),
   inputs=[("wavelength_um", "λ", "µm", "Vacuum wavelength of light at which the refractive index was measured; near-UV through mid-infrared.")],
   data="No rows preloaded. 14 crystal groups are queryable; ≤ 10 experiment calls of ≤ 20 points each. (The Real version holds out 3 of 14 crystals: CdSe, TiO₂, ZnWO₄.)",
   baselines=["cauchy_1836_eq15 (4-term Cauchy, reference)", "cauchy_1836_eq55", "sellmeier_1871"],
   caps="0 global fitted constants · ≤ 4 per-group parameters · one shared form",
   scored="S_S: structural match to the hidden law, up to algebraic equivalence",
   outcome="S_S = 1.0 (GPT-5.5)",
   lesson="Recovers the hidden n² = A + B/(λ² − C) + D·λ² family exactly. On the Real version the same model submits a Cauchy series and scores S_N = 0.21 — structure recovery and reference-beating are different outcomes."),
}

cases = []
for title, body in boxes:
    task, model, note = detex_title(title)
    card = CARDS[task]
    cases.append({"key": task, "model": model, "note": note, "card": card, "trace_html": convert(body)})

OUT.write_text("// Generated by scripts/build_cases.py from the paper's Appendix (case studies).\n"
               "// Traces are verbatim agent logs; see the appendix for the colour legend.\n"
               "window.SCILAWS_CASES = " + json.dumps(cases, ensure_ascii=False, indent=1) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes) with {len(cases)} cases:",
      [c['key'] for c in cases])
