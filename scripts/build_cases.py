#!/usr/bin/env python3
"""Build docs/assets/cases.js for the homepage from two sources:

  1. the paper's four reference trajectories (appendix_case_studies.tex,
     trajbox/trajlisting blocks), and
  2. docs/cases/*.txt -- four more, excerpted from the nine-model panel logs
     (baseline_agent/results/models/<model>/<track>/traces/ in the run archive)
     using the same markup. First line: `TITLE: <key> | <model> | <note>`.

Usage: python scripts/build_cases.py <path to appendix_case_studies.tex>
The paper source is not part of this repository; pass its path explicitly.

The traces are verbatim agent logs; the only transformation here is markup:
  |B|x|B|   -> <b>            our turn markers / header labels
  |EL|x|EL| -> <i class=el>   our elisions and margin notes
  |HL|x|HL| -> <span class=hl> interaction-design differences across the 2x2
  |N|x|N|   -> <span class=n>  our own words (task summary, score)
Everything else is HTML-escaped as-is.
"""
import html, json, re, sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("usage: build_cases.py <path to the paper's appendix_case_studies.tex>")
TEX = Path(sys.argv[1])
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
# metadata.yaml / eval/reference_metrics.json in the task tree
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
   inputs=[("year_decimal", "t", "yr", "Decimal year of the monthly observation (midpoint of the calendar month, NOAA convention). The task exposes five more columns — year, month, ndays, sdev, unc — that are calendar/QC fields, not physical predictors; the agent used only t.")],
   data="No rows preloaded. The agent designs its own measurements with <experiment> inside t ∈ [1958, 2026]; the simulator returns residual-calibrated noisy observations.",
   baselines=["noaa_curve_fit_k4 (reference, 12 constants)", "noaa_curve_fit_1989", "bacastow_1985_quadratic", "keeling_1960_linear"],
   caps="≤ 12 global constants · no fit() · no group_id",
   scored="S_S: does the submitted form match the hidden simulator's structure (0 / 0.25 / 0.5 / 0.75 / 1.0)",
   outcome="S_S = 0.25 (GPT-5.5)",
   lesson="The model fits the hidden exponential-accumulation family in its own analysis, then submits a quadratic. On the Real version of the same task it beats the 12-constant NOAA reference (S_N = 0.78)."),
 "optical_dispersion_sellmeier": dict(
   task_id="optical_dispersion_sellmeier__refractive_index", setting="Parallel", group="multi",
   discipline="Earth & Physics", domain="physics / optics", license="CC0-1.0",
   context="The refractive index of a transparent crystal is the factor by which it slows light, and its variation with wavelength is the optical dispersion studied in crystal optics and condensed-matter physics.",
   target=("refractive_index", "n", "dimensionless", "Real part of the refractive index at the specified vacuum wavelength, measured by the minimum-deviation goniometer method on single-crystal prisms at room temperature."),
   inputs=[("wavelength_um", "λ", "µm", "Vacuum wavelength of light at which the refractive index was measured; near-UV through mid-infrared.")],
   data="No rows preloaded. 14 crystal groups are queryable; ≤ 10 experiment calls of ≤ 20 points each. (The Real version holds out 3 of 14 crystals: CdSe, TiO₂, ZnWO₄.)",
   baselines=["cauchy_1836_eq15 (4-term Cauchy, reference)", "cauchy_1836_eq55", "sellmeier_1871"],
   caps="0 global fitted constants · ≤ 4 per-group parameters · one shared form",
   scored="S_S: structural match to the hidden law, up to algebraic equivalence",
   outcome="S_S = 1.0 (GPT-5.5)",
   lesson="Recovers the hidden n² = A + B/(λ² − C) + D·λ² family exactly. On the Real version the same model submits a Cauchy series and scores S_N = 0.21 — structure recovery and reference-beating are different outcomes."),
 # ---- cases excerpted from the panel logs (docs/cases/*.txt) ----
 "dna_melting": dict(
   task_id="dna_melting_temperature_khandelwal__Tm", setting="Real", group="single",
   discipline="Biology", domain="biology / biochemistry", license="CC-BY-4.0",
   context="The melting temperature of a double-stranded DNA oligonucleotide is the temperature at which half of the duplexes dissociate into single strands, a key parameter in molecular biology.",
   target=("Tm", "T_m", "°C", "Experimentally measured DNA duplex melting temperature — the temperature at which 50% of double-stranded DNA has dissociated into single strands, determined by UV-absorbance hyperchromic effect."),
   inputs=[("E", "E", "dimensionless", "DNA strength parameter per base — sum of overlapping dinucleotide step strength values divided by sequence length."),
           ("N", "N", "bp", "Oligonucleotide sequence length in base pairs."),
           ("salt_M", "[Na+]", "M", "Sodium ion concentration of the solution."),
           ("dna_conc_M", "C_DNA", "M", "Total nucleotide strand concentration.")],
   data="344 train rows, 100 test rows. Train C_DNA spans 10⁻⁶–10⁻⁴ M; the test set holds it at 2×10⁻⁶ M. E and [Na+] test ranges sit inside the train ranges.",
   baselines=["khandelwal_2010 (reference; Eq. 1, 5 constants; test RMSE 4.16 °C)"],
   caps="≤ 5 global constants · no per-group parameters · must not define fit()",
   scored="S_N: RMSE on the held-out test set, normalized against Khandelwal 2010 · S_V: 5-item frozen rubric",
   outcome="S_N = 0.788 · S_V = 1.0 (5/5) — GPT-5.5",
   lesson="A discovery-moat task: no model cold-recalls a formula for it. The agent builds a nearest-neighbour strength term, a quadratic log-salt correction and a length-scaled concentration term from the data, and cuts the published RMSE from 4.16 °C to 1.76 °C."),
 "hacks_law": dict(
   task_id="hacks_law_river_length_hydrosheds__main_length_km", setting="Real", group="single",
   discipline="Ecology & Hydrology", domain="earth_science / hydrology", license="HydroSHEDS-v1-Derivative",
   context="In geomorphology and hydrology, each river drainage basin has a main-stem channel running from its headwater divide to its outlet, characterized using a global hydrographic database.",
   target=("main_length_km", "L", "km", "Main-stem river length, measured along the longest channel from the drainage divide to the basin outlet."),
   inputs=[("upland_area_skm", "A", "km²", "Total upstream drainage area at the outlet."),
           ("catch_area_skm", "A_c", "km²", "Local catchment area of the outlet segment alone."),
           ("segment_length_km", "L_s", "km", "Length of the terminal outlet segment alone."),
           ("endorheic", "e", "—", "1 if the basin terminates at an inland sink, 0 if it reaches the ocean."),
           ("strahler_order", "ω", "—", "Strahler stream order of the outlet segment.")],
   data="28,340 train rows, 7,085 test rows. Train basins have A ∈ [100, 1,088] km²; test basins run from 1,088 to 5.9×10⁶ km² — an extrapolation of almost four orders of magnitude.",
   baselines=["hack_1957 (Eq. 3; L = c·A^h with h = 0.6; test log-MAE 0.099)", "dodds_rothman_2000 (Eq. 36; h = 0.5; test log-MAE 0.105)"],
   caps="≤ 2 global constants · no per-group parameters · must not define fit()",
   scored="S_N: log-MAE on the held-out test set, normalized against the stronger reference · S_V: 5-item frozen rubric",
   outcome="S_N = 0.580 · S_V = 1.0 (5/5) — GPT-5.5",
   lesson="With only two constants allowed, the agent keeps Hack's power law but adds the terminal segment length as a free additive term and fixes the exponent at 0.55. Test log-MAE 0.083 against 0.099 (Hack) and 0.105 (Dodds–Rothman), on basins far larger than any it trained on. Every model but GPT-5.4-mini passes the full validity rubric here."),
 "eclipsing_binary": dict(
   task_id="eclipsing_binary_mass_luminosity_debcat__log_L_Lsun", setting="Parallel", group="single",
   discipline="Astronomy", domain="astronomy / stellar_astrophysics", license="CC-BY-4.0",
   context="Eclipsing binaries are pairs of stars whose mutual eclipses allow precise measurement of stellar properties, used in stellar astrophysics to characterise main-sequence stars.",
   target=("log_L_Lsun", "log₁₀(L/L☉)", "dex", "Base-10 logarithm of bolometric stellar luminosity in solar luminosities."),
   inputs=[("log_M_Msun", "log₁₀(M/M☉)", "dex", "Base-10 logarithm of stellar mass in solar masses. (The Real version adds a metallicity [M/H] column as a distractor; the reference laws use mass only.)")],
   data="No rows preloaded. log₁₀(M/M☉) ∈ [−0.976, 1.436]; ≤ 10 experiment calls, ≤ 23 points × 3 samples each (≤ 690 rows). The Real version trains on 238 pre-2018 stars and tests on 355 post-2018 stars.",
   baselines=["eker_2018_six_piece (six-piece broken power law, 12 constants)", "henry_1993_two_piece", "kuiper_1938_pure_power"],
   caps="≤ 12 global constants · no fit() · no group_id",
   scored="S_S: structural match to the hidden law (0 / 0.25 / 0.5 / 0.75 / 1.0)",
   outcome="S_S = 1.0 — DeepSeek-V4 Pro (also Claude Opus 4.8, Gemini 3.5 Flash)",
   lesson="The published laws are chains of straight segments in log–log space; the hidden law is one smooth cubic. The agent fits the textbook broken power laws first, sees the cubic beat them (RMSE 0.170 vs 0.186 on group means), and submits the cubic — the paper's flagship structure-recovery case."),
 "running_endurance": dict(
   task_id="running_endurance_iaaf__velocity", setting="Parallel", group="single",
   discipline="Social Sciences", domain="social_science / sports_science", license="CC-BY-SA-4.0",
   context="This task uses athletics world records for human running events across a range of race distances, relating to exercise physiology.",
   target=("velocity", "v̄", "m/s", "Average race velocity over the full event distance, computed as race distance divided by the world-record finishing time."),
   inputs=[("distance_m", "d", "m", "Official race distance for the event, sprint to ultra-marathon."), ("sex_M", "s_M", "—", "1 for men, 0 for women.")],
   data="No rows preloaded. d ∈ [60, 100,000] m; ≤ 10 experiment calls, ≤ 10 points × 3 samples each (≤ 300 rows). The Real version has 28 train records at 60 m–16 km and 14 test records at 20–100 km.",
   baselines=["riegel_1981 (v = K·d^(1−b), per-sex constants)", "emig_2020"],
   caps="≤ 4 global constants · no fit() · no group_id",
   scored="S_S: structural match to the hidden law",
   outcome="S_S = 0.25 — GPT-5.5, and every other model",
   lesson="The hidden law adds a second, steeper power-law term (the anaerobic component) to Riegel's single power law. The agent tests Riegel's form, finds an inverse-log form fits better, and submits that; a sum of two power laws is never on its candidate list. All nine models land at 0.25."),
}

found = {}
for title, body in boxes:
    task, model, note = detex_title(title)
    found[task] = {"key": task, "model": model, "note": note, "card": CARDS[task], "trace_html": convert(body)}

CASES_DIR = Path(__file__).resolve().parent.parent / "docs" / "cases"
for f in sorted(CASES_DIR.glob("*.txt")):
    head, body = f.read_text().split("\n", 1)
    assert head.startswith("TITLE:"), f
    key, model, note = [x.strip() for x in head[len("TITLE:"):].split("|")]
    found[key] = {"key": key, "model": model, "note": note, "card": CARDS[key], "trace_html": convert(body)}

# display order: row 1 = Real, row 2 = Parallel; one discipline each where possible
ORDER = ["dna_melting", "proton_em_form_factor", "hacks_law", "co2_adsorption_toth",
         "eclipsing_binary", "keeling_curve", "running_endurance", "optical_dispersion_sellmeier"]
missing = [k for k in ORDER if k not in found]
assert not missing, f"cases not found: {missing}"
cases = [found[k] for k in ORDER]

OUT.write_text("// Generated by scripts/build_cases.py from the paper's Appendix (case studies).\n"
               "// Traces are verbatim agent logs; see the appendix for the colour legend.\n"
               "window.SCILAWS_CASES = " + json.dumps(cases, ensure_ascii=False, indent=1) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes) with {len(cases)} cases:",
      [c['key'] for c in cases])
