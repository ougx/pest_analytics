"""
Fit of Napa2026_03 vs Napa2026_04 as on the Fit Changes page: measurement phi by iteration with each run's
own weights, then both best iterations scored with the same weights (04's, 03's, each run's own,
weight_orig), by obs group (grpType) and by site (grpSite). Supports note section 2.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
RUN_03 = r"D:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT_LSPC\PEST\zzArchive\napa2026_03.zip"
RUN_04 = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import pandas as pd
import compare
from utils import parse_zip

pd.set_option("display.width", 200)
r03, r04 = parse_zip({"pestpath": RUN_03}, False, True), parse_zip({"pestpath": RUN_04}, False, True)
for label, r in (("03", r03), ("04", r04)):
    phi = {i: round(compare.measurement_phi(rei)) for i, rei in sorted(r[4].items())}
    print(f"Napa2026_{label} phi by iteration (own weights): {phi}; best: {compare.pick_iteration(r[4], compare.BEST)}")
i03, i04 = compare.pick_iteration(r03[4], compare.BEST), compare.pick_iteration(r04[4], compare.BEST)
print()

# 04 is the reference run, 03 the other run, compared 03 -> 04
for choice in compare.WEIGHTS:
    w_from, w_to = compare.run_weights(choice, r04[2], r03[2], to_is_ref=True)
    paired = compare.paired_residuals(r03[4][i03], r04[4][i04], w_from, w_to, r04[2].set_index("obsname").grpType)
    fit = compare.fit_by_group(paired)
    total_from, total_to = fit.phi_from.sum(), fit.phi_to.sum()
    print(f"{choice}: phi 03 = {total_from:.4g}, 04 = {total_to:.4g} ({100 * (total_to / total_from - 1):+.1f}%)")
    if choice == compare.WEIGHTS[0]:
        print(fit[["n", "phi_from", "phi_to", "rmse_from", "rmse_to", "drmse"]].round(3).to_string())
print()

w_from, w_to = compare.run_weights(compare.WEIGHTS[0], r04[2], r03[2], to_is_ref=True)
sites = compare.fit_by_group(compare.paired_residuals(r03[4][i03], r04[4][i04], w_from, w_to,
                                                      r04[2].set_index("obsname").grpSite))
s = sites[sites.n > 0]
print(f"sites weighted in both runs: {len(s)}; RMSE better in 04 at {int((s.drmse < 0).sum())}, "
      f"worse at {int((s.drmse > 0).sum())}")
print("largest RMSE increases:")
print(s.sort_values("drmse", ascending=False)[["n", "rmse_from", "rmse_to", "drmse"]].head(5).round(2).to_string())
print("largest RMSE decreases:")
print(s.sort_values("drmse")[["n", "rmse_from", "rmse_to", "drmse"]].head(5).round(2).to_string())
