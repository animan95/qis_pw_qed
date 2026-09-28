#!/usr/bin/env python
"""Figures for the committed results (reads results/*.json). Okabe-Ito
colorblind-safe palette; thin marks, recessive axes, direct labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/projectsn/mp1009_1/am4655/qis_pw_qed/results")
FIG = Path("/projectsn/mp1009_1/am4655/qis_pw_qed/figures"); FIG.mkdir(exist_ok=True)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          skyblue="#56B4E9", yellow="#F0E442", purple="#CC79A7", ink="#222222", mut="#777777")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ECECEC", "grid.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": OI["ink"],
    "axes.labelcolor": OI["ink"], "xtick.color": OI["ink"], "ytick.color": OI["ink"],
})
load = lambda n: json.load(open(RES / f"{n}.json"))["data"]
# lowest optically-active singlet (consistent, physical "first bright excitation")
bright_singlet = lambda st: min((x for x in st if x["S2"] < 0.5 and x["mu_norm"] > 0.02), key=lambda x: x["dE_eV"])
bright_td = lambda st: min((x for x in st if x["f"] > 0.005), key=lambda x: x["dE_eV"])

# ---------- Fig 1: LiCN bright excitation by method, gas vs Ar-embedded ----------
gas, emb = load("gas_licn"), load("embedded_licn")
methods = ["TDDFT/PBE", "CASSCF\n(QIS_QE)", "EOM-CCSD"]
gas_e = [bright_td(gas["tddft_pbe"])["dE_eV"], bright_singlet(gas["casscf_qiskit_exactdiag"])["dE_eV"], min(gas["eomccsd_singlet_dE_eV"])]
emb_e = [bright_td(emb["tddft_pbe"])["dE_eV"], bright_singlet(emb["casscf_qiskit_exactdiag"])["dE_eV"], min(emb["eomccsd_singlet_dE_eV"])]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 4.3))
bars = [ax.bar(x - w/2, gas_e, w, color=OI["skyblue"], label="gas phase"),
        ax.bar(x + w/2, emb_e, w, color=OI["vermillion"], label="Ar-embedded")]
for grp in bars:
    for b in grp:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.5, color=OI["ink"])
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("lowest bright singlet excitation (eV)")
spread = max(gas_e) - min(gas_e)
ax.set_title(f"LiCN first bright excitation: methods disagree by ~{spread:.1f} eV", fontsize=11.5)
ax.set_ylim(0, 6.9); ax.legend(frameon=False, loc="upper left")
ax.annotate("", xy=(2, 6.35), xytext=(0, 6.35), arrowprops=dict(arrowstyle="<->", color=OI["mut"], lw=1))
ax.text(1, 6.45, f"~{spread:.1f} eV method disagreement", ha="center", color=OI["mut"], fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig1_method_comparison.png"); plt.close(fig)

# ---------- Fig 2: WFT correction of subsystem TDDFT (scissor) ----------
wc = load("wft_corrected_subsystem_tddft")
td, cas = wc["tddft_fragment_bright_eV"], wc["casscf_correction_bright_eV"]
fig, ax = plt.subplots(figsize=(5.8, 4.4))
for t, c in zip(td, cas):
    ax.hlines(t, -0.18, 0.18, color=OI["blue"], lw=3)
    ax.hlines(c, 0.82, 1.18, color=OI["orange"], lw=3)
    ax.annotate("", xy=(0.82, c), xytext=(0.18, t), arrowprops=dict(arrowstyle="->", color=OI["mut"], lw=1.2))
ax.text(0, min(td)-0.13, "TDDFT/PBE\nfragment", ha="center", va="top", color=OI["blue"], fontsize=10)
ax.text(1, max(cas)+0.08, "CASSCF\ncorrection", ha="center", va="bottom", color=OI["orange"], fontsize=10)
ax.text(0.5, (np.mean(td)+np.mean(cas))/2, f"+{np.mean(cas)-np.mean(td):.2f} eV\nscissor", ha="center", color=OI["ink"], fontsize=10)
ax.text(0.5, min(td)-0.35, r"transition dipole |$\mu$|: 0.02 $\to$ 0.13 a.u.  (~6$\times$)", ha="center", color=OI["mut"], fontsize=9)
ax.set_xlim(-0.55, 1.7); ax.set_ylim(min(td)-0.6, max(cas)+0.4)
ax.set_ylabel("excitation energy (eV)"); ax.set_xticks([])
ax.set_title("WFT correction of subsystem TDDFT (LiCN bright doublet)", fontsize=11.5)
fig.tight_layout(); fig.savefig(FIG / "fig2_wft_correction.png"); plt.close(fig)

# ---------- Fig 3: cavity-QED polaritons of the embedded dimer ----------
q = load("embedded_dimer_qed")
bare = q["excitons_eV"][int(np.argmax(q["osc"]))]
models = {"JC": (q["qed"]["jc"], OI["blue"]), "PF": (q["qed"]["pf"], OI["orange"]),
          "QED-TDDFT": (q["qed"]["tddft"], OI["green"])}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.axhline(bare, color=OI["mut"], lw=1.3, ls="--")
ax.text(len(models)-0.55, bare, "  bare bright exciton", va="center", color=OI["mut"], fontsize=9)
for i, (name, (m, col)) in enumerate(models.items()):
    pol = m["polariton_eV"]
    for p in pol:
        ax.hlines(p, i-0.26, i+0.26, color=col, lw=3.5)
    lo, hi = min(pol), max(pol)
    ax.annotate("", xy=(i, hi), xytext=(i, lo), arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
    ax.text(i+0.3, (lo+hi)/2, f"{m['rabi_meV']:.0f} meV", va="center", color=col, fontsize=9.5)
ax.set_xticks(range(len(models))); ax.set_xticklabels(models.keys())
ax.set_xlim(-0.6, len(models)-0.1); ax.set_ylabel("polariton energy (eV)")
ax.set_title(r"Cavity-QED polaritons, embedded LiCN dimer ($\lambda$=0.05)", fontsize=11.5)
fig.tight_layout(); fig.savefig(FIG / "fig3_qed_polaritons.png"); plt.close(fig)

print("wrote:", *(p.name for p in sorted(FIG.glob("*.png"))))
