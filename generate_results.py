#!/usr/bin/env python
"""Generate the committed result set for the qiskit-pyscf embedded-WFT pipeline.

Every result JSON embeds `provenance` (software + code versions). Systems:
  - LiCN (single): gas vs Ar-embedded, by TDDFT / CASSCF(->QIS_QE) / EOM-CCSD
  - LiCN dimer in Ar: embedded per-fragment CASSCF -> Forster -> cavity QED
  - WFT-corrected subsystem TDDFT (scissor CASSCF onto TDDFT fragment states)

Embedding potentials (.snpy) come from converged eDFTpy runs (see README).
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/qis_pw_qed")
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE/src")
from pyscf import tdscf
from casidapy.embed_wft import (build_embedded_mf, casscf_excitations,
                                eomccsd_excitations, forster_couple)
from casidapy.subsystem_coupling import wft_corrected_coupled_casida
from casidapy.qed_post import run_qed_post
import embed_qiskit
from provenance import provenance

EV = 27.211386245988
ANG2BOHR = 1.8897259886
RES = "/projectsn/mp1009_1/am4655/qis_pw_qed/results"
PROV = provenance()
SMALL = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small"
LiCN = "Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310"
DIMER = {
    "A": ("Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310", f"{SMALL}/dimer_u/sub_licn_a.snpy"),
    "B": ("Li 2.770 5.310 0.000; C 4.723 5.310 0.000; N 5.897 5.310 0.000", f"{SMALL}/dimer_u/sub_licn_b.snpy"),
}


def save(name, meta, data):
    obj = {"provenance": PROV, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "meta": meta, "data": data}
    json.dump(obj, open(f"{RES}/{name}.json", "w"), indent=2, default=float)
    print(f"  saved results/{name}.json")


def states_json(states):
    return [{"dE_eV": w*EV, "mu_au": np.asarray(mu).tolist(),
             "mu_norm": float(np.linalg.norm(mu)), "S2": s2} for w, mu, s2 in states]


def single_licn(tag, snpy):
    """CASSCF(QIS_QE exact-diag) + EOM-CCSD + TDDFT for one LiCN (gas if snpy=None)."""
    mf = build_embedded_mf(LiCN, extemb=snpy)
    cas, mc = casscf_excitations(mf, 6, 6, nstates=8)
    cas_qk = embed_qiskit.casscf_excitations_qiskit(mc, nstates=8)   # qiskit path
    eom, ecorr = eomccsd_excitations(mf, nroots=5)
    mfk = build_embedded_mf(LiCN, extemb=snpy, xc="pbe")
    td = tdscf.TDDFT(mfk); td.nstates = 6; td.verbose = 0; td.kernel()
    return {
        "casscf_qiskit_exactdiag": states_json(cas_qk),
        "casscf_plain_pyscf": states_json(cas),
        "eomccsd_singlet_dE_eV": [x*EV for x in eom],
        "tddft_pbe": [{"dE_eV": float(td.e[i]*EV), "f": float(td.oscillator_strength()[i])}
                      for i in range(len(td.e))],
    }


print("[1/4] gas-phase LiCN ..."); g = single_licn("gas", None)
save("gas_licn", {"system": "LiCN gas-phase", "geom_A": LiCN, "basis": "cc-pVDZ",
                  "active_space": "CAS(6,6)", "methods": ["CASSCF/QIS_QE", "EOM-CCSD", "TDDFT/PBE"]}, g)

print("[2/4] Ar-embedded LiCN (single) ..."); e = single_licn("emb", f"{SMALL}/LiCN/sub_licn.snpy")
save("embedded_licn", {"system": "LiCN embedded in Ar (single, size_small)", "geom_A": LiCN,
                       "basis": "cc-pVDZ", "active_space": "CAS(6,6)", "vemb": "sub_licn.snpy (26-Ar, ONCV, converged)",
                       "methods": ["CASSCF/QIS_QE", "EOM-CCSD", "TDDFT/PBE"]}, e)

print("[3/4] embedded LiCN dimer -> Forster -> QED ...")
frag = {}
for lab, (atom, snpy) in DIMER.items():
    mf = build_embedded_mf(atom, extemb=snpy)
    st, _ = casscf_excitations(mf, 6, 6, nstates=8)
    frag[lab] = [(w, mu) for w, mu, s2 in st if s2 < 0.5][:4]   # bright singlets
R = np.array([0, 0, 5.310]) * ANG2BOHR
w_c, mu_c, osc = forster_couple(frag["A"], frag["B"], R)
bright = int(np.argmax(osc))
pol = (mu_c[bright] / np.linalg.norm(mu_c[bright])).tolist()
qed = run_qed_post(w_c, mu_c, f=osc, models=("pf", "jc", "tddft"),
                   omega_c=float(w_c[bright]), lam_sweep=[0.01, 0.03, 0.05], pol=pol)
qed_summary = {}
for model in ("pf", "jc", "tddft"):
    b = qed["results"][model][-1]
    om = np.atleast_1d(b["omega"]).astype(float)*EV
    pf = np.atleast_1d(b.get("photon_frac", np.zeros_like(om))).astype(float)
    pol_states = [float(x) for x, p in zip(om, pf) if 0.05 < p < 0.95 and abs(x-w_c[bright]*EV) < 1.0]
    qed_summary[model] = {"polariton_eV": pol_states,
                          "rabi_meV": (max(pol_states)-min(pol_states))*1000 if len(pol_states) >= 2 else None}
save("embedded_dimer_qed", {"system": "LiCN dimer in Ar (size_small, sep 5.31 A)",
                            "vemb": "dimer_u/sub_licn_{a,b}.snpy (18-Ar, ONCV+gaussians, converged)",
                            "coupling": "Forster dipole-dipole", "cavity": "omega_c resonant with bright exciton",
                            "lambda_sweep": [0.01, 0.03, 0.05]},
     {"excitons_eV": (w_c*EV).tolist(), "osc": osc.tolist(),
      "davydov_meV": float(abs(w_c[1]-w_c[0])*EV*1000), "qed": qed_summary})

print("[4/4] WFT-corrected subsystem TDDFT ...")
fr, wf = [], []
for lab, (atom, snpy) in DIMER.items():
    mfk = build_embedded_mf(atom, extemb=snpy, xc="pbe")
    td = tdscf.TDDFT(mfk); td.nstates = 6; td.verbose = 0; td.kernel()
    mf = build_embedded_mf(atom, extemb=snpy)
    st, _ = casscf_excitations(mf, 6, 6, nstates=8)
    fr.append({"omega": np.asarray(td.e, float), "dip_tran": np.asarray(td.transition_dipole(), float),
               "Z": np.eye(len(td.e)), "eigenvectors": np.eye(len(td.e))})
    wf.append([(w, mu, s2) for w, mu, s2 in st if s2 < 0.5])
MK = dict(energy_weight_ev=0.15, dipole_weight=1.0, max_cost=0.4)
corr = wft_corrected_coupled_casida(fr, {}, wf, tda=True, match_kwargs=MK)
o = np.argsort(corr["omega"])
save("wft_corrected_subsystem_tddft",
     {"system": "LiCN dimer in Ar", "scheme": "diagonal CASSCF correction + WFT dipoles, TDDFT coupling kept",
      "matching": "dipole-character driven, gated (max_cost=0.4), singlet-filtered"},
     {"tddft_fragment_bright_eV": [float(fr[0]["omega"][i]*EV) for i in range(2)],
      "casscf_correction_bright_eV": [float(w*EV) for w, _, _ in wf[0][:2]],
      "corrected_coupled_eV": [float(corr["omega"][i]*EV) for i in o[:8]],
      "corrected_coupled_mu": [float(np.linalg.norm(corr["d_mode"][i])) for i in o[:8]]})
print("done.")
