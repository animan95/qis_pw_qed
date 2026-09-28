#!/usr/bin/env python
"""Phase 0: gas-phase LiCN excitation energies + transition dipoles via
QIS_QE (from_pyscf_casci -> diagonalize_active_space -> transition_dipole),
validated three ways:
  (1) PySCF CASCI trans_rdm1 transition dipoles (same active space)
  (2) QIS_QE run_qeom energies vs exact diagonalization
  (3) PySCF TDDFT (PBE) as the TDDFT baseline the WFT method should improve on

This exercises the full QIS_QE pipeline on LiCN and establishes the reference
numbers before the embedding (Phase 1) folds eDFTpy's v_emb into hcore.
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")
from pyscf import gto, scf, mcscf, fci, dft, tdscf
from pyscf import ao2mo

import tensors, solvers

HARTREE2EV = 27.211386245988
OUT = Path("/projectsn/mp1009_1/am4655/qis_pw_qed/results")
OUT.mkdir(parents=True, exist_ok=True)

# --- LiCN: linear Li-C#N (Li-C 1.953, C-N 1.174 A), matching the pw_qed geometry ---
mol = gto.M(
    atom="Li 0 0 0.0; C 0 0 1.953; N 0 0 3.127",
    basis="cc-pVDZ", spin=0, charge=0, symmetry=False, verbose=0,
)

# ground-state RHF, then state-averaged CASSCF(6,6) for balanced excited orbitals
mf = scf.RHF(mol).run()
NCAS = int(os.environ.get("QIS_NCAS", 6))
NELECAS = int(os.environ.get("QIS_NELECAS", 6))
NROOTS = int(os.environ.get("QIS_NROOTS", 8))
mc = mcscf.CASSCF(mf, NCAS, NELECAS)
mc.fcisolver = fci.direct_spin0.FCI()   # singlet-adapted
mc = mc.state_average_([1.0 / NROOTS] * NROOTS)
mc.run()
ncore = mc.ncore
mo_cas = mc.mo_coeff[:, ncore:ncore + NCAS]

# active-space dipole integrals <p|r|q> in the CAS MO basis (electronic mu = -r)
ao_r = mol.intor("int1e_r", comp=3)                       # (3, nao, nao)
dip_mo = -np.einsum("xuv,up,vq->xpq", ao_r, mo_cas, mo_cas)  # (3, ncas, ncas)

# ============ QIS_QE: build H, exact-diagonalize, transition dipoles ============
ham, num_particles = tensors.from_pyscf_casci(mc)
res = solvers.diagonalize_active_space(ham, num_particles, n_states=NROOTS)
res_s = solvers.filter_singlets(res, tol=1e-3)  # keep S^2 ~ 0

E0 = res.energies[0]
qis_rows = []
for n in range(1, len(res.energies)):
    dE = (res.energies[n] - E0) * HARTREE2EV
    mu = solvers.transition_dipole(res.transition_dms[n - 1], dip_mo)  # (3,) a.u.
    mu2 = float(np.vdot(mu, mu).real)
    f = 2.0 / 3.0 * (res.energies[n] - E0) * mu2                       # oscillator strength
    qis_rows.append(dict(state=n, dE_eV=dE, S2=float(res.spin_squared[n]),
                         mu=[float(x) for x in mu.real], mu_norm=float(np.sqrt(mu2)), f=float(f)))

# ============ cross-check (1): PySCF CASCI trans_rdm1 on same orbitals ============
casci = mcscf.CASCI(mf, NCAS, NELECAS)
casci.fcisolver = fci.direct_spin1.FCI()   # full CI vectors so trans_rdm1 between roots works
casci.fcisolver.nroots = NROOTS
_ck = casci.kernel(mc.mo_coeff)
e_casci, ci = _ck[0], _ck[2]   # kernel returns (e_tot, e_cas, fcivec, mo, mo_energy)
pyscf_rows = []
for n in range(1, NROOTS):
    tdm = casci.fcisolver.trans_rdm1(ci[0], ci[n], NCAS, NELECAS)   # active-MO transition density
    mu = np.einsum("pq,xpq->x", tdm, dip_mo)
    mu2 = float(np.vdot(mu, mu).real)
    dE = (e_casci[n] - e_casci[0]) * HARTREE2EV
    f = 2.0 / 3.0 * (e_casci[n] - e_casci[0]) * mu2
    pyscf_rows.append(dict(state=n, dE_eV=dE, mu_norm=float(np.sqrt(mu2)), f=float(f)))

# ============ cross-check (2): QIS_QE run_qeom energies vs exact diag ============
qeom_dE = None
if os.environ.get("QIS_RUN_QEOM") == "1":   # opt-in: VQE+qEOM is CPU-heavy -> run via SLURM
    try:
        qeom = solvers.run_qeom(ham, num_particles, excitations="sd")
        ev = np.sort(np.asarray(qeom.eigenvalues).real)      # total energies (ground + excited)
        qeom_dE = [float((ev[k] - ev[0]) * HARTREE2EV) for k in range(1, len(ev))]
    except Exception as e:
        import traceback
        print("[qeom] FAILED:", type(e).__name__, e)
        traceback.print_exc()

# ============ cross-check (3): PySCF TDDFT (PBE) baseline ============
mfd = dft.RKS(mol); mfd.xc = "pbe"; mfd.run()
td = tdscf.TDDFT(mfd); td.nstates = NROOTS; td.kernel()
osc = td.oscillator_strength()
tddft_rows = [dict(state=i + 1, dE_eV=float(td.e[i] * HARTREE2EV), f=float(osc[i]))
              for i in range(len(td.e))]

# ---------------------------------- report ----------------------------------
def fmt(rows, cols):
    return "\n".join("  " + "  ".join(f"{r[c]:>10.4f}" if isinstance(r[c], float) else f"{r[c]:>10}"
                                      for c in cols) for r in rows)

print(f"LiCN gas-phase  CAS({NELECAS},{NCAS})  cc-pVDZ  SA-CASSCF/{NROOTS} states")
print(f"RHF E = {mf.e_tot:.6f}   SA-CASSCF E0 = {mc.e_states[0]:.6f}\n")
print("QIS_QE exact-diag (state  dE_eV  S2  mu_norm[a.u.]  f):")
print(fmt(qis_rows, ["state", "dE_eV", "S2", "mu_norm", "f"]))
print("\nPySCF CASCI trans_rdm1 cross-check (state  dE_eV  mu_norm  f):")
print(fmt(pyscf_rows, ["state", "dE_eV", "mu_norm", "f"]))
if qeom_dE is not None:
    print("\nQIS_QE run_qeom transition energies (eV):", [f"{x:.4f}" for x in qeom_dE])
print("\nPySCF TDDFT/PBE baseline (state  dE_eV  f):")
print(fmt(tddft_rows, ["state", "dE_eV", "f"]))

# agreement summary: QIS_QE vs PySCF CASCI (should be machine precision)
dE_diff = max(abs(q["dE_eV"] - p["dE_eV"]) for q, p in zip(qis_rows, pyscf_rows))
mu_diff = max(abs(q["mu_norm"] - p["mu_norm"]) for q, p in zip(qis_rows, pyscf_rows))
print(f"\n[validation] max |dE| diff QIS_QE vs PySCF CASCI = {dE_diff:.2e} eV")
print(f"[validation] max |mu_norm| diff                   = {mu_diff:.2e} a.u.")

out = dict(system="LiCN_gas", cas=[NELECAS, NCAS], basis="cc-pVDZ", nroots=NROOTS,
           rhf=float(mf.e_tot), qis=qis_rows, pyscf_casci=pyscf_rows,
           qeom_dE_eV=qeom_dE, tddft_pbe=tddft_rows,
           validation=dict(max_dE_diff_eV=dE_diff, max_mu_diff_au=mu_diff))
tag = "" if (NELECAS, NCAS) == (6, 6) else f"_cas{NELECAS}_{NCAS}"
(OUT / f"phase0_licn_gas{tag}.json").write_text(json.dumps(out, indent=2))
np.savez(OUT / f"phase0_licn_gas{tag}.npz",
         qis_energies=res.energies, qis_s2=res.spin_squared,
         transition_dms=res.transition_dms, dip_mo=dip_mo,
         casci_energies=np.asarray(e_casci))
print(f"\nsaved -> {OUT/('phase0_licn_gas'+tag+'.json')} and .npz")
