"""Validate the casidapy(plain)/QIS_QE(qiskit) split AND run the excitation-
convergence study on the gaussians-only .snpy sweep (e5/e6/e7)."""
import sys, numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")   # for embed_qiskit + tensors/solvers
from casidapy.embed_wft import build_embedded_mf, build_embedded_casscf, casscf_excitations, eomccsd_excitations
import embed_qiskit
EV = 27.211386245988
LiCN = "Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310"
SNPY1 = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small/LiCN/sub_licn.snpy"

print("### (1) plain casidapy CASSCF  vs  qiskit QIS_QE CASSCF  (same embedded mc) ###")
mf = build_embedded_mf(LiCN, extemb=SNPY1)
mc = build_embedded_casscf(mf, 6, 6, nstates=8)
plain, _ = casscf_excitations(mf, mc=mc, nstates=8)
qk = embed_qiskit.casscf_excitations_qiskit(mc, nstates=8)
dmax = max(abs(p[0]-q[0]) for p, q in zip(plain, qk))*EV
print(f"  plain vs qiskit max |dE| = {dmax:.2e} eV  ({'PASS' if dmax<1e-6 else 'FAIL'})")
eom, _ = eomccsd_excitations(mf, nroots=3)
print(f"  embedded EOM-CCSD (eV): {[round(x*EV,3) for x in eom]}")

print("\n### (2) excitation-convergence vs SCF tightness (gaussians-only sweep) ###")
base = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small"
FRAG_A = "Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310"
prev = None
for tag in ("e5", "e6", "e7"):
    snpy = f"{base}/dimer_g_{tag}/sub_licn_a.snpy"
    mfA = build_embedded_mf(FRAG_A, extemb=snpy)
    st, _ = casscf_excitations(mfA, 6, 6, nstates=8)
    sing = [(w*EV, np.linalg.norm(mu)) for w, mu, s2 in st if s2 < 0.1]
    bright = max(sing, key=lambda x: x[1]) if sing else (float('nan'), 0)
    tail = ""
    if prev is not None:
        tail = f"   Δ(bright vs prev) = {abs(bright[0]-prev)*1000:6.2f} meV"
    print(f"  {tag}: lowest singlet={sing[0][0]:.4f} eV  brightest={bright[0]:.4f} eV (|mu|={bright[1]:.3f}){tail}")
    prev = bright[0]
