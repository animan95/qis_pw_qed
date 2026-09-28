#!/usr/bin/env python
"""Phase 1 capstone: embedded LiCN dimer -> QIS_QE per fragment -> Forster
(Frenkel-exciton) coupling -> cavity-QED post-processing.

Each LiCN fragment is embedded in the SAME Ar matrix it sees in eDFTpy (its own
converged sub_licn_{a,b}.snpy folded into hcore), solved by CASSCF(6,6)->QIS_QE
for singlet excitation energies + transition-dipole VECTORS. The two chromophores
are coupled by dipole-dipole (Forster) interaction into Frenkel excitons
(Davydov pair), whose bright component is then dressed by a cavity via
casidapy.run_qed_post (PF / JC / QED-TDDFT).
"""
import sys, json
import numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/qis_pw_qed")
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")
from pyscf import gto, scf, mcscf, fci
from casidapy import pyscf_embed
import tensors, solvers
from casidapy.qed_post import run_qed_post

EV = 27.211386245988
ANG2BOHR = 1.8897259886
D = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small/dimer_u"
OUT = "/projectsn/mp1009_1/am4655/qis_pw_qed/results"

# fragment geometries (Angstrom) and their embedding potentials
FRAG = {
    "A": ("Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310", f"{D}/sub_licn_a.snpy"),
    "B": ("Li 2.770 5.310 0.000; C 4.723 5.310 0.000; N 5.897 5.310 0.000", f"{D}/sub_licn_b.snpy"),
}
R_AB = np.array([0.0, 0.0, 5.310]) * ANG2BOHR      # A->B vector (Bohr), min-image along z


def embed_fragment(atom, snpy, nsing=4, n_states=12):
    """Embedded CASSCF(6,6)->QIS_QE; return singlet (omega_Ha, mu_vec[a.u.]) list."""
    mol = gto.M(atom=atom, basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    pyscf_embed.enable_embedding(mf, extemb=snpy, unit="Ry", ref="farfield")
    mf.run()
    mc = mcscf.CASSCF(mf, 6, 6); mc.fcisolver = fci.direct_spin0.FCI()
    mc = mc.state_average_([1/6]*6).run()
    mo = mc.mo_coeff[:, mc.ncore:mc.ncore+6]
    dip = -np.einsum("xuv,up,vq->xpq", mol.intor("int1e_r", comp=3), mo, mo)
    ham, npart = tensors.from_pyscf_casci(mc)
    r = solvers.diagonalize_active_space(ham, npart, n_states=n_states)
    states = []
    for k in range(1, len(r.energies)):
        if r.spin_squared[k] < 0.1:                       # singlet
            w = float(r.energies[k] - r.energies[0])
            mu = solvers.transition_dipole(r.transition_dms[k-1], dip).real
            states.append((w, np.asarray(mu)))
    states.sort(key=lambda s: s[0])
    return states[:nsing]


def dip_dip(mu1, mu2, R):
    """Forster dipole-dipole coupling (a.u.)."""
    r = np.linalg.norm(R); n = R / r
    return (np.dot(mu1, mu2) - 3*np.dot(mu1, n)*np.dot(mu2, n)) / r**3


print("Embedding fragment A ..."); A = embed_fragment(*FRAG["A"])
print("Embedding fragment B ..."); B = embed_fragment(*FRAG["B"])
for lab, S in (("A", A), ("B", B)):
    for i, (w, mu) in enumerate(S):
        print(f"  {lab}{i}: dE={w*EV:.3f} eV  mu={np.round(mu,3)}  |mu|={np.linalg.norm(mu):.3f}")

# ---- Frenkel exciton Hamiltonian in basis {A_i, B_j} (inter-fragment Forster) ----
nA, nB = len(A), len(B)
E = [w for w, _ in A] + [w for w, _ in B]
MU = [mu for _, mu in A] + [mu for _, mu in B]
H = np.diag(E).astype(float)
for i in range(nA):
    for j in range(nB):
        J = dip_dip(A[i][1], B[j][1], R_AB)
        H[i, nA+j] = H[nA+j, i] = J
w_coupled, C = np.linalg.eigh(H)
# coupled transition dipoles: sum of fragment dipoles weighted by eigenvector
mu_coupled = np.array([sum(C[k, n]*MU[k] for k in range(nA+nB)) for n in range(nA+nB)])
osc = np.array([2/3 * w_coupled[n] * mu_coupled[n]@mu_coupled[n] for n in range(len(w_coupled))])

print("\n=== Coupled Frenkel excitons (Davydov) ===")
for n in range(len(w_coupled)):
    print(f"  exciton {n}: dE={w_coupled[n]*EV:.4f} eV  |mu|={np.linalg.norm(mu_coupled[n]):.3f}  f={osc[n]:.4f}")
bright = int(np.argmax(osc))
# lowest bright Davydov pair splitting (first two nearly-degenerate manifold)
print(f"  brightest exciton {bright} at {w_coupled[bright]*EV:.4f} eV")
# Davydov splitting for the lowest pair:
dav = abs(w_coupled[1]-w_coupled[0])*EV*1000
print(f"  lowest-pair Davydov splitting = {dav:.1f} meV")

# ---- cavity-QED on the coupled dimer states ----
pol = mu_coupled[bright] / np.linalg.norm(mu_coupled[bright])
qed = run_qed_post(np.asarray(w_coupled), np.asarray(mu_coupled),
                   f=osc, models=("pf", "jc", "tddft"),
                   omega_c=float(w_coupled[bright]),          # resonant with bright exciton
                   lam_sweep=[0.01, 0.03, 0.05], pol=pol.tolist())
print(f"\n=== cavity-QED on coupled dimer (omega_c = {w_coupled[bright]*EV:.3f} eV, "
      f"resonant w/ bright exciton) ===")
rabi = {}
for model in ("pf", "jc", "tddft"):
    blocks = qed["results"][model]
    b = blocks[-1]                                   # largest lambda in the sweep
    om = np.atleast_1d(b["omega"]).astype(float) * EV
    pf_ = np.atleast_1d(b.get("photon_frac", np.zeros_like(om))).astype(float)
    ff = np.atleast_1d(b.get("f", np.zeros_like(om))).astype(float)
    print(f"  [{model}]  lambda={b.get('lam_mag', 0.05)}:")
    pols = []
    for e, p, fo in zip(om, pf_, ff):
        if e <= 1e-6:
            continue
        tag = "  <- polariton" if p > 0.05 else ("  (dark)" if fo < 1e-4 else "")
        print(f"     {e:8.4f} eV   photon={p:5.3f}   f={fo:.4f}{tag}")
        # polariton = light-matter mixed (0.05<photon<0.95) near the cavity,
        # excluding the bare-photon replicas (photon~1) of the DSE ladder
        if 0.05 < p < 0.95 and abs(e - w_coupled[bright]*EV) < 1.0:
            pols.append(e)
    if len(pols) >= 2:
        rabi[model] = max(pols) - min(pols)
        print(f"     -> lower/upper polariton splitting ~ {rabi[model]*1000:.0f} meV")

json.dump({"A":[[w*EV, m.tolist()] for w,m in A], "B":[[w*EV, m.tolist()] for w,m in B],
           "coupled_eV":(w_coupled*EV).tolist(), "osc":osc.tolist(),
           "davydov_meV":dav, "bright":bright},
          open(f"{OUT}/phase1_dimer_embed_qed.json","w"), indent=2)
print(f"\nsaved -> {OUT}/phase1_dimer_embed_qed.json")
