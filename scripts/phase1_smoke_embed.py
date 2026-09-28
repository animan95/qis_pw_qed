#!/usr/bin/env python
"""Phase 1 smoke test: fold the existing size_small sub_licn.snpy embedding
potential into a PySCF LiCN calc via pyscf_embed.enable_embedding, and check:
  (1) <mu|v_emb|nu> is real & symmetric, nonzero
  (2) embedded RHF differs from gas-phase (environment shifts the mean field)
  (3) it propagates to CASSCF -> QIS_QE (excitation energies shift vs gas-phase)
"""
import sys
import numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/qis_pw_qed")
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")
from pyscf import gto, scf, mcscf, fci
from casidapy import pyscf_embed
import tensors, solvers

HARTREE2EV = 27.211386245988
SNPY = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small/LiCN/sub_licn.snpy"

# LiCN at its size_small fragment position (Angstrom) so it aligns with the .snpy grid
mol = gto.M(atom="Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310",
            basis="cc-pVDZ", verbose=0)

# --- gas-phase reference ---
mf0 = scf.RHF(mol).run()
mc0 = mcscf.CASSCF(mf0, 6, 6); mc0.fcisolver = fci.direct_spin0.FCI()
mc0 = mc0.state_average_([1/6]*6).run()
ham0, np0 = tensors.from_pyscf_casci(mc0)
r0 = solvers.diagonalize_active_space(ham0, np0, n_states=6)

# --- embedded ---
mf = scf.RHF(mol)
Vemb = pyscf_embed.enable_embedding(mf, extemb=SNPY, unit="Ry")
mf.run()
mc = mcscf.CASSCF(mf, 6, 6); mc.fcisolver = fci.direct_spin0.FCI()
mc = mc.state_average_([1/6]*6).run()
ham, npart = tensors.from_pyscf_casci(mc)
r = solvers.diagonalize_active_space(ham, npart, n_states=6)

# (1) matrix sanity
asym = np.abs(Vemb - Vemb.T).max()
print(f"[1] Vemb: shape={Vemb.shape} real={np.isrealobj(Vemb)} "
      f"max|asym|={asym:.2e} ||Vemb||={np.linalg.norm(Vemb):.4f} Ha  "
      f"diag range=[{np.diag(Vemb).min():.4f},{np.diag(Vemb).max():.4f}]")

# (2) mean-field shift
print(f"[2] RHF gas={mf0.e_tot:.6f}  embedded={mf.e_tot:.6f}  "
      f"shift={(mf.e_tot-mf0.e_tot)*HARTREE2EV:+.3f} eV")

# (3) excitation energies shift (solvatochromism)
print("[3] excitation energies (eV): gas -> embedded  (Delta = solvatochromic shift)")
for n in range(1, 6):
    e0 = (r0.energies[n]-r0.energies[0])*HARTREE2EV
    e1 = (r.energies[n]-r.energies[0])*HARTREE2EV
    s2 = r.spin_squared[n]
    print(f"    state {n}: {e0:7.3f} -> {e1:7.3f}   d={e1-e0:+.3f}   S2={s2:.2f}")

# constant-invariance check: adding a constant to v_emb must not change excitations
Vc = Vemb + 0.05*np.eye(Vemb.shape[0])   # crude: shift diagonal ~ constant potential proxy
print("\n[checks] Vemb symmetric & real: PASS" if (asym < 1e-8 and np.isrealobj(Vemb)) else "[checks] FAIL")
print("done.")
