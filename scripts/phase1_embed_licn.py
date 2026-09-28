#!/usr/bin/env python
"""Phase 1 (single LiCN): gas vs Ar-embedded excitations by THREE solvers, with
far-field-referenced v_emb:
  - CASSCF(6,6)->QIS_QE  (energies + transition dipoles)
  - EOM-EE-CCSD          (energies)   [user asked about CCSD]
  - TDDFT/PBE            (energies + f)  baseline
Also: (i) far-field referencing makes embedded RHF total ~ gas phase;
      (ii) excitations are invariant to the v_emb constant (ref vs no-ref).
"""
import sys
import numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/qis_pw_qed")
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")
from pyscf import gto, scf, mcscf, fci, cc, dft, tdscf
from casidapy import pyscf_embed
import tensors, solvers

EV = 27.211386245988
SNPY = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small/LiCN/sub_licn.snpy"
mol = gto.M(atom="Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310",
            basis="cc-pVDZ", verbose=0)


def casscf_qis(mf, n=5):
    mc = mcscf.CASSCF(mf, 6, 6); mc.fcisolver = fci.direct_spin0.FCI()
    mc = mc.state_average_([1/6]*6).run()
    ncore = mc.ncore; mo = mc.mo_coeff[:, ncore:ncore+6]
    dip = -np.einsum("xuv,up,vq->xpq", mol.intor("int1e_r", comp=3), mo, mo)
    ham, npart = tensors.from_pyscf_casci(mc)
    r = solvers.diagonalize_active_space(ham, npart, n_states=8)
    out = []
    for k in range(1, len(r.energies)):
        dE = (r.energies[k]-r.energies[0])*EV
        mu = solvers.transition_dipole(r.transition_dms[k-1], dip)
        out.append((dE, float(r.spin_squared[k]), float(np.linalg.norm(mu))))
    return out


def eomccsd(mf, nroots=5):
    mycc = cc.CCSD(mf); mycc.verbose = 0; mycc.kernel()
    eom = mycc.EOMEESinglet()
    e = np.atleast_1d(eom.kernel(nroots=nroots)[0])
    return [float(x*EV) for x in e], mycc.e_corr


def tddft_pbe(mf_ks, nroots=5):
    td = tdscf.TDDFT(mf_ks); td.nstates = nroots; td.verbose = 0; td.kernel()
    return list(zip((td.e*EV).tolist(), td.oscillator_strength().tolist()))


def run(embed, ref="farfield"):
    mf = scf.RHF(mol)
    Vemb = None
    if embed:
        Vemb = pyscf_embed.enable_embedding(mf, extemb=SNPY, unit="Ry", ref=ref)
    mf.run()
    cas = casscf_qis(mf)
    eom_e, ecorr = eomccsd(mf)
    mfk = dft.RKS(mol); mfk.xc = "pbe"
    if embed:
        pyscf_embed.enable_embedding(mfk, extemb=SNPY, unit="Ry", ref=ref)
    mfk.run()
    td = tddft_pbe(mfk)
    return dict(erhf=mf.e_tot, cas=cas, eom=eom_e, td=td, Vemb=Vemb)


gas = run(embed=False)
emb = run(embed=True, ref="farfield")
emb_raw = run(embed=True, ref=None)   # for constant-invariance check

print("=== RHF total energy (far-field referencing) ===")
print(f"  gas={gas['erhf']:.6f}  embedded(farfield)={emb['erhf']:.6f}  "
      f"raw(no-ref)={emb_raw['erhf']:.6f} Ha")
print(f"  farfield shift={ (emb['erhf']-gas['erhf'])*EV:+.2f} eV   "
      f"raw shift={(emb_raw['erhf']-gas['erhf'])*EV:+.1f} eV (gauge constant)")

print("\n=== CASSCF(6,6)->QIS_QE : gas -> embedded (eV), transition dipole (a.u.) ===")
for k,(g,e) in enumerate(zip(gas['cas'], emb['cas']), 1):
    print(f"  state {k}: {g[0]:6.3f} -> {e[0]:6.3f}  d={e[0]-g[0]:+.3f}  "
          f"S2={e[1]:.2f}  |mu|_emb={e[2]:.3f}")

print("\n=== EOM-EE-CCSD singlet : gas -> embedded (eV) ===")
for k,(g,e) in enumerate(zip(gas['eom'], emb['eom']), 1):
    print(f"  root {k}: {g:6.3f} -> {e:6.3f}  d={e-g:+.3f}")

print("\n=== TDDFT/PBE : gas -> embedded (eV, f) ===")
for k,(g,e) in enumerate(zip(gas['td'], emb['td']), 1):
    print(f"  state {k}: {g[0]:6.3f}(f={g[1]:.3f}) -> {e[0]:6.3f}(f={e[1]:.3f})  d={e[0]-g[0]:+.3f}")

# constant-invariance: embedded CASSCF excitations must match with/without far-field ref
dmax = max(abs(a[0]-b[0]) for a,b in zip(emb['cas'], emb_raw['cas']))
print(f"\n[invariance] max |dE| (farfield vs no-ref) = {dmax:.2e} eV  "
      f"({'PASS' if dmax<1e-6 else 'FAIL'})")
