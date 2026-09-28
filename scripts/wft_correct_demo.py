#!/usr/bin/env python
"""Demonstrate the WFT correction of subsystem TDDFT on the small embedded dimer.

Per fragment: embedded PySCF TDDFT = the 'subsystem-TDDFT' fragment states;
embedded CASSCF = the correction. Then casidapy.wft_corrected_coupled_casida
scissor-corrects the fragment on-site energies to CASSCF (matched by
energy+dipole), swaps in WFT transition dipoles, and re-diagonalizes.
"""
import sys
import numpy as np
sys.path.insert(0, "/projectsn/mp1009_1/am4655/QIS_QE")
from pyscf import tdscf
from casidapy.embed_wft import build_embedded_mf, casscf_excitations
from casidapy.subsystem_coupling import (
    match_fragment_states, apply_wft_correction, wft_corrected_coupled_casida,
)
EV = 27.211386245988
D = "/projectsn/mp1009_1/am4655/pw_qed/subs_qed/size_small/dimer_u"
FRAG = {
    "A": ("Li 2.770 5.310 5.310; C 4.723 5.310 5.310; N 5.897 5.310 5.310", f"{D}/sub_licn_a.snpy"),
    "B": ("Li 2.770 5.310 0.000; C 4.723 5.310 0.000; N 5.897 5.310 0.000", f"{D}/sub_licn_b.snpy"),
}


def tddft_fragment(atom, snpy, nstates=6):
    """Embedded PySCF TDDFT (PBE) -> (omega_Ha, dip_tran(n,3)) = subsystem-TDDFT frag."""
    mf = build_embedded_mf(atom, extemb=snpy, xc="pbe")
    td = tdscf.TDDFT(mf); td.nstates = nstates; td.verbose = 0; td.kernel()
    # transition dipoles (length gauge) from PySCF
    mu = td.transition_dipole()          # (nstates, 3), a.u.
    return np.asarray(td.e, float), np.asarray(mu, float), mf


def casscf_fragment(atom, snpy, nstates=8):
    mf = build_embedded_mf(atom, extemb=snpy)
    st, _ = casscf_excitations(mf, 6, 6, nstates=nstates)
    return st  # list of (omega_Ha, mu_vec, s2)


# character-driven matching: dipole alignment dominates (correction is a LARGE
# energy shift, so energy can't drive the match), gated by max_cost; and WFT is
# filtered to SINGLETS (RKS/TDDFT gives singlet excitations).
MK = dict(energy_weight_ev=0.15, dipole_weight=1.0, max_cost=0.4)

frag_results, wft_fragments = [], []
for lab in ("A", "B"):
    atom, snpy = FRAG[lab]
    om_t, mu_t, _ = tddft_fragment(atom, snpy)
    wft_all = casscf_fragment(atom, snpy)
    wft = [(w, mu, s2) for w, mu, s2 in wft_all if s2 < 0.5]   # singlets only
    frag_results.append({"omega": om_t, "dip_tran": mu_t,
                         "Z": np.eye(len(om_t)), "eigenvectors": np.eye(len(om_t))})
    wft_fragments.append(wft)
    print(f"--- fragment {lab} ---")
    print(f"  TDDFT omega (eV): {np.round(om_t*EV,3)}  |mu|={np.round(np.linalg.norm(mu_t,axis=1),3)}")
    print(f"  CASSCF singlets (eV): {np.round([w*EV for w,_,_ in wft],3)}  "
          f"|mu|={np.round([np.linalg.norm(mu) for _,mu,_ in wft],3)}")

print("\n=== state matching (TDDFT -> CASSCF singlet, dipole-character driven, gated) ===")
for lab, res, wft in zip("AB", frag_results, wft_fragments):
    m = match_fragment_states(res["omega"], res["dip_tran"],
                              [w for w,_,_ in wft], [mu for _,mu,_ in wft],
                              **MK)
    for i, j in sorted(m.items()):
        print(f"  {lab}: TDDFT#{i} {res['omega'][i]*EV:.3f} eV  ->  CASSCF#{j} {wft[j][0]*EV:.3f} eV")

# energy-corrected fragment results (uncoupled -> isolates the correction)
frag_ec, frag_wd, matches = apply_wft_correction(frag_results, wft_fragments,
                                                 match_kwargs=dict(**MK))
print("\n=== per-fragment on-site energies: TDDFT -> WFT-corrected (eV) ===")
for lab, r0, rc in zip("AB", frag_results, frag_ec):
    print(f"  {lab}: {np.round(r0['omega']*EV,3)}  ->  {np.round(rc['omega']*EV,3)}")

# coupled, uncoupled K (isolate correction): coupled spectrum should be the WFT-corrected diagonal
res_unc = wft_corrected_coupled_casida(frag_results, {}, wft_fragments, tda=True,
                                       match_kwargs=dict(**MK))
print("\n=== WFT-corrected coupled spectrum (K=0, so = corrected fragment states) ===")
order = np.argsort(res_unc["omega"])
for n in order[:8]:
    mu = res_unc["d_mode"][n]
    print(f"  state: {res_unc['omega'][n]*EV:7.3f} eV   |mu|={np.linalg.norm(mu):.3f}")
print("\nOK: correction layer replaces TDDFT fragment excitations with matched CASSCF, "
      "carries WFT dipoles, and flows through assemble_coupled_casida.")
