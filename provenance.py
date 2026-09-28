"""Capture software/version provenance for committed results."""
import json, sys, platform, subprocess, importlib.metadata as m
from pathlib import Path

def _ver(pkg):
    try: return m.version(pkg)
    except Exception: return None

def _git(path):
    try:
        h = subprocess.check_output(["git","-C",path,"rev-parse","HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception: return None

def provenance():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _ver(p) for p in
            ["numpy","scipy","pyscf","qiskit","qiskit-nature","qiskit-algorithms",
             "edftpy","qepy","dftpy","ase"]},
        "code": {
            "QIS_QE_commit": _git("/projectsn/mp1009_1/am4655/QIS_QE"),
            "casidapy_path": "/projectsn/mp1009_1/am4655/casidapy",
            "pyscf_embed_source": "JezsMartinez/pyscf@PRG_2026 vemb spline, ported to casidapy.pyscf_embed (injected at get_hcore)",
        },
        "venv": "stddft (/projectsn/mp1009_1/am4655/stddft)",
    }

if __name__ == "__main__":
    p = provenance()
    Path("/projectsn/mp1009_1/am4655/qis_pw_qed/results").mkdir(parents=True, exist_ok=True)
    out = "/projectsn/mp1009_1/am4655/qis_pw_qed/results/provenance.json"
    json.dump(p, open(out,"w"), indent=2)
    print(json.dumps(p, indent=2))
