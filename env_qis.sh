# Isolated env for QIS_QE + PySCF + Qiskit (Phase 0/1 of qis_pw_qed).
# Loads the community python module ONLY for system libs (libffi.so.7 etc.),
# then activates the isolated qisenv venv and points at QIS_QE. PYTHONPATH is
# set (not appended) so the module's site-packages cannot shadow qisenv.
module use /projects/community-old/modulefiles
module load python/3.10-mp1009
export LD_LIBRARY_PATH="/projects/community-old/libffi/3.3/gc563/lib64:/projectsn/mp1009_1/am4655/subsystem_vs_lr_tddft/libfix:/projects/community-old/python/3.10/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source /projectsn/mp1009_1/am4655/qis_pw_qed/qisenv/bin/activate
export PYTHONPATH="/projectsn/mp1009_1/am4655/QIS_QE"
