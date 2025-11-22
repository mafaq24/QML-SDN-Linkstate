# Minimal hybrid kernel utilities (classical RBF + quantum ZZFeatureMap)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

# Qiskit versions aligned with your project notes
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel

def gamma_scale(X):
    # Same spirit as your script
    return 1.0 / (X.shape[1] * X.var())

def scale_train_val_test(Xtr, Xva, Xte):
    sc = MinMaxScaler()
    return sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)

def classical_kernels(Xtr_s, Xva_s, Xte_s):
    g = gamma_scale(Xtr_s)
    Kc_tr   = rbf_kernel(Xtr_s, Xtr_s, gamma=g)
    Kc_va   = rbf_kernel(Xva_s, Xtr_s, gamma=g)
    Kc_te_t = rbf_kernel(Xte_s, Xtr_s, gamma=g)
    return Kc_tr, Kc_va, Kc_te_t

def quantum_kernels(Xtr_s, Xva_s, Xte_s, reps=1):
    backend = Aer.get_backend("statevector_simulator")
    fmap = ZZFeatureMap(feature_dimension=Xtr_s.shape[1], reps=reps, entanglement="full")
    qi = QuantumInstance(backend=backend)
    qk = QuantumKernel(feature_map=fmap, quantum_instance=qi)
    Kq_tr   = qk.evaluate(x_vec=Xtr_s)
    Kq_va   = qk.evaluate(x_vec=Xva_s, y_vec=Xtr_s)
    Kq_te_t = qk.evaluate(x_vec=Xte_s, y_vec=Xtr_s)
    return Kq_tr, Kq_va, Kq_te_t

def hybrid(Kc, Kq, alpha):
    Kc_tr, Kc_va, Kc_te_t = Kc
    Kq_tr, Kq_va, Kq_te_t = Kq
    Kh_tr   = (1 - alpha) * Kc_tr   + alpha * Kq_tr
    Kh_va   = (1 - alpha) * Kc_va   + alpha * Kq_va
    Kh_te_t = (1 - alpha) * Kc_te_t + alpha * Kq_te_t
    return Kh_tr, Kh_va, Kh_te_t
