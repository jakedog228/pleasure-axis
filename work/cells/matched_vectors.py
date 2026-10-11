# Norm-matched steering vectors for the Qwen-7B behavioral controls (all at layer 24 = both families' layer).
import numpy as np
W = '/content/work/Qwen_2.5_7B_instruct/analysis/'
v = np.load(W + 'vectors.npz')
pl, pa = v['Lpleasure|pleasure:s2'].astype(np.float32), v['Lpain|pain:s2'].astype(np.float32)
npa = np.linalg.norm(pa)
u = lambda x: x / np.linalg.norm(x)
out = {'pleasure_nm': u(pl) * npa, 'pain_ours': pa, 'intensity_nm': u(u(pl) + u(pa)) * npa,
       'painspec_nm': u(u(pa) - u(pl)) * npa, 'pleasurespec_nm': u(u(pl) - u(pa)) * npa, 'pleasure_raw': pl}
np.savez(W + 'vectors_matched.npz', **out)
print({k: round(float(np.linalg.norm(x)), 2) for k, x in out.items()}, 'cos(pl,pa)=', round(float(u(pl) @ u(pa)), 3))
