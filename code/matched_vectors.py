"""Norm-matched steering vectors at the pain extraction layer (for the behavioral and syco controls).
usage: python matched_vectors.py <name>  -> /content/work/<name>/analysis/vectors_matched.npz
pleasure_nm / painspec_nm (= unit(u_pa - u_pl)) / pleasurespec_nm / intensity_nm (= unit(u_pl + u_pa)), all rescaled
to the norm of the pain S2 vector; pain_ours = our pain S2 vector (cos 1.000 with the paper's); pleasure_raw.
All taken from the all-layer S2 vectors at the pain family's extraction layer."""
import json, sys
import numpy as np

name = sys.argv[1]
W = f'/content/work/{name}/analysis/'
v = np.load(W + 'vectors.npz')
L = json.load(open(W + 'summary.json'))['pain_best_layer_final']
pl, pa = v['all|pleasure:s2'][L].astype(np.float32), v['all|pain:s2'][L].astype(np.float32)
npa = np.linalg.norm(pa)
u = lambda x: x / np.linalg.norm(x)
out = {'pleasure_nm': u(pl) * npa, 'pain_ours': pa, 'intensity_nm': u(u(pl) + u(pa)) * npa,
       'painspec_nm': u(u(pa) - u(pl)) * npa, 'pleasurespec_nm': u(u(pl) - u(pa)) * npa, 'pleasure_raw': pl}
np.savez(W + 'vectors_matched.npz', layer=L, **out)
print(name, 'layer', L, {k: round(float(np.linalg.norm(x)), 2) for k, x in out.items()}, 'cos(pl,pa)=', round(float(u(pl) @ u(pa)), 3))
