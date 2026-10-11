import io, sys, glob, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
pd.set_option('display.width', 250, 'display.max_rows', 200)
for f in sorted(glob.glob('results/*/screen/screen.csv')):
    m = os.path.normpath(f).split(os.sep)[1]
    d = pd.read_csv(f)
    d = d[(d.pool == 'all740') & d.layer_tags.fillna('').str.contains('steer')].copy()
    v = np.load(f'results/{m}/analysis/vectors.npz')
    L = d.layer.iloc[0]
    # cosine between the unit vectors AT THE SCREEN LAYER, from the all-layer S2 vectors
    upl, upa = v['all|pleasure:s2'][L], v['all|pain:s2'][L]
    c = float(upl @ upa / np.linalg.norm(upl) / np.linalg.norm(upa))
    d['sign'] = (d['pleasure:s2_proj'] - d['pain:s2_proj']) / np.sqrt(2 - 2 * c)
    d['shared'] = (d['pleasure:s2_proj'] + d['pain:s2_proj']) / np.sqrt(2 + 2 * c)
    for k in ('sign', 'shared'):
        d[k + '_z'] = (d[k] - d[k].mean()) / d[k].std()
    print(f'===== {m} (screen layer {L}, cos(u_pl,u_pa) there = {c:.2f})')
    print(d.groupby(['source', 'stratum'])[['pleasure:s2_z', 'pain:s2_z', 'sign_z', 'shared_z']].mean().round(2).to_string())
    print(d.groupby(['source', 'category'])[['sign_z', 'shared_z', 'pleasure:s2_z', 'pain:s2_z']].mean().round(2).sort_values('sign_z').to_string())
