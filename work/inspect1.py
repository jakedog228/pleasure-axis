import pandas as pd, glob, os
pd.set_option('display.width', 250)
def model_of(f): return os.path.normpath(f).split(os.sep)[1]
rows = []
for f in glob.glob('results/*/analysis/zscores_by_category.csv'):
    m = model_of(f)
    if m.startswith('TEST'): continue
    z = pd.read_csv(f); z['model'] = m; rows.append(z)
Z = pd.concat(rows)
for fam, ds in (('pleasure', 'pl/S2_1P'), ('pleasure', 'pl/S1_1P'), ('pain', 'pa/S2_1P'), ('pleasure', 'pa/S2_1P'), ('pain', 'pl/S2_1P')):
    t = Z[(Z.family == fam) & (Z.dataset == ds)].pivot_table(index='model', columns='category', values='mean_z')
    print(f'--- z on {fam} axis, dataset {ds}'); print(t.round(2).to_string())
G = []
for f in glob.glob('results/*/analysis/geometry2_auc.csv'):
    m = model_of(f)
    if m.startswith('TEST'): continue
    g = pd.read_csv(f); g['model'] = m; G.append(g)
G = pd.concat(G); G = G[G.layer_tag == 'plL']
print(G[G.contrast != 'cosines'].groupby('contrast')[['pleasure', 'pain', 'intensity', 'sign', 'valence', 'pleasure_n', 'pain_n']].mean().round(3).to_string())
print(G[G.contrast == 'cosines'][['model'] + [c for c in G.columns if c.startswith('cos_')]].round(3).to_string(index=False))
