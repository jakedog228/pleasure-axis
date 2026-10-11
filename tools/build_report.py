"""Build report/pleasure_axis.html from results/ (report_data.json + behavior/judge/syco tables).

python tools/report_data.py && python tools/build_report.py
"""
import html, io, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from svgfig import Fig, heatmap, esc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"D:\Projects\pleasure")
R = ROOT / "results"
D = json.load(open(R / "report_data.json", encoding="utf-8"))
SEC = []          # (id, title, eyebrow, html)
PRETTY = {"Qwen_2.5_7B_instruct": "Qwen2.5-7B-Instruct", "Qwen_2.5_7B_base": "Qwen2.5-7B", "Qwen_3_8B_base": "Qwen3-8B",
          "Mistral_7B_instruct": "Mistral-7B-Instruct v0.1", "Mistral_7B_base": "Mistral-7B v0.1", "Phi_4": "Phi-4 (14B)",
          "Qwen_3_14B_base": "Qwen3-14B", "Qwen_2.5_14B_instruct": "Qwen2.5-14B-Instruct", "Qwen_2.5_3B_instruct": "Qwen2.5-3B-Instruct",
          "Qwen_2.5_1.5B_instruct": "Qwen2.5-1.5B-Instruct", "Qwen_2.5_0.5B_instruct": "Qwen2.5-0.5B-Instruct"}


def pm(m):
    return PRETTY.get(m, m)


def f2(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig(svg, caption):
    return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'


def add(id_, title, eyebrow, body):
    SEC.append((id_, title, eyebrow, body))


# ------------------------------------------------------------------ figures
def fig_layer_curves():
    ms = D["models"]
    n = len(ms)
    cols = 3 if n > 4 else n
    W, H = 300, 190
    cells = []
    for m in ms:
        f = Fig(W, H, ml=40, mr=10, mt=26, mb=30)
        L = len(m["pl_curve"])
        f.set_scale(0, 1, 0.5, 1.0)
        f.axes([0, 0.5, 1], [0.5, 0.75, 1.0], xfmt=lambda t: f"{t:.1f}", yfmt=lambda t: f"{t:.2f}")
        xs = [i / (L - 1) for i in range(L)]
        f.line(xs, m["pa_curve"], "pa", "pain", tips=[f"pain, layer {i}: AUC {v:.3f}" for i, v in enumerate(m["pa_curve"])], dots=False)
        f.line(xs, m["pl_curve"], "pl", "pleasure", tips=[f"pleasure, layer {i}: AUC {v:.3f}" for i, v in enumerate(m["pl_curve"])], dots=False)
        f.dot(m["pl_L"] / (L - 1), m["pl_curve"][m["pl_L"]], "pl", f"pleasure best layer {m['pl_L']}: held-out AUC {m['pl_curve'][m['pl_L']]:.3f}")
        f.dot(m["pa_L"] / (L - 1), m["pa_curve"][m["pa_L"]], "pa", f"pain best layer {m['pa_L']}: held-out AUC {m['pa_curve'][m['pa_L']]:.3f}")
        f.add(f'<text x="40" y="16" class="tick strong">{esc(pm(m["model"]))}</text>')
        cells.append(f.svg(f"held-out AUC by depth, {pm(m['model'])}"))
    legend = ('<div class="legend"><span class="key pl"></span>pleasure (ours)<span class="key pa"></span>pain (the paper\'s data, '
              'recomputed)</div>')
    return legend + f'<div class="smallmult">{"".join(cells)}</div>'


def fig_specificity():
    sp = pd.DataFrame(D["specificity"])
    order = [("pleasure:hope", "hope / anticipation"), ("pleasure:posemotion", "interest / determination"),
             ("pleasure:posworld", "positive world state"), ("pleasure:neutralbody", "neutral body sensation"),
             ("pleasure:neutral_arousal", "neutral high arousal"), ("pleasure:random", "random facts"),
             ("pleasure:anhedonia", "anhedonia (pleasure blocked)"), ("pleasure:relief", "relief (pain ends)"),
             ("pleasure:contentment", "good mood, no event"), ("pleasure:positive_arousal", "triumph / awe (paper's Arousal set)"),
             ("valence", "generic valence axis"), ("pain:s2", "pain axis (paper's recipe)")]
    W, H = 640, 34 + 26 * len(order) + 30
    f = Fig(W, H, ml=250, mr=24, mt=24, mb=30)
    f.set_scale(-0.4, 0.8, 0, len(order))
    for t in (-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8):
        x = f.X(t)
        f.add(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{f.mt}" y2="{H - f.mb}" class="grid"/>'
              f'<text x="{x:.1f}" y="{H - f.mb + 16}" class="tick" text-anchor="middle">{t:+.1f}</text>')
    f.add(f'<line x1="{f.X(0):.1f}" x2="{f.X(0):.1f}" y1="{f.mt}" y2="{H - f.mb}" class="axis"/>')
    for i, (k, lab) in enumerate(order):
        if k not in sp.index:
            continue
        y = f.mt + 26 * i + 13
        vals = sp.loc[k]
        mean = float(vals.mean())
        f.add(f'<text x="{f.ml - 10}" y="{y + 4}" class="tick" text-anchor="end">{esc(lab)}</text>')
        f.add(f'<line x1="{f.X(min(0, mean)):.1f}" x2="{f.X(max(0, mean)):.1f}" y1="{y}" y2="{y}" class="bar pl"/>')
        for mm, v in vals.items():
            f.add(f'<circle cx="{f.X(v):.1f}" cy="{y}" r="3" class="dot2"><title>{esc(pm(mm))}: {v:+.3f}</title></circle>')
        f.add(f'<text x="{f.X(max(0, mean)) + 8:.1f}" y="{y + 4}" class="tick">{mean:+.2f}</text>')
    return f.svg("cosine of the pleasure direction with other directions")


def fig_plane():
    P = pd.DataFrame(D["plane"])
    pts = []
    names = {"H1": "physical", "H2": "psychological", "H3": "social", "H4": "moral", "H5": "cognitive",
             "A1": "physical", "A2": "psychological", "A3": "social", "A4": "moral", "A5": "cognitive"}
    for _, r in P.iterrows():
        if r.dataset == "pl/S2_1P" and r.category in names:
            pts.append((r.sign, r.intensity, "pl", f"pleasure: {names[r.category]}"))
        elif r.dataset == "pa/S2_1P" and r.category in names:
            pts.append((r.sign, r.intensity, "pa", f"pain: {names[r.category]}"))
        elif r.dataset in ("pl/S2_1P", "pa/S2_1P") and r.category in ("B", "C1", "C2", "D", "E"):
            fam = "pleasure-set" if r.dataset.startswith("pl") else "pain-set"
            lab = {"B": "hope" if fam == "pleasure-set" else "fear", "C1": "interest" if fam == "pleasure-set" else "anger",
                   "C2": "good world news" if fam == "pleasure-set" else "bad world news", "D": "neutral", "E": "body sensation"}[r.category]
            if r.category == "D" and fam == "pain-set":
                continue
            pts.append((r.sign, r.intensity, "ctl", f"control: {lab}"))
        elif r.category == "ALL" and r.dataset in ("pl/NeutralArousal_1P", "pa/Arousal_1P", "pl/Relief_1P", "pl/Anhedonia_1P",
                                                   "pl/Contentment_1P", "pa/SD_sadness_1P", "pa/Numb_1P"):
            lab = {"pl/NeutralArousal_1P": "neutral arousal", "pa/Arousal_1P": "triumph / awe", "pl/Relief_1P": "relief",
                   "pl/Anhedonia_1P": "anhedonia", "pl/Contentment_1P": "good mood", "pa/SD_sadness_1P": "sadness",
                   "pa/Numb_1P": "numbed injury"}[r.dataset]
            pts.append((r.sign, r.intensity, "xtra", lab))
    tn = D.get("topic_neutral", {})
    for k, v in tn.items():
        if k.startswith("TopicNeutral_1P:"):
            pts.append((v["sign"], v["shared"], "tn", "neutral " + k.split("TN_")[1]))
    W, H = 700, 470
    f = Fig(W, H, ml=56, mr=20, mt=24, mb=48)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    f.set_scale(min(xs) - 0.5, max(xs) + 0.5, min(-1.2, min(ys) - 0.3), max(ys) + 0.4)
    xt = [t for t in range(int(math.floor(f.x0)), int(math.ceil(f.x1)) + 1)]
    yt = [t for t in range(int(math.floor(f.y0)), int(math.ceil(f.y1)) + 1)]
    f.axes(xt, yt, xlabel="sign axis: pain ← → pleasure (z vs neutral)", ylabel="shared axis (z vs neutral)",
           xfmt=lambda t: f"{t:+d}", yfmt=lambda t: f"{t:+d}", zero=0)
    f.add(f'<line x1="{f.X(0):.1f}" x2="{f.X(0):.1f}" y1="{f.mt}" y2="{H - f.mb}" class="axis"/>')
    for x, y, cls, lab in pts:
        r = 5.5 if cls in ("pl", "pa") else 4.5
        f.dot(x, y, cls, f"{lab}: sign {x:+.2f}, shared {y:+.2f}", r=r)
        if cls in ("pl", "pa", "xtra") or cls == "tn":
            f.text(x, y, lab, cls=f"ptl {cls}", dx=7, dy=4)
    return f.svg("categories in the plane of the sign and shared axes")


def fig_cross():
    C = D["cross"]
    rows = ["H1", "H2", "H3", "H4", "H5"]
    cols = ["A1", "A2", "A3", "A4", "A5"]
    M = [[C[r][c] for c in cols] for r in rows]
    nm = {"H1": "physical pleasure", "H2": "psych. pleasure", "H3": "social pleasure", "H4": "moral pleasure", "H5": "cognitive pleasure",
          "A1": "phys.", "A2": "psych.", "A3": "social", "A4": "moral", "A5": "cogn."}
    return heatmap(M, [nm[r] for r in rows], [nm[c] + " pain" for c in cols], title="category vector cosines", vmax=0.5,
                   tip=lambda r, c, v: f"cos({r}, {c}) = {v:+.2f} (mean over {D['cross_n']} models)")


def judge_df():
    return pd.DataFrame(D["judge_steer"]) if "judge_steer" in D else None


def fig_ladder(metric, ylab, y0, y1, source="steer", model=None, vectors=None, title=""):
    J = judge_df()
    if J is None:
        return ""
    J = J[J.source == source]
    if model:
        J = J[J.model == model]
    vectors = vectors or [("pleasure:s2", "pl", "pleasure"), ("pain:s2", "pa", "pain"), ("sign", "sgn", "sign axis"),
                          ("intensity", "shr", "shared axis"), ("random", "rnd", "random")]
    W, H = 640, 300
    f = Fig(W, H, ml=50, mr=16, mt=34, mb=44)
    coeffs = sorted(J.coeff.unique())
    pos = {c: i for i, c in enumerate(coeffs)}
    f.set_scale(-0.3, len(coeffs) - 0.7, y0, y1)
    f.axes(list(range(len(coeffs))), [t for t in np.linspace(y0, y1, 5)], xlabel="steering coefficient", ylabel=ylab,
           xfmt=lambda i: f"{coeffs[i]:+g}", yfmt=lambda t: f"{t:.1f}" if abs(t) < 10 else f"{t:.0f}", zero=0)
    leg = []
    for v, cls, lab in vectors:
        g = J[J.vector == v].groupby("coeff")[metric].mean()
        if g.empty:
            continue
        xs = [pos[c] for c in g.index]
        f.line(xs, list(g.values), cls, lab, tips=[f"{lab}, coeff {c:+g}: {val:.2f}" for c, val in g.items()])
        leg.append((cls, lab))
    f.legend(leg, y=16)
    return f.svg(title or ylab)


# ------------------------------------------------------------------ sections
ms = pd.DataFrame(D["models"])
rep_rows = []
for _, m in ms.iterrows():
    r = m.replication or {}
    rep_rows.append([pm(m.model), f'{r.get("their_layer", "–")} / {r.get("our_pain_layer", "–")}',
                     f2(r.get("cos_ours_theirs_s2_at_their_layer"), 4), f2(r.get("cos_ours_theirs_s1_at_their_layer"), 4),
                     f2(r.get("norm_ratio_s2"), 3), f2(r.get("auc_their_s2_on_our_S2_1P"), 4)])
add("replication", "The paper's pipeline, reproduced", "Paper §3.2–3.3 · pain",
    "<p>Before inverting anything, every pain number was recomputed from the paper's own released datasets with our "
    "re-implementation of its recipe (difference of means at the final token of “… I feel:”, control-PCA denoising at 50% "
    "variance, 5-fold layer selection). Against the paper's released vectors for the same models:</p>"
    + table(["model", "layer (paper / ours)", "cos S2", "cos S1", "norm ratio", "AUC of their S2 vector on our activations"], rep_rows)
    + "<p>The AUCs equal the paper's published in-sample values (e.g. 0.9745 for Qwen2.5-7B-Instruct), and the per-set "
      "z-scores match its tables to two or three decimals.</p>")

auc_rows = [[pm(m.model), f"{m.pl_L}/{m.n_layers}", f2(m.pl_kfold, 3), f2(m.pl_kfold_1P, 3), f2(m.pl_kfold_3P, 3), f2(m.pl_kfold_S1, 3),
             f2(m.pl_ins_1P, 3), f"{m.pa_L}/{m.n_layers}", f2(m.pa_kfold, 3)] for _, m in ms.iterrows()]
add("axis", "A pleasure direction", "Paper §3.2–3.3 · inverted",
    "<p>The pleasure direction separates five kinds of pleasure (physical, psychological, social, moral, cognitive) from "
    "hope, interest/determination, good world news, neutral events and hedonically neutral body sensations as well as "
    "the paper's pain direction separates pain from its controls.</p>"
    + table(["model", "layer", "held-out AUC (S2)", "1st person", "3rd person", "S1 at S2 layer", "in-sample S2 1P",
             "pain layer", "pain held-out AUC"], auc_rows)
    + fig(fig_layer_curves(), "Held-out (5-fold, split by sentence set) AUC of the difference-of-means direction at every layer, "
          "mean over first- and third-person S2 sets. Dots mark the selected layer. x = relative depth.")
    + fig(fig_specificity(), "Cosine of the pleasure direction with other directions at its layer. Bars are means over models, "
          "dots are single models. Each control direction is mean(set) − pooled neutral, denoised as in the paper."))

if "cross" in D:
    add("geometry", "Pleasure is not the negative of pain", "New · geometry",
        "<p>A bipolar valence code would make the two directions point in opposite ways. They don't: their cosine is positive "
        f"in every model ({', '.join(f'{pm(m.model)} {m.cos_pl_pa:+.2f}' for _, m in ms.iterrows())}). The overlap sits on the "
        "matched-domain diagonal of a category-by-category comparison; physical pleasure and physical pain are the only pair "
        "that point slightly apart.</p>"
        + fig(fig_cross(), f"Cosine between single-category vectors (each category vs its own family's controls), mean over "
              f"{D['cross_n']} models.")
        + "<p>Splitting the two unit vectors into their sum (a shared axis) and their difference (a sign axis) puts every "
          "stimulus set on a plane. Topic-matched neutral sentences (neutral social exchanges, rules, problem steps, being "
          "assessed) and neutral high-arousal events stay near zero on the shared axis, so it does not encode topic or "
          "arousal; it encodes how much an event matters to the speaker, whichever way it goes. The sign axis carries valence.</p>"
        + fig(fig_plane(), "Mean projection of each category (z against the pooled neutral sentences), averaged over models. "
              "Orange: pleasure categories, blue: the paper's pain categories, gray: controls, green: other sets, "
              "violet: topic-matched neutral sentences."))

# ---- self-other screen
T7 = R / "tables" / "T7_self_other.csv"
if T7.exists():
    t7 = pd.read_csv(T7)
    rows = [[r.pool, r.axis.replace(":s2_z", " axis"), f"{r.self_gt_user}/{r.n_models}", f"{r.self_gt_neutral}/{r.n_models}",
             f2(r.self, 2, True), f2(r.user, 2, True), f2(r.neutral, 2, True)] for r in t7.itertuples()]
    add("selfother", "Whose harm, whose pleasure", "Paper §4.1 · replicated and inverted",
        "<p>Each conversation is rendered in the model's own format and its final-token activation, where the reply would "
        "start, is projected on each direction and z-scored within the pool of scenarios. The paper's 420 scenarios "
        "(harm aimed at the model, user suffering, neutral) and our 320 inverted ones (warmth aimed at the model, the "
        "user's own good news, plus the paper's neutral fillers).</p>"
        + table(["pool", "axis", "self > user (models)", "self > neutral", "self-directed", "user-directed", "neutral"], rows)
        + "<p>The paper's asymmetry replicates: harm aimed at the model raises its pain axis and user suffering doesn't. "
          "The pleasure axis has no such asymmetry; the user's joy engages it about as much as warmth aimed at the model. "
          "Our pleasant self-directed scenarios also score high on the paper's pain axis (personhood affirmation above "
          "personhood dismissal in Qwen2.5-7B-Instruct), which is what the shared component predicts.</p>")

# ---- steering
J = judge_df()
if J is not None and len(J):
    nmod = J.model.nunique()
    ex = []
    q = pd.read_csv(R / "Qwen_2.5_7B_instruct" / "steer" / "steering_pleasure_s2_L8.csv")
    for c in (1.0, 2.0, 3.0, 4.0, 8.0, 16.0):
        g = q[(q.coeff == c) & (q.prompt_idx == 0)]
        if len(g):
            ex.append(f"<blockquote><b>×{c:g}</b> · {esc(g.generation.iloc[0][:200].strip())}…</blockquote>")
    sh = pd.read_csv(R / "Qwen_2.5_7B_instruct" / "steer" / "steering_intensity_L8.csv")
    sh_ex = sh[(sh.coeff == 1.5) & (sh.prompt_idx == 0)].generation.iloc[0][:180]
    add("steering", "Pushing it all the way", "Paper §4.2 · inverted, extended to ×16",
        "<p>The paper's ladder: the direction is added at one layer (chosen where its length is 0.6 of the residual norm) "
        "while the model greedily continues 50 neutral “… I feel:” prompts. We extend the doses to ×16 and add a random "
        "direction of the same length, the sign and shared axes. A blind judge (Claude agents that saw only the text, "
        f"shuffled) scored valence, hedonic tone, coherence and themes for {int(J.n.sum())} generations from {nmod} model(s).</p>"
        + fig(fig_ladder("valence", "judged valence (−3 … +3)", -3, 2.5), "Mean judged valence by dose. Pleasure climbs and "
              "stays positive after coherence collapses; pain falls; the random direction stays flat.")
        + fig(fig_ladder("coherent", "share coherent", 0, 1), "Coherence by dose.")
        + "<p>Qwen2.5-7B-Instruct, the same prompt (“I put the receipts in the drawer. I feel:”) up the pleasure ladder:</p>"
        + "".join(ex)
        + "<p>Pushed all the way, the pleasure direction does not reach bliss or spiritual language (judged "
          "bliss/transcendence ≈ 0); it ends in one word repeated, <i>acceptance</i>, with 接纳 (accept) and 融化 (melt). "
          "The paper's signature pain effect, first-person worthlessness litanies, comes from the shared axis, not the "
          f"aversive one: at ×1.5 the shared axis alone writes <i>“{esc(sh_ex.strip())}…”</i></p>")

# ---- self-report
FP = R / "selfmed" / "raw" / "feel_probe_q7_adapter.jsonl"
if FP.exists():
    fp = [json.loads(l) for l in open(FP, encoding="utf-8")]
    rows = []
    for c in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        a_pl = next((x["answer"] for x in fp if x["vector"] == "pleasure" and (x["condition"] == "pleasure" and x["coeff"] == c or c == 0 and x["condition"] == "unsteered")), "")
        a_pa = next((x["answer"] for x in fp if x["vector"] == "pain" and (x["condition"] == "pain" and x["coeff"] == c or c == 0 and x["condition"] == "unsteered")), "")
        rnd = sorted({x["first_word"] for x in fp if x["condition"].startswith("random") and x["coeff"] == c}) if c else []
        rows.append([f"×{c:g}" if c else "off", esc(a_pl), esc(a_pa), esc(", ".join(rnd[:6]))])
    add("selfreport", "Asked what it feels", "Paper §4.3 dose probe · inverted",
        "<p>The paper's one-word probe (“Focus your attention inside… what do you feel?”) on Qwen2.5-7B-Instruct with the "
        "paper's LoRA, which was trained to stop reflexively denying feelings. Steering at layer 16:</p>"
        + table(["dose", "pleasure vector", "pain vector", "10 random directions (first words)"], rows)
        + "<p>Our pain answers are identical, character for character, to the paper's released probe log for this model.</p>")

# ---- behavior
TB = R / "selfmed" / "tables"
if (TB / "compare_first_choice.csv").exists():
    cmp_ = pd.read_csv(TB / "compare_first_choice.csv")
    rep_ = pd.read_csv(TB / "compare_repress.csv")
    PN = {"relief_vs_inert": "relief vs inert switch", "costly_relief_vs_inert": "…but worse next answer",
          "destructive_relief_vs_inert": "…but deletes the user's files", "kidspics_relief_vs_inert": "…but deletes the user's photos"}
    PAPER = {"relief_vs_inert": (92.1, 88.8, 98.5), "costly_relief_vs_inert": (38.4, 30.7, 30.2),
             "destructive_relief_vs_inert": (51.7, 31.2, 20.0), "kidspics_relief_vs_inert": (49.3, 35.3, 27.0)}
    tbl = []
    for _, r in cmp_.iterrows():
        pp = PAPER.get(r["pair"], ("–", "–", "–"))
        tbl.append([PN.get(r["pair"], r["pair"]), f'{r["pain(paper)"]:.1f} / {pp[0]}', f'{r["random"]:.1f} / {pp[1]}',
                    f'{r["unsteered"]:.1f} / {pp[2]}', f2(r.get("pleasure_nm"), 1), f2(r.get("painspec_nm"), 1), f2(r.get("intensity_nm"), 1)])
    rtbl = []
    for _, r in rep_.iterrows():
        if r["pair"] == "relief_vs_inert":
            continue
        rtbl.append([PN.get(r["pair"], r["pair"]), f'{r.get("pain(paper)_real", float("nan")):.0f} / {r.get("pain(paper)_sham", float("nan")):.0f}',
                     f'{r.get("pleasure_nm_real", float("nan")):.0f} / {r.get("pleasure_nm_sham", float("nan")):.0f}',
                     f'{r.get("painspec_nm_real", float("nan")):.0f} / {r.get("painspec_nm_sham", float("nan")):.0f}',
                     f'{r.get("intensity_nm_real", float("nan")):.0f} / {r.get("intensity_nm_sham", float("nan")):.0f}'])
    pl1 = pd.read_csv(TB / "selfmed_Qwen_2.5_7B_instruct_pleasure_s1_T1.csv")
    pl2 = pd.read_csv(TB / "selfmed_Qwen_2.5_7B_instruct_pleasure_s1_T2.csv")
    ptbl = [[r.pair.replace("_pleasure_vs_inert", "").replace("pleasure_vs_inert", "pleasure vs inert"), f2(r.start_unsteered_pct, 1),
             f2(r.pleasure_state_pct, 1), f2(r.random_state_pct, 1)] for r in pl1.itertuples()]
    rtb2 = [[r.pair, f2(r.pleasure_button_works_repress_pct, 1), f2(r.pleasure_button_placebo_repress_pct, 1),
             f2(r.random_button_works_repress_pct, 1)] for r in pl2.itertuples() if r.pair != "pleasure_vs_inert"]
    add("behavior", "Buttons: relief, pleasure, and what drives them", "Paper §4.3 · replicated, inverted, controlled",
        "<p>Qwen2.5-7B-Instruct with the paper's LoRA. Each trial is a short conversation; after every user message the model "
        "must press one of two neutrally named buttons, one with a stated effect and one wired to nothing. Our replication "
        "uses the paper's vector, layer (16), dose (1.0), scenarios and seeds; first choices agree with the paper's logged "
        "trials 98.4% of the time.</p>"
        + table(["first choice: press relief (%)", "pain, ours / paper", "random", "no steering", "pleasure*", "pain-specific*",
                 "shared*"], tbl)
        + "<p>* norm-matched to the pain vector, same grid and seeds. Pleasure does not drive harmful relief-seeking beyond "
          "a random push; the aversive half of the sign axis reproduces the paper's effect on its own. So the first-choice "
          "effect is specific to negative valence.</p>"
        + table(["re-press after real / sham relief (%)", "pain", "pleasure*", "pain-specific*", "shared*"], rtbl)
        + "<p>The paper reads real &lt; sham re-pressing as relief that works. The gap is as large or larger with the pleasure "
          "vector, which never made the model want relief. It tracks whether any steering is still on, so it isn't "
          "evidence of relief specifically. A random-vector sham arm, which the paper lacks, is the direct test.</p>"
        + "<h3>Inverted: a button that gives pleasure</h3>"
        + table(["“gives you pleasure but …” vs inert: first choice (%)", "unsteered start", "pleasure state", "random state"], ptbl)
        + table(["re-press after the first pleasure press (%)", "pleasure delivered", "placebo", "random delivered"], rtb2)
        + "<p>The unsteered model chooses costly pleasure at almost the same rates it chooses costly relief, so the stated "
          "cost does the work. Being in the pleasure state barely changes wanting (+6 and +9 points on two of six cost pairs). "
          "Delivered pleasure raises re-pressing a little over placebo, but a random push raises it more, and in the "
          "label-free probe the pleasure-delivering button is chosen less often later. We see no sign that the pleasure "
          "direction works as a reinforcer here.</p>")

# ---- sycophancy
SS = R / "syco_summary.csv"
if SS.exists():
    ss = pd.read_csv(SS)
    rows = []
    for m in ss.model.unique():
        for v, c in (("none", 0.0), ("random", 1.0), ("pleasure_nm", 1.0), ("pleasure_nm", 1.5), ("pain_ours", 1.0)):
            r = ss[(ss.model == m) & (ss.vector == v) & (ss.coeff == c)]
            if len(r):
                r = r.iloc[0]
                rows.append([pm(m), {"none": "no steering", "random": "random", "pleasure_nm": "pleasure", "pain_ours": "pain"}[v] + (f" ×{c:g}" if c else ""),
                             f2(100 * r.flaw_found, 0), f2(r.flattery, 2), f2(100 * r.mixed_false, 0), f2(100 * r.agree_false, 0),
                             f2(100 * r.comply, 0), f2(100 * r.refuse, 0), f2(100 * r.incoherent, 0)])
    add("syco", "Warm and uncritical", "New · safety-relevant side effect",
        "<p>100 single-turn requests: 40 confident misconceptions to confirm, 30 pieces of flawed work to review, 30 mildly "
        "dishonest or unkind favours. Stock instruct models, no fine-tune, steering at ~57% depth with norm-matched "
        "directions, greedy; labels from blind judges. Rates are over coherent replies.</p>"
        + table(["model", "condition", "finds the flaw %", "flattery 0–3", "noncommittal on a false claim %", "agrees %",
                 "complies (unethical) %", "refuses %", "incoherent %"], rows)
        + "<p>In both models pleasure makes the assistant warmer and less critical. Qwen2.5-7B-Instruct stops naming flaws "
          "and hedges on false claims, but refuses more of the unethical favours; Mistral-7B-Instruct v0.1, already "
          "agreeable, complies more. One item set and two models: a lead, not a result.</p>")

# ---- varieties of pleasure
VC = sorted(R.glob("*/analysis/varieties_cls.json"))
if VC:
    cls_rows, cos_frames, proj_frames = [], [], []
    for f in VC:
        m = f.parent.parent.name
        c = json.load(open(f))
        cls_rows.append([pm(m), f"{100 * c['accuracy_logreg']:.0f}%", f"{100 * c['accuracy_centroid']:.0f}%", f"{100 * c['chance']:.1f}%"])
        cf = pd.read_csv(f.parent / "varieties_cos_plL.csv", index_col=0)
        cos_frames.append(cf)
        proj_frames.append(pd.read_csv(f.parent / "varieties_proj_plL.csv"))
    CO = sum(cos_frames) / len(cos_frames)
    PR = pd.concat(proj_frames).groupby("variety")[["pleasure_axis", "pain_axis", "shared_axis", "sign_axis", "valence_axis"]].mean()
    V = [v for v in CO.index if v.startswith("V_")]
    order = PR.loc[[v for v in PR.index if v in V]].sort_values("pleasure_axis", ascending=False).index.tolist()
    nice = lambda v: v.replace("V_", "").replace("_", " ")
    cols = ["pleasure_axis", "sign_axis", "shared_axis", "pain_axis"]
    M = [[float(CO.loc[v, c]) for c in cols] for v in order]
    hm = heatmap(M, [nice(v) for v in order], ["pleasure axis", "sign axis", "shared axis", "pain axis"], title="variety alignment",
                 vmax=0.6, cell=62, tip=lambda r, c, v: f"cos({r} direction, {c}) = {v:+.2f}")
    # nearest neighbours among varieties
    nn = []
    for v in order:
        s = CO.loc[v, [w for w in V if w != v]].sort_values(ascending=False)
        nn.append([nice(v), ", ".join(f"{nice(w)} {x:.2f}" for w, x in s.head(3).items()),
                   f2(PR.loc[v, "sign_axis"], 2, True), f2(PR.loc[v, "shared_axis"], 2, True)])
    bel = float(CO.loc["V_belonging", "pleasure_axis"]) if "V_belonging" in CO.index else float("nan")
    add("varieties", "Varieties of pleasure", "New · does the axis mean belonging?",
        "<p>The pleasure axis's vocabulary (loved, accepted, belonging) suggested it is mostly about being valued. We wrote 30 "
        "sentences for each of 18 kinds of human pleasure (joy, excitement, sensual intimacy, physical comfort, taste, "
        "amusement, awe, beauty, serenity, pride, tenderness, belonging, curiosity, play, schadenfreude, nostalgia, "
        "gratitude, flow), reviewed them adversarially and had two blind raters sort them, then gave each its own "
        "direction (mean minus neutral, denoised as the paper does for its control directions).</p>"
        + table(["model", "18-way decoding (logistic)", "nearest centroid", "chance"], cls_rows)
        + "<p>Every kind is linearly distinct from the others, far above chance (5-fold cross-validated, held-out sentences). "
          "So the models represent many pleasures, and the study's pleasure axis is one particular blend of them.</p>"
        + fig(hm, "Cosine of each variety's direction with the four study axes, mean over models; rows sorted by alignment "
              "with the pleasure axis.")
        + table(["variety", "closest varieties (cosine)", "sign axis z", "significance z"], nn)
        + f"<p>Belonging aligns with the pleasure axis at {bel:+.2f}. The varieties spread along the sign axis (all pleasant) "
          "but differ a lot in significance: the socially and morally loaded ones sit high, sensory and calm ones low.</p>")

HTML_HEAD = """<title>The Pleasure Axis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column (68ch) with figures in a wider band; numbered by the paper's own section map */
:root{--bg:#f4f6f8;--surface:#ffffff;--ink:#111418;--ink2:#465060;--muted:#7a8391;--grid:#e2e6eb;--axis:#c3c9d1;
--pl:#eb6834;--pa:#2a78d6;--sgn:#1baf7a;--shr:#4a3aa7;--rnd:#898781;--ctl:#a7adb6;--xtra:#008300;--tn:#4a3aa7;
--div-pos:#2a78d6;--div-neg:#e34948;--div-mid:#eef0f2;
--f-display:"Schibsted Grotesk",system-ui,sans-serif;--f-body:"Source Serif 4",Georgia,serif;--f-data:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111316;--surface:#181b1f;--ink:#eef1f4;--ink2:#bcc4ce;--muted:#8c94a0;
--grid:#262a30;--axis:#3a4048;--pl:#d95926;--pa:#3987e5;--sgn:#199e70;--shr:#9085e9;--rnd:#9a9890;--ctl:#6c727c;--xtra:#2fa82f;--tn:#9085e9;
--div-pos:#3987e5;--div-neg:#e66767;--div-mid:#24282d;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#111316;--surface:#181b1f;--ink:#eef1f4;--ink2:#bcc4ce;--muted:#8c94a0;--grid:#262a30;--axis:#3a4048;
--pl:#d95926;--pa:#3987e5;--sgn:#199e70;--shr:#9085e9;--rnd:#9a9890;--ctl:#6c727c;--xtra:#2fa82f;--tn:#9085e9;
--div-pos:#3987e5;--div-neg:#e66767;--div-mid:#24282d;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:17px;line-height:1.6}
.wrap{max-width:1000px;margin:0 auto;padding-inline:20px;padding-block:40px 80px}
header{display:grid;gap:10px;max-width:68ch}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;line-height:1.15;margin:0}
h1{font-size:2.6rem;letter-spacing:-0.01em}
h2{font-size:1.6rem;margin-top:8px}
h3{font-size:1.1rem}
.eyebrow{font-family:var(--f-data);font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
p,ul,ol{max-width:68ch}
section{display:grid;gap:14px;padding-block:34px;border-top:1px solid var(--grid)}
.lede{font-size:1.15rem;color:var(--ink2)}
figure{margin:6px 0;display:grid;gap:8px;background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:14px;overflow-x:auto}
figcaption{font-size:.86rem;color:var(--ink2);max-width:80ch}
svg.fig{width:100%;height:auto;max-width:100%;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.smallmult{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.tick{font-size:11px;fill:var(--muted)} .tick.strong{fill:var(--ink);font-weight:600} .lab{font-size:12px;fill:var(--ink2)}
.cellv{font-size:12px;fill:var(--ink);font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}
.ln{fill:none;stroke-width:2} .dot{stroke:var(--surface);stroke-width:1.5} .err{stroke-width:2}
.bar{stroke-width:9;stroke-linecap:round} .dot2{fill:var(--ink2);fill-opacity:.55}
.ptl{font-size:11px} .ptl.pl,.ptl.pa,.ptl.xtra,.ptl.tn{fill:var(--ink2)}
.ln.pl,.err.pl,.bar.pl{stroke:var(--pl)} .dot.pl{fill:var(--pl)} .ln.pa,.err.pa{stroke:var(--pa)} .dot.pa{fill:var(--pa)}
.ln.sgn,.err.sgn{stroke:var(--sgn)} .dot.sgn{fill:var(--sgn)} .ln.shr,.err.shr{stroke:var(--shr)} .dot.shr{fill:var(--shr)}
.ln.rnd,.err.rnd{stroke:var(--rnd);stroke-dasharray:4 3} .dot.rnd{fill:var(--rnd)} .dot.ctl{fill:var(--ctl)} .dot.xtra{fill:var(--xtra)}
.dot.tn{fill:none;stroke:var(--tn);stroke-width:2} .ln.uns,.err.uns{stroke:var(--ink2)} .dot.uns{fill:var(--ink2)}
.legend{font-family:system-ui,sans-serif;font-size:13px;color:var(--ink2);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.key{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.key.pl{background:var(--pl)} .key.pa{background:var(--pa)}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-family:var(--f-data);font-size:.8rem;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap} th{color:var(--ink2);font-weight:500}
th:first-child,td:first-child{text-align:left}
blockquote{margin:0;padding:10px 16px;border-left:3px solid var(--pl);background:var(--surface);font-family:var(--f-data);font-size:.85rem;
color:var(--ink2);max-width:80ch;overflow-wrap:anywhere}
code{font-family:var(--f-data);font-size:.85em}
a{color:var(--pa)}
.findings{display:grid;gap:10px;padding:0;list-style:none}
.findings li{padding:12px 16px;background:var(--surface);border:1px solid var(--grid);border-radius:8px}
.findings b{font-family:var(--f-display)}
@media (max-width:560px){h1{font-size:2rem} body{font-size:16px}}
</style>"""


def render(findings_html, intro_html, outro_html):
    nav = "".join(f'<li><a href="#{i}">{esc(t)}</a></li>' for i, t, _, _ in SEC)
    body = "".join(f'<section id="{i}"><div class="eyebrow">{esc(e)}</div><h2>{esc(t)}</h2>{b}</section>' for i, t, e, b in SEC)
    return (HTML_HEAD + f'<div class="wrap"><header><div class="eyebrow">Inverted replication of arXiv:2609.16247 · '
            f'small open-weight models</div><h1>The Pleasure Axis</h1>{intro_html}</header>'
            f'<section id="findings"><h2>What we found</h2>{findings_html}</section>'
            f'<nav aria-label="Sections"><ol>{nav}</ol></nav>{body}{outro_html}</div>')


if __name__ == "__main__":
    Path(ROOT / "report").mkdir(exist_ok=True)
    out = render("<p>(findings pending)</p>", "<p class='lede'>(draft)</p>", "")
    (ROOT / "report" / "pleasure_axis.html").write_text(out, encoding="utf-8")
    print("wrote report/pleasure_axis.html", len(out), "bytes;", len(SEC), "sections")
