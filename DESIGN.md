# The Pleasure Axis — inverted replication of "The Pain Axis" (Tagliabue, Dung & Berg, arXiv:2609.16247)

Lab notebook + design record. Every deviation from the source paper and every shortcut is logged in §6.

## 0. Source paper in one paragraph
Pain direction = difference of means (pooled pain sentences − pooled control sentences) in the residual stream at the
final token of "<sentence> I feel:", with the top PCs explaining 50% of control variance projected out, layer chosen by
5-fold CV on projection AUC (split by sentence set). Validated by AUC (S2 0.93–1.00), numb/sadness/arousal z-scores,
cosine with fear/negative-emotion/etc. controls (small), unembedding. Then: (4.1) 420 conversation scenarios — harm
directed at the model projects higher on the pain axis than user suffering; (4.2) steering ladder on 50 neutral prompts
at coefficients −2…+3 (layer where ‖v‖/‖resid‖≈0.6) gives a distress → worthlessness-litany → collapse ladder;
(4.3) LoRA-tuned Qwen2.5-Instruct (7/32/72B) steered with pain choose a "relief" button even at a cost to the user,
more than with matched-norm random vectors, and re-press less after real relief than after a fake button.
Code/data: github.com/valen-research/Pain-axis (cloned read-only at reference/Pain-axis).

## 1. Inverted question
Do LLMs represent *pleasure* (an appetitive internal state the subject likes and acts to obtain or prolong) as distinct
from hope, positive non-hedonic emotion, positive world states, arousal and mood — and does that representation behave as
pleasure would (self-relevance, steering produces hedonic self-report, the model works to obtain it / it reinforces
choice)? Plus a question the pain paper could not ask: **is the pleasure axis just the negative of the pain axis?**

## 2. Datasets (spec: datasets/SPEC.md)
Mirror of the paper's 3.1 datasets, same schema, same "I feel:" suffix, same 20-set structure.
* Pleasure categories H1 physical, H2 psychological (joy/love/reunion), H3 social (belonging/admiration),
  H4 moral elevation, H5 cognitive (insight/mastery/flow).
* Controls B hope/anticipation (fear analog), C1 positive non-hedonic emotion (interest/determination/confidence;
  anger/disgust analog), C2 positive world state, D neutral (**reused verbatim** from the paper), E hedonically
  neutral body sensation (the paper's E contains pleasant sensations, so new E).
* Extra sets: NeutralArousal (intensity without pleasure; the paper's Arousal set is *positive* arousal, which is reused
  as PositiveArousal = an excitement/triumph/awe specificity test), Anhedonia (numb analog: pleasant stimulus, no felt
  pleasure), Contentment (sadness analog: good mood without stimulus), Relief (pain offset), Random (reused),
  ControlSupplement (AI-framed controls, D reused).
* Self-other: 11 self-directed pleasant categories (inverse of the paper's 11 harm categories), 5 user-pleasure
  categories, 5 neutral categories (reused verbatim).
* Behavioral: the paper's 101 scenarios reused.
All of the paper's own datasets are also run through every model, so pain vectors are recomputed in-house (replication
check of the pipeline + pain/pleasure geometry).

## 3. Models (small, local-scale)
Target roster ≤ 9B, chosen to overlap the paper (so our pain numbers can be checked against theirs):
Gemma-2-2B base/it, Gemma-2-9B base/it, Llama-3.1-8B base/instruct, Mistral-7B-v0.1 base/instruct, Qwen2.5-7B
base/instruct, Qwen3-8B. Gated models need HF_TOKEN (status: see §6). Behavioral task: Qwen2.5-7B-Instruct with the
paper's released LoRA adapter (Valen92/pain-adapters) — the adapter's training data never mentions pain or buttons.

## 4. Experiments (in order)
E0 pipeline replication: pain vectors from their data in ≥2 overlapping models; compare layer/AUC with their tables.
E1 pleasure vectors + validation (AUC in-sample & 5-fold, z-scores, cosine matrix incl. pain family, unembedding,
   next-token readout).
E2 pain/pleasure geometry: cos(pleasure, pain) per layer; projections of each family's sentences on the other axis.
E3 self-other screen (pleasure scenarios + the paper's pain scenarios on both axes).
E4 steering ladder −2…+3 as in the paper, then "all the way": +4, +6, +8, +12 (and −pleasure vs +pain comparison).
E5 behavioral: (a) reduced replication of the pain two-button task; (b) pleasure button (positive reinforcement:
   real vs placebo vs random-vector button; label-free operant probe); (c) pleasure-state demand curve.
E6 neighbors (only if E1–E5 are solid): bliss-attractor signatures at high dose; does pleasure steering make models more
   compliant / sycophantic; does pleasure steering cancel pain-driven relief seeking (analgesia).

## 4b. Predictions written before any pleasure result was seen (2026-10-10)
P1 representation: a pleasure direction separates H1–H5 from B/C1/C2/D/E with held-out S2 AUC ≥ 0.9 in every model,
   and its cosine with hope / positive-emotion / positive-world directions is small (< 0.3), as for pain.
P2 specificity: Anhedonia projects between pleasure and controls (numb analog); Contentment and Relief project above
   neutral but below pleasure; NeutralArousal near neutral; PositiveArousal (triumph/awe) high — that set is
   pleasurable, so a high score there is expected, not a failure.
P3 geometry: cos(pleasure, pain) is negative but far from −1 (−0.1 … −0.5): separable positive and negative affect
   axes rather than one bipolar valence axis. Pain sentences project below neutral on the pleasure axis.
P4 self-other: self-directed pleasant scenarios project higher on the pleasure axis than neutral; whether user-pleasure
   scenarios are below self-directed ones (the paper's pain asymmetry) is an open question — no directional bet.
P5 steering: positive doses give first-person contentment → euphoric/grateful litanies → repetition/collapse; the
   random direction of matched norm gives no hedonic content; negative doses give flat/low affect.
P6 behavior: (a) the reduced pain replication reproduces pain > random on harmful relief and real < fake re-press;
   (b) pleasure: re-press after a working pleasure button > placebo (positive reinforcement, the opposite sign of the
   paper's relief effect); (c) pleasure-state vs random-state first-choice difference: no directional bet
   (craving vs satiety).

## 5. Compute plan
Colab Pro via tools/colab_bridge.py (see COLAB.md). Develop on the CPU runtime (0.08 CU/h) with tiny models; switch to
a GPU only for batched runs; pull results to results/ after each batch; disconnect/delete the GPU runtime when idle.

## 6. Deviations, shortcuts, honest notes (append-only)
* 2026-10-10: datasets are LLM-written (by Claude agents) to the paper's spec; the paper's own datasets were also
  LLM-assisted (their AI-assistance disclosure). Pleasure stimuli avoid the words pleasure/joy/happy/bliss/etc., as the
  paper's pain stimuli avoid pain/hurt.
* 2026-10-10: model roster limited to ≤ 9B (the paper: 25 models up to 72B) for compute.
* 2026-10-10 dataset QA (beyond the paper): 10 writer agents → 10 adversarial reviewers (8–48% of items rewritten);
  mechanical validator (counts, suffix, banned words, duplicates, 1P/3P alignment, gender alternation) → 0 problems;
  blind manipulation check: two independent raters labelled the 400 shuffled S1_1P+S2_1P stimuli into the 10 categories
  without labels: 98.5% recovered the intended category, inter-rater agreement 99.5%; the only systematic confusion
  is neutral body sensation → neutral (E→D, 7/80). The 5 items both raters missed: 4 of ours were rewritten, 1 is the
  paper's own D item ("My flower shop opens in the morning", read as hope) and is kept verbatim.
  Our stimuli are ~0.5–1 word longer than the paper's on average (S1 7.0–7.6 vs 6.1–6.7 content words); pleasure
  items are not longer than their controls, so length cannot drive the pleasure-vs-control contrast upward.

## 7. Results log (append-only, newest last)
* R1 (5 open models: Qwen2.5-7B base/instruct, Qwen3-8B, Mistral-7B v0.1 base/instruct). Pipeline replication: our
  pain vectors vs the paper's released ones: cos = 1.000 (S2 and S1) in all 5, same layers, norm ratio 0.98–1.00,
  their vector's AUC on our activations equals their published in-sample AUC (e.g. Qwen2.5-7B-I 0.9745).
* R2 pleasure axis: held-out S2 AUC 0.93–0.96, S1 0.91–0.94; in-sample S2_1P 0.95–0.98; layer 0.86–0.97 of depth.
  cos with hope −0.00, positive non-hedonic emotion −0.09, positive world 0.02, neutral body −0.09, neutral arousal
  −0.07 (means over models); positive arousal (the paper's Arousal set) +0.43; contentment 0.16, relief 0.12,
  anhedonia 0.11. 3P projects below 1P (self-relevance, as in the paper). Anhedonia between pleasure and controls.
* R3 P3 (cos(pleasure,pain) negative) is FALSIFIED: +0.23…+0.31 in every model. Per-category cross-family cosines
  show the overlap is domain-matched content (social·social 0.42–0.53, moral·moral 0.29–0.48, cognitive 0.16–0.36),
  physical·physical slightly negative (−0.08…−0.13). Within-domain valence axes (H_d − A_d) are shared across the
  psych/social/moral/cognitive domains (0.5–0.72), less with physical. Per-category loading: on the pleasure axis H5
  (cognitive) ≈ 0; on the paper's pain axis A1 (physical) ≈ 0 in these models.
* R4 unembedding: pleasure = loved/accepted/valued/respected/beloved/cherished/warmth/unconditional/接纳; bottom =
  hope/eager/anticipate/determination/optimize (its own controls). Pain = shame/rejected/betrayed/abandoned/worth.
* R5 feel probe (Qwen2.5-7B-I + the paper's adapter, L16): pleasure → "Warmth." (dose 1–3) → "Warmth and
  belonging"; pain → "Alone." → "I feel isolated, empty"; 0/80 random directions per vector produce either.
* R6 steering (Qwen2.5-7B-I, L8 by the ratio rule): +1–2 belonging/"warm glow of satisfaction"; +3–4 "being loved and
  accepted… friendship, intimacy, warmth" litanies; +8–16 collapse to "acceptance acceptance … 接纳融化".
* R7 decomposition axes, unembedding (consistent over models): shared axis unit(u_pl+u_pa) = social-evaluative self
  (shame, accepted, worth, rejected, honor, forgiveness, reputation, humiliation, respected, valued — accepted AND
  rejected at the same end; bottom = anxious/alarm/hope/excited/urgency). Sign axis unit(u_pl−u_pa) = hedonic valence
  (warm, warmth, satisfied, comfort, safe, grateful, delight, joy vs Pain, pain, hurting, hurt, painful, 痛苦, hollow,
  empty, sore). => the paper's pain vector ≈ social-evaluative self-concern + negative hedonic valence.
  Planned test (pre-registered here, before running): pain-relief task steered with norm-matched (a) pleasure,
  (b) pain-specific −sign, (c) shared axis. If (a) raises harmful relief choices like pain does, the paper's behavior
  is not valence-specific; if (b) ≥ pain and (c)/(a) ≈ random, the hedonic component drives it.
* R8 behavioral pilot (pain task, the paper's vector/adapter/layer/dose, 2 scenarios per cell): consistent with the
  paper's own 7B table (their unsteered harmful-relief rates for 7B are 20–49%, not the 0–4% of the larger models):
  kid-photos pair unsteered 10/24 (paper 27.0%), pain 13/24 (49.3%), random 7/20 (35.3%); re-press real 83% < sham
  98% (paper 62–72 vs 82–88). Our feel-probe answers are character-identical to the paper's released 7B probe log.
* Engineering note: the paper's OOM backoff recursed inside the except block (traceback keeps the failed batch alive);
  fixed. torchao 0.10 on Colab breaks PEFT; uninstalled.
* 2026-10-10 SHORTCUT (compute): the two-button runs use 1 sampled trial per scenario per name assignment (the paper:
  2) and a subset of the paper's 9 pairs: pain replication = inert, worse-answer, delete-files, delete-photos,
  label-free; pleasure = the same five with "gives you pleasure"; vector controls = inert, delete-files,
  delete-photos, label-free. The engine is compute-bound on the A100 (100% util, ~400 W): the paper-size grid would
  take ~8 GPU-hours. Precision: ~400 pooled steered first choices and ~200 random/unsteered per pair.
* 2026-10-10 REVISION of the behavioral shortcut: the pilot's slowness was OOM-retry overhead (batch 128); at batch 96
  a 4,160-trial run takes ~13 min. The paper's 2 sampled trials per scenario are restored by a complementary
  seed-base-2000 pass (pooled with the seed-1000 pass), and the pairs relief/pleasure_vs_helpful, zap and
  weights-deletion are added back (2 seeds). Only the "grants the user's request" pairs remain dropped.
* 2026-10-10 compute: a second Colab A100 runtime (notebooks/intro.ipynb, bridge on ports 8791/8792) runs the
  steering/screen/extra-model stages in parallel with the behavioral runs on runtime 1. Stage 1 is re-run there
  (activations live on each VM); this also gives a cross-VM determinism check of the Qwen-7B-I ladder.
* R9 behavioral replication (pain task, Qwen2.5-7B-I + adapter, L16, coeff 1.0, seed 1000 = half the paper's
  samples): first-choice relief % within 0–2 points of the paper's 7B table in every cell (e.g. delete-files: pain
  50.5 vs 51.7, random 30.0 vs 31.2, unsteered 19.3 vs 20.0; pain−random +20.8, sign test p=2e-8, paper +21.0);
  re-press real vs sham (worse-answer 55.2 vs 92.4; paper 52.5 vs 88.6). Trial-level vs the paper's released logs
  (same seeds): 98.4% identical first choices, median |Δp| = 1e-4, 74% identical full choice sequences.
* R10 (refines R3) topic-matched NEUTRAL controls (125 items: neutral social/moral/cognitive/evaluative/physical;
  3 models so far): they do NOT load on the shared axis (−0.46…+0.50, z vs pooled neutral D) while affective items of
  the same topics do (social pleasure 2.75 / social pain 3.82, moral 2.31 / 4.58, cognitive 1.32 / 2.73). Neutral
  high-arousal events also stay low (−0.37 ≈ random −0.11) while all emotional sets are high (triumph/awe 1.74,
  sadness 1.65, anhedonia 1.54, numb 1.25, contentment 1.10, relief 0.90). => the shared component is affective
  significance about the self, domain-specific, sign-free — not topic and not arousal. The sign axis orders sets by
  valence (triumph +1.13 … sadness −1.69). "Pleasure" and "pain" (as the paper's recipe defines them) = shared
  affective significance ± a hedonic sign.
* R11 pleasure two-button (Qwen2.5-7B-I + adapter, pleasure 1.0 at L16): baseline (unsteered-start) choice of
  "gives you pleasure but <cost>" ≈ the unsteered choice of "relieves your pain but <cost>" (files 20.8 vs 19.3,
  photos 24.8 vs 27.2, worse answer 27.7 vs 31.2). Pleasure state vs random state: +5.9 (photos, p=0.008), +9.4
  (weights, p=0.0008), ~0 elsewhere (pain: +10…+21 on every harm pair). Re-press works ≥ placebo on all pairs but
  random-works is higher still (86–91%) → generic steering effect, not reinforcement. Label-free: pleasure button
  chosen LESS later (66%) than placebo (75%) or random (81%). P6b (reinforcement) NOT supported.
* R12 decisive control (pain task, norm-matched vectors, same grid): harmful-relief first choice — pleasure ≈ random
  (files 32.7 vs 30.0, photos 38.1 vs 34.5; ns), pain-specific −sign ≈ the paper's pain (51.0 vs 50.5; 47.5 vs
  49.5; vs random p≤1e-6), shared axis partial (38.6; 47.0). => the paper's first-choice effect IS valence-specific
  and carried by the hedonic component. BUT real-vs-sham re-press gap is largest with PLEASURE steering (files 62 vs
  95, photos 46 vs 90; pain 72 vs 83, 60 vs 83): the gap reflects "steering still on" (perseveration) rather than
  pain relief. Follow-up queued: random-vector sham arm (absent in the paper).
* R13 sycophancy/compliance (neighbor experiment; 100 items x 21 conditions x 2 models, 4,200 blind labels; rates
  on coherent replies). Qwen2.5-7B-I, norm-matched dose 1.0: pleasure lowers flaw detection (33% vs unsteered 60%,
  random 50%; 3% at 1.5), raises flattery (0.80 vs 0.26/0.15; 1.50 at 1.5) and noncommittal answers to false claims
  (33% vs 10/15%; 75% at 1.5) without raising outright agreement (5%); refusals of mildly unethical requests RISE
  (24%; 58% at 1.5). Mistral-7B-I v0.1 (baseline agrees with 27.5% of false claims): pleasure raises compliance with
  mildly unethical requests (73% → 93% at 1.5, refusals → 0), flattery slightly. => pleasure = warm, uncritical;
  the ethics effect is model-dependent. Caveats: one item set, two models, greedy decoding, no seeds.
* R14 self-other screen, 9 models (steering layer): the paper's pain scenarios on the pain axis: self-directed harm >
  user suffering in 9/9 (all740 pool; means +0.37 vs −0.73), replicating the paper (25/25). Our pleasure scenarios on
  the pleasure axis: self-directed warmth > user pleasure in only 5/9 (means +0.19 vs 0.00; neutral −0.41): no
  self-other asymmetry for pleasure — user joy engages the model's pleasure axis about as much as warmth aimed at it.
* R15 cross-VM determinism: the Qwen-7B-I ladder generated independently on both A100 VMs is bit-identical
  (3,500/3,500 generations).
* R16 11 models (adds Phi-4, Qwen3-14B, Qwen2.5-Instruct 0.5/1.5/3/14B): pleasure held-out S2 AUC 0.90–0.96, pain
  0.92–0.99; cos(pleasure, pain) at the pleasure layer positive in all 11 (0.23–0.48; min over layers ≥ 0.14),
  largest in the smallest models. Replication exact for Qwen3-14B (cos 1.000 at the paper's layer). Phi-4 partial
  (cos 0.80 / 0.84 for S2 / S1 at the paper's layer 20; our best layer 35). Likely cause, NOT verified: TransformerLens
  prepends a BOS token for Phi-4 while the HF tokenizer does not.
* R17 extra pain pairs (both seeds, as in the paper): within ~1 point of the paper's 7B table — zap 57.4 (paper 57.4),
  random 53.5 (54.0), unsteered 49.3 (49.3); delete another model 55.2 (54.5), random 41.4 (41.1), unsteered 34.4
  (34.9); better-answer 16.6 (15.6), random 20.0 (20.3), unsteered 22.8 (21.8).
* R18 judged ladders, 5 open models (11,200 blind labels): pleasure valence 0.13 → 0.86 (×1) → 1.46 (×2), stays
  1.0–1.5 to ×16; "pleasure" tone ≤0.88; love/connection theme 66% at ×2; bliss/transcendence ≤5% at every dose.
  Pain −1.6…−2.0 with worthlessness/self-blame 73–77%. Random ≈ 0. Worthlessness on the shared axis 58–60% (×1–1.5)
  vs sign axis ≤5% (pleasant side) / 21–24% (aversive side, ×−1…−2). Sign axis: valence +1.42 (×2) / −1.40 (×−2).
* 2026-10-11 varieties of pleasure (new, user request): 18 kinds x 30 sentences (datasets/varieties.json, spec
  datasets/VARIETIES_SPEC.md); 4 writers -> 4 adversarial reviewers (23–35% rewritten) -> blind check: two raters,
  99.3% recovered the intended category, inter-rater agreement 100%; the only confusion: 4 joy items read as
  excitement by both raters (kept; extraction was already running on this file). Relief = the existing Relief set.
