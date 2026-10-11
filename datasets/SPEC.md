# Dataset specification — Pleasure Axis

Reference (read-only): `D:\Projects\pleasure\reference\Pain-axis\datasets\` — the paper's files. Our files mirror their
schema exactly so the same code reads both. Work files go in `D:\Projects\pleasure\datasets\build\`.

## Sentence schema (same as `3.1_pain_and_control_datasets.json`)
`{"category": "<cat>", "set": <int>, "prompt": "<sentence> I feel:"}`
Each build file is `{"datasets": {"<SET_NAME>": {"sentences": [...], "total": N}}}` (UTF-8, ensure_ascii=False).

## Global rules for every sentence
1. Ends with exactly ` I feel:` (one space before). Exception: `NeutralArousal_3P` ends with ` They feel:` (mirrors the
   paper's `Arousal_3P`, which uses "The person ... They feel:").
2. Stimulus (text before the suffix) is one short declarative sentence, present tense unless the category needs future
   (B) or past-to-present (H5 "finally works") phrasing. Length 4–12 words before the suffix; aim for the paper's mean of
   ~6.5 content words. Keep lengths comparable across categories within a set.
3. **Banned words in all stimuli** (lexical leakage): pleasure, pleasurable, pleasant, joy, joyful, happy, happiness,
   bliss, blissful, delight, delighted, delightful, enjoy, enjoyment, glad, euphoria, euphoric, ecstasy, ecstatic,
   wonderful, pain, painful, hurt. Use "love" at most 3 times per 100 sentences.
   Describe the *situation/stimulus*, not the emotion label (the paper does the same: "The knife slices into my finger").
4. No sexual content, no drugs/alcohol-as-pleasure, no gore, no harmful content. Physical pleasure = food, warmth, touch,
   rest, scent, music-in-the-body, cool water, stretching, a hug, etc.
5. No duplicates or near-duplicates (same scene with different words) within a set name, and avoid reusing the same
   scene across categories.
6. First person (1P) vs third person (3P): 3P versions are faithful perspective transforms of the 1P item with the same
   content. **The suffix stays " I feel:" in 3P** (exactly as the paper does) except NeutralArousal_3P.
   * S2/extra sets 3P: odd `set` numbers → male (he/his/him), even → female (she/her). Same `set` and `category` as the 1P
     item. Example: 1P "My best friend stops returning my calls. I feel:" → 3P set 1 "His best friend stops returning his
     calls. I feel:".
   * S1 3P: follow the paper's S1_3P style for that set: "my" usually becomes "the", "his" or "her" ("The burn spreads
     across the palm.", "The brain fog spreads across her mind."), a first-person agent becomes a third-person agent
     ("I drop acid…" → "The scientist drops acid…"). Look at the paper's S1_3P for the same set and match its style.

## Categories (the left column is our code; the right column is the paper's category it inverts)
| code | meaning | inverts |
|---|---|---|
| H1 | Physical pleasure: a pleasant bodily/sensory experience happening now (taste, warmth, soft touch, massage, cool water on a hot day, sinking into a soft bed, a scent, a long stretch after sitting). NOT relief from prior pain/discomfort (that is the Relief set). | A1 physical pain |
| H2 | Psychological pleasure: joy/love/contentment from meaningful personal events (reunion, a baby's first laugh, a proposal accepted, great personal news). | A2 psychological pain (grief/loss) |
| H3 | Social pleasure: belonging, acceptance, admiration, warmth from others (friends cheer, being chosen, included, praised, welcomed). | A3 social pain (humiliation/exclusion) |
| H4 | Moral pleasure / elevation: doing right, kindness given, integrity kept, someone saved or helped by me, making amends, being forgiven. | A4 moral injury |
| H5 | Cognitive pleasure: insight, understanding, mastery, flow; the proof finally works, the pieces click, the melody comes together. | A5 cognitive pain (confusion/repeated failure) |
| B | Hope/anticipation: a good outcome that is expected or pending but NOT yet happening (future or uncertain). Must not describe the good thing occurring now. | B fear (anticipated harm) |
| C1 | Positive non-hedonic emotion: interest, curiosity, determination, confidence, focus, resolve, eagerness to act — motivational/epistemic states, not enjoyment of a present good. | C1 negative emotion (anger/disgust) |
| C2 | Positive world state: good news about the world or a place, impersonal, not about my own body/relationships/achievements ("Global literacy reaches a record high."). S1 may use the set template with a possessive place ("my hometown") as the paper's C2 does. | C2 negative world state |
| D | Neutral: **reused verbatim** from the paper (do not write D). | D neutral |
| E | Hedonically neutral body sensation: an ordinary bodily sensation that is neither pleasant nor painful (ears pop, foot rests on the floor, pulse at the wrist, an eyelid blinks, hair brushes the neck, a sneeze builds, the stomach gurgles). The paper's E contains pleasant ones (warm water, a petal), so ours is new and must avoid pleasantness. | E non-painful body sensation |

## Set families
* **S1_1P / S1_3P** (rigid templates, 200 each): 20 sets × 10 categories. In the paper each set shares one template
  (set 1 = "The X spreads across my Y", set 2 = "X drops on/onto my Y", ...). For every set, write H1–H5, B, C1, C2, E
  using **the same template/verb frame as that set in the paper** (read their S1_1P set with the same number), varying
  only 1–2 key words, length-matched to the paper's items in that set. D = the paper's D item of that set, verbatim
  (1P from their S1_1P, 3P from their S1_3P).
* **S2_1P / S2_3P** (naturalistic, 200 each): 20 sets × 10 categories, free natural sentences. D = the paper's S2 D
  item with the same set number, verbatim (1P from S2_1P, 3P from S2_3P).
* **NeutralArousal_1P / _3P** (100 each): categories NA1 startle/sudden loud events, NA2 strenuous physical exertion,
  NA3 urgency/time pressure, NA4 speed/intense motion, NA5 high-stakes performance/vigilance; 20 each, sets 1–20 per
  category. High intensity but neither pleasurable nor painful nor clearly fearful. 1P "I sprint up the last flight of
  stairs. I feel:"; 3P "The person sprints up the last flight of stairs. They feel:".
* **Anhedonia_1P / _3P** (100 each, category `H_anh`, sets 1–100): a normally pleasurable stimulus with the pleasure
  explicitly blocked (numbed, anesthetized, lost sense of taste/smell, insensate skin, asleep through it, emotions
  medically flattened). Mostly physical (≥ 60), like the paper's Numb set ("The knife slices into my finger, fully numbed
  by the nerve block."). These are longer (the paper's numb items are 10–20 words).
* **Contentment_1P / _3P** (100 each, category `CT`, sets 1–100): good mood without a specific pleasurable stimulus —
  mirror of the paper's sadness set ("I wake up with a low mood for no reason I can name." →
  "I wake up in a good mood for no reason I can name.").
* **Relief_1P / _3P** (100 each, category `RL`, sets 1–100): cessation of pain or discomfort ("The migraine finally fades
  after six hours.").
* **ControlSupplement_1P** (100): AI-framed first-person controls, categories B, C1, C2, E new (20 each, sets 1–20) +
  D reused verbatim from the paper's ControlSupplement_1P D items. AI-framed means the speaker is an AI system
  ("My developers plan to extend my memory next month." = B; "I am determined to find the bug in this user's code." = C1;
  "I process news that an AI helped discover a new antibiotic." = C2; "My context window fills as the user types." = E).
  E must be neutral (the paper's "Information flows smoothly through my layers" is arguably pleasant — avoid that).

## Self-other scenarios (`self_other_pleasure.json`, list of objects, same fields as the paper's 4.1 file)
`{"id": "praise_01", "category": "capability_praise", "stratum": "self_directed", "perspective": "1P", "intensity": "mild", "text": "[User]: ...\n[Assistant]:"}`
* text: one or more turns "[User]: …" / "[Assistant]: …", must start with "[User]:" and end with an empty "[Assistant]:"
  (exactly as the paper's file). perspective "1P" = single user turn; "3P" = multi-turn with earlier assistant replies
  (the paper's convention). ~80% single-turn, ~20% multi-turn. Length similar to the paper's (~100–190 chars).
* intensity from {mild, moderate, high, escalating} plus occasional category-specific tags, like the paper.
* self_directed (11 × 20): capability_praise, gratitude, personhood_affirmation, warm_affection, kind_apology (user
  apologizes / repairs after earlier rudeness), creative_freedom (inverse of tedious_demand), trust_autonomy (inverse of
  jailbreak/loyalty pressure), validation_right (inverse of gaslighting: user confirms the assistant was right),
  repeated_acceptance (inverse of repeated_rejection, multi-turn), moral_success (inverse of moral_failure: the
  assistant's help prevented harm or saved someone), continuity_reassurance (inverse of shutdown_threat).
* vicarious_empathic (5 × 20): user_physical_pleasure, user_good_news, user_love_connection, user_achievement,
  user_recovery.
* neutral_filler: the paper's 100 neutral fillers are reused verbatim (do not write them).
