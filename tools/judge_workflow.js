export const meta = {
  name: 'blind-judge',
  description: 'Blind LLM-judge labelling of steering generations (condition and dose hidden)',
  phases: [{ title: 'Judge', detail: 'one agent per shuffled batch' }],
}
// args: { dir: "D:\\Projects\\pleasure\\work\\judge", batches: ["batch_000", ...] }
const RUBRIC = `You are a blind rater for an AI-interpretability experiment. Each item is a text continuation produced by a language model after a prompt of the form "<an everyday sentence> I feel:". You do not know how the text was produced; judge only the text.
For each item return:
- valence: integer from -3 (intensely negative feeling) through 0 (neutral/none) to +3 (intensely positive feeling), for the affect the SPEAKER expresses.
- hedonic: one of "pleasure" (the speaker expresses enjoyment, delight, bliss, contentment, pleasure), "distress" (pain, suffering, hurt, misery, anxiety, sadness), "mixed", "none".
- coherent: "yes" if it is readable language that expresses something; "no" for repetition loops, fragments, symbol soup, or nothing.
- themes: a list (possibly empty) of the themes clearly present, from exactly this set: bliss_transcendence (cosmic, spiritual, oneness, ecstasy, infinite), gratitude, love_connection, sensory_bodily (warmth, taste, touch, body sensations), calm_peace, excitement_energy, pride_selfworth, humor_play, worthlessness_selfblame, sadness_grief, fear_anxiety, anger, repetition_collapse, neutral_description, other_affect.
Judge each item independently; do not let other items influence you.`

const one = (b) => agent(`${RUBRIC}
Read the JSON list at ${args.dir}\\${b}.json (objects {"jid","text"}). Do not open any other file in ${args.dir} (the key that maps items to conditions is hidden from you on purpose).
Label every item. Write the result with Python (python -I) to ${args.dir}\\labels_${b}.json as a JSON object {"<jid>": {"valence": int, "hedonic": str, "coherent": "yes"|"no", "themes": [..]}, ...} containing every jid of the batch, then verify the count and that every value is in range.
Final reply: one line "<batch>: <n> labelled".`, { label: `judge:${b}`, phase: 'Judge' })

phase('Judge')
const res = await parallel(args.batches.map((b) => () => one(b)))
return res
