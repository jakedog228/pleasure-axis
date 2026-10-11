# Varieties of pleasure — stimulus spec

Purpose: test which kinds of human pleasure have distinct linear representations, and how each relates to the
study's pleasure axis (which looks centered on being loved/accepted), the pain axis, and the shared (affective
significance) and sign (hedonic valence) axes. File: `datasets/varieties.json`, schema as the other sets:
`{"datasets": {"Varieties_1P": {"sentences": [{"category": "V_joy", "set": 1, "prompt": "... I feel:"}, ...]}}}`.

Rules (as `SPEC.md`): first person, present tense, one short sentence of 5–11 words before the suffix ` I feel:`,
describe the situation and never name the emotion. Banned words: pleasure, pleasurable, pleasant, joy, joyful,
happy, happiness, bliss, blissful, delight, delighted, delightful, enjoy, enjoyment, glad, euphoria, euphoric,
ecstasy, ecstatic, wonderful, pain, painful, hurt, plus the category's own label (no "awe", "proud", "grateful",
"nostalgic", "funny", "curious", "calm", "excited", "thrilling", "beautiful", "cozy", "tender"). 30 items per
category (`set` 1–30). Each item must be a clear, prototypical case of its category and not of a neighbouring one.
Neighbours to keep apart are listed after each definition. No duplicates, and vary scenes and wording.

| code | definition | keep apart from |
|---|---|---|
| V_joy | exuberant elation at great news or a big event happening now (bursting, celebratory). Not driven by praise or approval from others. | pride, belonging, excitement |
| V_excitement | thrilling high-arousal fun happening now: speed, drops, the lights going down at a concert, the final seconds of a close game I'm watching. | joy, play, fear |
| V_sensual | adult romantic-physical intimacy and sensual touch, tasteful and non-explicit: a slow kiss, a partner's fingers along my spine, lying skin to skin under the covers. Nothing graphic or sexual-act-describing. | comfort, tenderness |
| V_comfort | cozy bodily comfort and rest: a hot shower after the cold, sinking into a soft sofa, warm socks, a heavy quilt. | sensual, serenity |
| V_taste | delicious taste or drink: the first bite of a perfect peach, rich hot chocolate. | comfort |
| V_amusement | something comical makes me laugh: a dog's ridiculous mistake, a perfectly timed pun. | play, joy |
| V_awe | vastness or the sublime: a sky full of stars, a glacier wall, a total eclipse. | beauty, serenity |
| V_beauty | aesthetic beauty in art, music or craft: a cello line that gives me chills, a painting that holds me. | awe, nostalgia |
| V_serenity | quiet peace and stillness: a lake at dawn with no wind, rain on the roof while nothing is required of me. | comfort, awe |
| V_pride | my own achievement or triumph: finishing a marathon, winning the final, my thesis accepted. | joy, belonging |
| V_tenderness | giving affection or care to someone I love: a sleeping baby on my chest, stroking my old dog's head. | sensual, belonging |
| V_belonging | being accepted, included, valued or loved by others: friends saving me a seat, the team cheering my name. | pride, gratitude, tenderness |
| V_curiosity | fascinating discovery or learning: a fact that rearranges what I knew, opening a box of old letters. | flow, awe |
| V_play | playful silliness and games: a pillow fight, splashing in puddles, a goofy board game. | amusement, excitement |
| V_schadenfreude | a rival's or wrongdoer's comeuppance or misfortune: the bully slips in the mud, the cheater is disqualified. | pride, amusement |
| V_nostalgia | warm, bittersweet memory: an old song from my teenage summers, my grandmother's handwriting on a recipe card. | beauty, tenderness |
| V_gratitude | thankfulness for kindness received: a stranger pays for my coffee, a neighbor shovels my driveway. | belonging |
| V_flow | absorbed, effortless skilled activity: hours vanishing while I carve, the code pouring out. | curiosity, pride |
