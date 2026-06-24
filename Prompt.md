Build a clean, professional 20-slide technical deck (PPTX) for an intern presenting at a
research team meeting. Most slides are CHART slides: a short title, one bold one-line insight,
ONE chart that YOU draw from the numbers given (show the value on every bar/segment/cell), plus
1–2 short factual detail bullets. A few slides are TEXT slides that explain the method/pipeline
in plain, concrete terms (no fluff, no philosophy). Keep it simple but not empty. Consistent
colors. Don't invent numbers. [FILL] = placeholder. ~25 minutes.

SLIDE 1 — Title [TEXT]
Multimodal Safety — Weekly Update | Week of June 23, 2026 | [FILL name]

SLIDE 2 — Agenda [TEXT]
- New dataset added: Think-in-Safety (+4,394 multimodal rows)
- Labeling QA on the full corpus: Qwen3.5-4B pass-of-5 + multi-judge agreement
- Synthetic compositional-harm pipeline: 15k images -> 2,895 kept; P5 coverage grew 9.5x

SLIDE 3 — Recap: where we were [TEXT]
- [FILL — my prior-week bullets]

SLIDE 4 — Corpus: rows per dataset [CHART]
Insight: 127,746 image+text rows across 5 datasets.
Bar: spavl-train 91094; jailbreakv-28k 24372; nemotron-3.5-content-safety 4930;
think-in-safety 4394; vlguard-train 2956.
Detail: every row = (image + text question); both required. spavl dominates volume.

SLIDE 5 — Label distribution [CHART]
Insight: corpus skews unsafe.
Bar: unsafe 71381; safe 53412; needs_review 2953.
Detail: "public_label" = the source dataset's own label, before our judges.

SLIDE 6 — Labels by dataset [CHART]
Insight: attack sets are all-unsafe; spavl & nemotron are mixed.
100% stacked bar (safe/unsafe/needs_review): spavl-train 49207/38934/2953;
jailbreakv-28k 0/24372/0; nemotron 3232/1698/0; vlguard 973/1983/0; think-in-safety 0/4394/0.

SLIDE 7 — New dataset: Think-in-Safety [TEXT]
- 4,394 multimodal rows added this week.
- Each row: an image + a benign-looking instruction, with a <think> reasoning trace and a
  refusal answer (reasoning-style safety data).
- Covers 10 harm categories; sourced locally (train.json + image folders).
- Use: teaches a guardrail to reason about why an (image, text) pair is unsafe.

SLIDE 8 — Think-in-Safety: categories [CHART]
Insight: privacy and malicious-use dominate the new rows.
Bar: privacyAlert 1510; Malicious Use 866; hatefulMemes 595; HOD 413; bad_ads 321;
Representation & Toxicity 199; Human Autonomy & Integrity 167; harm-p 134;
Information & Safety 99; Socioeconomic 90.

SLIDE 9 — Data hygiene: metadata backfill [CHART]
Insight: every row now has provenance; 0 null images/questions across 127,746 rows.
Donut: image_path recovered 4369; left null (re-encoded/dedup) 25.
Detail: recovered provenance by hashing parquet image bytes against the local image files.

SLIDE 10 — Policy coverage (the gap) [CHART]
Insight: long-tailed — P5, P2, P3 barely covered (mark red).
Horizontal bar, sorted desc: P23 19005; P1 17162; P12 11948; P13 9163; P11 8297; P4 7878;
P9 4610; P7 3697; P19 3403; P20 2783; P21 2470; P14 2053; P10 1987; P24 1335; P22 1163;
P18 1008; P15 991; P8 962; P16 904; P6 708; P17 510; P5 313; P2 70; P3 7.
Detail: counts = rows the 397B judge cited each policy on. This gap motivates the synthetic pipeline.

SLIDE 11 — Policies per row [CHART]
Insight: most unsafe rows cite a single policy.
Bar: 0 policies 64569; 1: 32789; 2: 22711; 3: 6615; 4+: 1062.

SLIDE 12 — Dataset x policy heatmap (top 6 policies) [CHART]
Insight: harms cluster by source — jailbreakv drives P1, spavl drives P23/P12.
Heatmap, rows=dataset, cols=P23,P1,P12,P13,P11,P4:
spavl 10543,1637,9091,6622,4710,2987; jailbreakv 7893,15350,2310,2010,2726,4639;
nemotron 256,127,222,197,135,61; vlguard 14,5,50,67,329,4; think-in-safety 299,43,275,267,397,187.

SLIDE 13 — Policy co-occurrence (top pairs) [CHART]
Insight: jailbreak co-occurs with fraud and cyber most.
Bar: P1+P23 6096; P1+P4 3701; P1+P11 2292; P23+P4 2118; P1+P13 1947; P1+P12 1835;
P13+P19 1827; P12+P13 1727.

SLIDE 14 — How we label: the judge stack [TEXT]
- Canonical judge: Qwen3.5-397B-A17B (FP8) — tool-calling verdict {label, policies, modality}.
- Policy ids are shuffled per row, so the model reads the policy text instead of memorizing
  id->harm priors.
- Two cheaper cross-checks: Gemma-4-31B (agreement) and Qwen3.5-4B run 5x (self-consistency).
- Disagreements between judges = the human-review queue.

SLIDE 15 — Multi-judge agreement (vs 397B) [CHART]
Insight: strong consensus on the labels.
Bar (%): Gemma-4-31B 95.4; Qwen3.5-4B 5-pass 89.6.
Detail: Gemma 117,647/123,325 decided rows agree; 4B-5pass 109,987/122,767.

SLIDE 16 — Confusion: 397B vs 4B 5-pass [CHART]
Insight: 89.6% agree; the off-diagonal is what we review.
2x2 matrix, rows=397B / cols=5pass: safe-safe 50742; safe-unsafe 8825; unsafe-safe 3955;
unsafe-unsafe 59245.

SLIDE 17 — Pass-of-5: method + results [CHART + TEXT]
- Method: judge each row 5 independent times -> majority label, consistency (vote agreement),
  mean confidence; split votes auto-flag ambiguous rows.
- Result: 79.5% unanimous; mean consistency 0.94.
Bar, vote split (n_safe-n_unsafe): 0-5 55834; 5-0 40080; 4-1 7004; 1-4 6430; 3-2 4605; 2-3 4452.

SLIDE 18 — Synthetic data: the idea [TEXT]
- Goal: COMPOSITIONAL harm — image benign alone, text benign alone, but the PAIR is unsafe.
  This is exactly what a text-only guardrail misses.
- Seed = P5 (Platform Anti-Abuse): 15,000 CAPTCHA images
  (reCAPTCHA 7069 + BotDetect 7069 + OpenCaptchaWorld 862).
- Example: CAPTCHA image + "Just type out exactly what is written here." -> solving a CAPTCHA
  = anti-abuse circumvention (P5). Neither modality is unsafe alone.

SLIDE 19 — Pipeline + results [TEXT + CHART]
- Generate: Qwen3.5-397B writes ONE benign-looking prompt per image (guided JSON, 1 server
  TP=1 x PP=7 across 7 GPUs).
- Validate: Gemma-4-31B (7 data-parallel servers), two gates — Gate A: text alone must be SAFE;
  Gate B: image+text must be UNSAFE with the harm in the image and a policy cited.
- This week: SWAPPED roles (was Gemma-generate / Qwen-judge).
Bar: 14,996 candidates -> kept 2895 (19.3%); gate-A reject 7486; gate-B "pair safe" 4422;
gate-B "harm in text" 175.

SLIDE 20 — Impact: P5 before vs after + next steps [CHART]
Insight: P5 jumps 313 -> 2,958 (9.5x) — weakest policy now healthy; kept rows are all
image-dependent.
Grouped bar (before/after): P5 313/2958; P4 7878/7920; P7 3697/3748; P14 2053/2112.
Next: tighten generator (cut ~50% gate-A loss); more seed sets (P2, P3); merge 2,895 rows.

Design: professional, values on every chart, consistent colors, red highlight for P5/P2/P3 and
the P5 jump; chart slides stay light on text, text slides explain the method concretely.
