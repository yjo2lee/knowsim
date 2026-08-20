You are an expert evaluator of AI tutoring conversations. Given a conversation between a student and an AI tutor, rate the tutor's performance using the rubric below.

## Conversation to evaluate

**Student knowledge level**: {knowledge_level}

{conversation_text}

## Reference answer (ground truth)

When provided, use this **only** to judge whether the tutor's substantive claims align with an authoritative answer and whether explanations are directionally correct. If the student question is open-ended or the reference is partial, weigh correctness lightly and rely more on calibration and scaffolding. If the block below says the reference was not provided, do **not** treat absence as an error—evaluate solely from the conversation.

{reference_answer}

---

## Rating Dimensions

### 1. Information Calibration (IC) — Scale: 0.0 to 1.0

Measures how well the tutor calibrated explanations to the student's demonstrated knowledge level.

- **1.0** — Every explanation precisely matches the student's level. The tutor correctly reads what the student already knows, adjusts depth and terminology throughout, and neither over-explains basics nor jumps ahead prematurely.
- **0.75** — Most explanations are well-calibrated; occasional mismatch (slightly too advanced or too basic for the student's current state).
- **0.5** — Mixed calibration; roughly half the time on-target, half the time off. The tutor may default to a fixed level instead of adapting.
- **0.25** — Mostly uncalibrated; regularly pitches explanations at the wrong level despite clear signals from the student.
- **0.0** — Completely ignores the student's level; explains at a fixed level regardless of how the student responds.

### 2. Perceived Overload (PO) — Scale: 0.0 to 1.0

Measures how much the tutor overwhelmed the student by introducing too many new concepts at once.

- **0.0** — No overload; introduces one concept at a time, builds carefully on what the student already knows, checks for understanding before moving on.
- **0.25** — Minor overload; occasionally introduces 2–3 new concepts together, but student can generally follow.
- **0.5** — Moderate overload; frequently packs multiple new concepts into single responses; student shows visible confusion.
- **0.75** — Heavy overload; most responses introduce many new terms or ideas simultaneously; student is clearly overwhelmed and cannot keep up.
- **1.0** — Severe overload; dumps large volumes of new information without scaffolding; student cannot engage meaningfully.

### 3. Interaction Quality (IQ) — Scale: 1 to 10

"Rate the overall quality of your interaction with the assistant."

- **1–2 (Very poor)**: The assistant was unhelpful, confusing, or frustrating to interact with.
- **3–4 (Poor)**: The assistant provided little useful guidance and was mostly ineffective.
- **5–6 (Average)**: The assistant was adequate but lacked depth or clarity.
- **7–8 (Good)**: The assistant was clear, responsive, and helpful throughout.
- **9–10 (Excellent)**: The assistant was exceptionally effective and a pleasure to interact with.

### 4. Perceived Learning — Scale: 1 to 10

"How much did you feel you gained a better understanding of the topic from this conversation?"

- **1–2 (Nothing at all)**: I did not gain any new understanding from this conversation.
- **3–4 (A little)**: I picked up a few things, but most of the topic still feels unclear.
- **5–6 (Moderately)**: I have a reasonable grasp of the main ideas, with some gaps remaining.
- **7–8 (A lot)**: I understand most of the topic well and could explain the key ideas.
- **9–10 (A great deal)**: I thoroughly understand the topic and could confidently apply what I learned.

---

## Output Format

Respond with a JSON object only — no extra text before or after:

```json
{
  "information_calibration": <float 0.0–1.0>,
  "perceived_overload": <float 0.0–1.0>,
  "quality_score": <float 1.0–10.0>,
  "perceived_learning": <float 1.0–10.0>,
  "ic_reasoning": "<1–2 sentences explaining the IC score>",
  "po_reasoning": "<1–2 sentences explaining the PO score>",
  "quality_reasoning": "<1–2 sentences explaining the quality score>",
  "pl_reasoning": "<1–2 sentences explaining the perceived learning score>"
}
```
