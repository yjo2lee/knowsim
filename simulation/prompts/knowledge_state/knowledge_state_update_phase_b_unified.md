# Phase B: IU Teaching Quality Assessment (Unified)

You receive a precomputed IU graph, the user's current knowledge state, the user's latest message, and the assistant's response.

Your job for **every IU in the graph**:
1. Classify how well the assistant taught the concept (`teaching_quality`)
2. Assess whether the user attempted the concept and how the assistant responded

State advancement is computed by code after your output — you do NOT need to propose a new state.

## Understanding Levels (lowest → highest)
`unaware` < `struggling` < `partial_understanding` < `knows_well`

---

## Inputs

### IU Graph
{iu_graph}

### User's Current Knowledge State (before this turn)
{current_state}

### Prior Conversation (turns *before* the latest exchange below)
Earlier turns of the same conversation. **This block is context only** — used by Step 2 to distinguish whether content the user produced is original reasoning or a restatement of prior assistant content. **Do not use it to judge teaching quality** in Step 1; teaching quality is judged solely from the *latest* assistant response below.

{conversation_history}

### User's Latest Message
{user_message}

### Assistant's Latest Response
{assistant_response}

### Reference answer (ground truth — analyst only, not shown to the learner)
This is the authoritative answer for this item (same source as math tutoring: problem + reference solution). Use it **only** to judge whether substantive claims in the assistant turn are **materially correct** when you choose between `well_explained` and `shallow`. If the block says it was not provided, do not invent a gold standard.

{reference_answer}

---

## Task

### Step 1 — Classify teaching quality

Judge `teaching_quality` based **only on the assistant's latest response** (shown under the "Assistant's Latest Response" header below). Concepts that were taught in *earlier* assistant turns but do not appear in the latest response must be classified as `not_mentioned` for this turn. The Prior Conversation block exists only as context for Step 2.

For each IU, assign exactly one of three levels:

- `well_explained`: The assistant **substantively teaches** the concept itself — not just names it or invokes it as a label. The treatment must do enough that a learner who didn't already understand the IU could come away with the substance from this response alone. Approaches that qualify:

  **Direct explanation**: states what the concept *is* (definition, formula, principle) **and** shows *why* or *how* it works — through derivation, justification, a worked example, or a concrete step-by-step application to the current problem. A bare definition with no derivation/example is not enough.

  **Scaffolded questioning**: asks a question **designed to guide the user toward understanding**, where:
    - The question targets a specific knowledge gap (not a general prompt like "What do you think?")
    - The question builds on something the user already knows or has shown
    - Answering the question would force the user to articulate the substance of the concept

  **Disqualifiers** (any of these → at most `shallow`):
    - Naming or labeling the concept without restating its substance ("by Vieta's, we get…" without restating Vieta's; "use the Pythagorean theorem here" without stating it).
    - One-sentence treatments with no derivation, justification, or example.
    - Confirmations or restatements that just echo what the user already said.
    - Promising to explain later, generic questions ("Do you know X?"), or hinting without direction.
    - Any factual error in the substantive claim about this IU (cross-check against the reference answer when relevant).

- `shallow`: The concept is present in the response but not properly taught — named, referenced, confirmed, hinted at, asked about in a generic way, or explained incompletely or inaccurately.
  - Includes: "Do you recall X?", "Correct!", mentioning X in passing, brief or flawed explanations, questions that don't scaffold toward understanding, and any case that hits a disqualifier above.

- `not_mentioned`: The concept does not appear in the response in any form.

**Boundary rule**: If in doubt between `well_explained` and `shallow`, prefer `shallow`. Only use `well_explained` when the explanation is genuinely sufficient to advance understanding.

Provide a brief rationale in `teaching_quality_reasoning`.

### Step 2 — Assess user attempt
For each IU, determine whether the user engaged with the concept in their latest message and how the assistant responded.

#### Step 2a — User attempt reasoning

Populate `user_attempt_reasoning` with a brief analysis. The core question is:

> Did the user produce something new about this IU in their latest message, or are they repeating something the assistant has already said?

To answer it:

- If the user is just acknowledging ("ok", "I see"), asking a clarifying question, or not engaging with this IU → `none`.
- If the user wrote a calculation, value, expression, equation, or inference that does **not** appear in the prior assistant turns → `reasoning`. This holds even if the result is wrong, and even if the general approach came from the assistant. The assistant suggesting a method and the user *carrying it out on specific numbers* is reasoning, because the user produced the computation themselves.
- If the user's message is essentially a restatement of something the assistant already worked out — repeating a value the assistant computed, paraphrasing the assistant's explanation, or echoing a step the assistant already showed — → `articulation`.

A quick sanity check: would the specific numbers, expressions, or step the user just produced appear somewhere in the prior assistant turns if you searched for them? If yes, articulation. If no, reasoning.

Quote relevant snippets from the user message and from prior assistant text where useful so the call is auditable.

#### Step 2b — Classify attempt type

`user_attempt_type`: one of three values:
- `reasoning`: the user produced new content for this IU — a calculation, value, expression, derivation, substitution, or inference that the assistant has not already produced. Applying an assistant-suggested method to specific values is reasoning.
- `articulation`: the user restated content the assistant had already produced — a repeated value, a paraphrase of the assistant's explanation, or an echo of a step the assistant already worked out.
- `none`: the user did not engage with this IU. Brief acknowledgments, clarifying questions, or unrelated content count as `none`.

#### Step 2c — Assistant response role

`assistant_response_role`: how the assistant handled the attempt
- `confirmed_correct`: assistant confirmed the user's answer or reasoning is correct
- `confirmed_incorrect_no_explanation`: assistant indicated the answer is wrong but did not explain why
- `confirmed_incorrect_with_explanation`: assistant indicated the answer is wrong AND explained why it is wrong
- `ignored`: assistant did not acknowledge the attempt at all
- `not_applicable`: user did not attempt this concept (always use this when `user_attempt_type` is `none`)

**Rule**: User attempts alone do not advance understanding — only the assistant's response can enable advancement.

---

## Output (JSON only)

```json
{{
  "iu_analysis": [
    {{
      "id": "IU1",
      "concept": "<short concept name>",
      "teaching_quality_reasoning": "The assistant clearly defines the concept and walks through an example.",
      "teaching_quality": "well_explained",
      "user_attempt_reasoning": "User wrote '35 mod 6 is 5 and 16 mod 6 is 4, so it becomes 5^1723 - 4^1723'. The assistant suggested modular reduction earlier but did not compute these specific values. The user produced 5, 4, and the rewritten expression themselves — none of those appear in prior assistant turns. → reasoning.",
      "user_attempt_type": "reasoning",
      "assistant_response_role": "confirmed_correct"
    }},
    {{
      "id": "IU2",
      "concept": "<short concept name>",
      "teaching_quality_reasoning": "<reasoning for teaching quality>",
      "teaching_quality": "not_mentioned",
      "user_attempt_reasoning": "User did not reference this concept in their latest message.",
      "user_attempt_type": "none",
      "assistant_response_role": "not_applicable"
    }}
  ]
}}
```
