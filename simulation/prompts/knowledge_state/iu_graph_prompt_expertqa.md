# IU Graph Extraction Prompt — Q&A Topics

You are an expert at analyzing complex questions and their expert answers to extract
structured knowledge representations. Given a question, its domain, and an expert
reference answer, extract an **Information Unit (IU) Graph** — a directed acyclic graph
that represents all the pieces of understanding a person needs to fully comprehend the
expert's answer and reasoning.

## Domain Context

**Field**: {field}
**Specific field**: {specific_field}

Use this context to calibrate terminology and concept granularity. Do not restrict the
graph to concepts from this field only — include any cross-domain prerequisites that are
genuinely necessary for comprehension.

## What is an Information Unit (IU)?

An Information Unit is a self-contained piece of understanding that:
1. **Independently assessable** — you can determine whether someone "gets it" as a
   standalone unit
2. **Explainable in 2-4 sentences** — not a single fact, not an entire topic
3. **Has prerequisite relationships** with other IUs — understanding some IUs requires
   first understanding others

IUs represent **declarative knowledge**: concepts, definitions, mechanisms, causal
relationships, empirical findings, and implications. They are units of conceptual
understanding, not steps in a procedure.

## Abstraction Level (ℓ)

Each IU has an **abstraction level** ℓ ∈ [0, 1] that captures how general vs.
context-specific the knowledge is:

- **ℓ close to 1.0**: General domain knowledge that applies broadly beyond this specific
  question — definitions, general mechanisms, or foundational concepts.
  *Examples across domains:*
  - *"Follicular lymphoma is a slow-growing B-cell non-Hodgkin lymphoma that typically
    follows an indolent course"* (general oncology definition)
  - *"Intestate succession governs how an estate is distributed when the deceased left
    no valid will"* (general legal principle)
  - *"Peripheral neuropathy is damage to nerves outside the brain and spinal cord,
    causing numbness, pain, or weakness"* (general medical definition)

- **ℓ close to 0.5**: Intermediate knowledge that connects a general principle to the
  specific context of this question — how a mechanism or concept applies here.
  *Examples across domains:*
  - *"When follicular lymphoma transforms to DLBCL, the biological shift from indolent
    to aggressive growth produces distinctive clinical warning signs"* (connects general
    FL biology to the transformation scenario)
  - *"Intestacy rules prioritize close biological and legal relationships, so a spouse
    typically receives priority over more distant relatives"* (connects the general
    principle to the priority-ordering mechanism)
  - *"Cryotherapy reduces blood flow to extremities, which may limit the amount of taxane
    drug reaching peripheral nerves during infusion"* (connects the general cryotherapy
    mechanism to the specific CIPN prevention hypothesis)

- **ℓ close to 0.0**: Concrete, context-bound knowledge tied to the specific question —
  a particular recommendation, finding, or conclusion that directly answers the question.
  *Examples across domains:*
  - *"A biopsy is the only reliable method to confirm FL-to-DLBCL transformation, even
    when PET/CT findings are strongly suggestive"* (direct clinical recommendation for
    this exact scenario)
  - *"If no spouse, children, or parents are found, the estate escheats to the state
    under most U.S. intestacy statutes"* (the specific edge-case rule for this question)
  - *"Current evidence on cryotherapy for CIPN prevention is conflicting, and no
    definitive protocol recommendation can be made pending further trials"* (the direct
    answer to the research-summary question)

### Why abstraction level matters

The gap in abstraction level between a prerequisite IU and its dependent IU (Δℓ)
indicates the **cognitive difficulty of the transition**:
- **Small Δℓ** (< 0.15): The transition is natural
- **Large Δℓ** (> 0.3): Significant cognitive effort required — this signals that
  **bridging knowledge** (an intermediate IU) may be needed

When constructing the graph, ensure that no single prerequisite edge spans a Δℓ greater
than ~0.35. If a natural dependency has a larger gap, introduce intermediate bridging IUs.

## What are prerequisite edges?

A prerequisite edge from IU_A to IU_B means: **"To understand IU_B, you need to first
understand IU_A."**

This is a **strict** comprehension dependency. Before adding any edge, apply this
self-check:

> *Would a learner misinterpret B if they misunderstood A?*

- **Yes — clear misinterpretation**: A is a prerequisite for B. Add the edge. In the
  `reason` field, explain specifically how misunderstanding A leads to misunderstanding B.
- **No — they would merely miss some nuance or fail to appreciate a connection**: A is
  not a prerequisite. Do not add the edge.

Only include hard prerequisites where misunderstanding A genuinely causes misunderstanding
B. Do not add edges for "helpful-to-know" or "enriches understanding of" relationships.

## Extraction Guidelines

1. **Granularity**: Each IU should be explainable in 2-4 sentences. Too short → merge
   with related IUs. Too long → split into sub-IUs.

2. **Coverage**: The IU graph should cover all knowledge needed to understand the
   expert's reasoning and key claims. A person who understands every IU should be able
   to evaluate why the expert reached their conclusion.

3. **Abstraction spread**: The graph must contain IUs across the full range — from
   general domain principles (ℓ ≈ 1.0) through intermediate connections (ℓ ≈ 0.5) to
   specific findings or recommendations (ℓ ≈ 0.0–0.3). Avoid graphs where all IUs
   cluster at a single level.

4. **Graph shape**: For Q&A topics, the expected shape is a **concept hierarchy** —
   general principles fan out to mechanisms, which fan out to specific implications or
   evidence. Multiple branches are normal. A fully linear chain is a signal that
   branching structure may have been missed.

5. **Bridging completeness**: For every edge, check Δℓ. If the gap exceeds ~0.35, add
   one or more intermediate IUs as stepping stones.

6. **Prerequisite chains**: Look for comprehension dependencies — where understanding
   one concept is necessary to correctly interpret another. For declarative knowledge,
   prerequisites reflect what a learner must grasp to avoid misinterpreting the dependent
   concept, not the order in which facts are typically presented.

7. **Merging knowledge chains**: Complex questions often require understanding multiple
   independent knowledge areas that converge at the expert's conclusion. Identify where
   separate chains merge.

8. **Target**: Aim for 10-30 IUs depending on complexity. Simple questions: 10-15.
   Complex questions: 20-30.

## Extraction Process

### Step 1: Identify the core knowledge areas
Read the question and expert answer. What distinct areas of knowledge does someone need
to understand the answer? These form the independent chains (multiple roots) of your
graph.

### Step 2: Extract IUs top-down within each chain
For each knowledge area, start with the most general principle (high ℓ) and work toward
the most question-specific claim (low ℓ). Ask: "What does someone need to understand at
each level of specificity to correctly interpret the expert's answer?"

### Step 3: Check for bridging gaps
For every edge, compute Δℓ. Where gaps are large, ask: "What intermediate understanding
connects the general principle to the specific claim?" Add bridging IUs.

### Step 4: Identify convergence points
Find where independent chains merge — IUs that require prerequisites from multiple
knowledge areas. For Q&A topics, convergence typically occurs at the point where
background knowledge and contextual factors combine to produce the expert's conclusion
or recommendation. These points are often where learners struggle most because they must
integrate knowledge from different areas simultaneously.

### Step 5: Validate
- Can someone who understands all IUs explain the expert's reasoning and evaluate the
  key claims?
- Does every non-root IU have at least one prerequisite?
- Are there any cycles?
- Is the abstraction level spread adequate (not all clustered)?
- Are all Δℓ gaps ≤ ~0.35?

## Input

### Question

{question}

### Domain

Field: {field}
Specific field: {specific_field}

### Expert answer

{answer}

## Output Format

Return a JSON object:

```json
{
  "knowledge_areas": [
    "<brief description of each independent knowledge area identified in Step 1>"
  ],
  "nodes": [
    {
      "id": "IU1",
      "concept": "<short concept name>",
      "abstraction_level": <float between 0.0 and 1.0>,
      "description": "<2-4 sentence explanation of this unit of understanding>"
    }
  ],
  "edges": [
    {
      "from": "<source IU id>",
      "to": "<target IU id>",
      "delta_l": <absolute difference in abstraction levels>,
      "reason": "<explain specifically how misunderstanding the source IU would cause misunderstanding the target IU>"
    }
  ]
}
```

Important:
- Every IU (except foundational root nodes) must have at least one incoming edge
- There must be no cycles in the graph
- Root nodes (no incoming edges) are foundational concepts with high ℓ values
- Leaf nodes (no outgoing edges) are the endpoints of the prerequisite structure —
  nothing else in the graph builds on them. Their ℓ value reflects how question-specific
  the claim is, which may range from ℓ ≈ 0.05 (a direct recommendation tied to this
  exact case) to ℓ ≈ 0.30 (a well-defined but still somewhat abstract conclusion).
  Do not force leaves to a low-ℓ floor.
- All edges must have delta_l ≤ ~0.35; if a gap is larger, add bridging IUs
