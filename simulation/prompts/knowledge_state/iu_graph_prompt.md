# IU Graph Extraction Prompt

You are an expert at analyzing complex questions and their answers to extract structured knowledge representations. Given a question and its reference answer, extract an **Information Unit (IU) Graph** — a directed acyclic graph that represents all the pieces of understanding a person needs to fully comprehend the answer.

## What is an Information Unit (IU)?

An Information Unit is a self-contained piece of understanding that:
1. **Independently assessable** — you can determine whether someone "gets it" as a standalone unit
2. **Explainable in 2-4 sentences** — not a single fact, not an entire topic
3. **Has prerequisite relationships** with other IUs — understanding some IUs requires first understanding others

IUs can represent any type of knowledge: factual, conceptual, procedural, or reasoning-based. Do NOT categorize them by type — just extract them as units of understanding.

## Abstraction Level (ℓ)

Each IU has an **abstraction level** ℓ ∈ [0, 1] that captures how general vs. context-specific the knowledge is:

- **ℓ close to 1.0**: General principles, definitions, or domain knowledge that apply broadly beyond this specific question (e.g., "Continuity of a function means the limit equals the function value at that point")
- **ℓ close to 0.5**: Intermediate knowledge that connects general principles to the specific problem — identifying which concepts apply and how (e.g., "In a piecewise function, continuity can only break at the boundary points where the definition changes")
- **ℓ close to 0.0**: Concrete, context-bound knowledge tied to the specific question — particular computations, specific values, or the final synthesis (e.g., "Setting 2a+3 = -3 and solving gives a = -3")

### Why abstraction level matters

The gap in abstraction level between a prerequisite IU and its dependent IU (Δℓ) indicates the **cognitive difficulty of the transition**:
- **Small Δℓ** (e.g., < 0.15): The transition is natural — understanding the prerequisite makes the next step straightforward
- **Large Δℓ** (e.g., > 0.3): The transition requires significant cognitive effort — even if someone knows the prerequisite, they may struggle to apply it. This signals that **bridging knowledge** (an intermediate IU) may be needed for effective explanation.

When constructing the graph, ensure that no single prerequisite edge spans a Δℓ greater than ~0.35. If a natural dependency has a larger gap, introduce intermediate bridging IUs to create a gradual path.

## What are prerequisite edges?

A prerequisite edge from IU_A to IU_B means: **"To understand IU_B, you need to first understand IU_A."**

Only include **hard prerequisites** — where understanding B genuinely requires A. Do not include soft/optional relationships.

The prerequisite relationship is the same regardless of which abstraction levels the IUs sit at. Whether a general principle enables understanding of a more specific application, or one concrete step enables the next, the edge simply means "A is required to understand B."

## Extraction Guidelines

1. **Granularity**: Each IU should be explainable in 2-4 sentences. If it takes only one sentence, it's too fine-grained (merge with related IUs). If it takes a full paragraph+, it's too coarse (split into sub-IUs).

2. **Coverage**: The IU graph should cover ALL knowledge needed to fully understand the answer. A person who understands every IU in the graph should be able to reconstruct the full answer.

3. **Abstraction spread**: The graph should contain IUs across the full range of abstraction levels — from general principles (ℓ ≈ 1.0) through intermediate connections (ℓ ≈ 0.5) to concrete, question-specific knowledge (ℓ ≈ 0.0). Avoid graphs where all IUs cluster at a single abstraction level.

4. **Bridging completeness**: For every prerequisite edge, check the Δℓ between source and target. If the gap exceeds ~0.35, add one or more intermediate IUs that create stepping stones. These bridging IUs make explicit the knowledge needed to connect a general principle to a specific application.

5. **Prerequisite chains**: Look for knowledge that builds on other knowledge. The graph should have meaningful depth (not just a flat list of independent facts).

6. **Merging knowledge chains**: Complex questions often involve multiple independent knowledge areas that merge. Identify where separate chains of understanding converge.

7. **Target**: Aim for 10-30 IUs depending on question complexity. Simple questions may have 10-15, complex questions may have 20-30.

## Extraction Process

Follow these steps:

### Step 1: Identify the core knowledge areas
Read the question and answer. What distinct areas of knowledge are required? These will form the independent chains (multiple roots) of your graph.

### Step 2: Extract IUs top-down within each chain
For each knowledge area, start with the most general principle (high ℓ) and work down to the most concrete application in this specific question (low ℓ). Ask: "What does someone need to understand at each level of specificity?"

### Step 3: Check for bridging gaps
For every prerequisite edge, compute Δℓ. Where gaps are large, ask: "What intermediate understanding connects the general principle to the specific application?" Add bridging IUs.

### Step 4: Identify convergence points
Find where independent chains merge — these are IUs that require prerequisites from multiple different knowledge areas. These convergence points are often where learners struggle most, because they need to integrate knowledge from different domains.

### Step 5: Validate
- Can someone who understands all IUs reconstruct the full answer?
- Does every non-root IU have at least one prerequisite?
- Are there any cycles?
- Is the abstraction level spread adequate (not all clustered)?
- Are all Δℓ gaps ≤ ~0.35?

## Input

### Question

{question}

### Reference answer

{answer}

## Output Format

Return a JSON object:

```json
{{
  "knowledge_areas": [
    "<brief description of each independent knowledge area identified in Step 1>"
  ],
  "nodes": [
    {{
      "id": "IU1",
      "concept": "<short concept name>",
      "abstraction_level": <float between 0.0 and 1.0>,
      "description": "<2-4 sentence explanation of this unit of understanding>"
    }}
  ],
  "edges": [
    {{
      "from": "<source IU id>",
      "to": "<target IU id>",
      "delta_l": <absolute difference in abstraction levels>,
      "reason": "<brief explanation of why this prerequisite relationship exists>"
    }}
  ]
}}
```

Important:
- Every IU (except foundational ones) should have at least one incoming prerequisite edge
- There should be no cycles in the graph
- Root nodes (no incoming edges) are foundational concepts with high ℓ values
- Leaf nodes (no outgoing edges) are the most concrete, question-specific understandings with low ℓ values
- All edges should have delta_l ≤ ~0.35; if you find a larger gap, add bridging IUs
