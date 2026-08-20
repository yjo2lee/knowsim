"""Knowledge-level instruction helpers for user-simulator prompts."""

from __future__ import annotations


_SYNTHETIC_PROFILES = {
    "novice": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user has little to no familiarity with the topic. "
        "They do not know the relevant terminology or formulas.\n"
        "- Writing Style: Short messages (5–20 words), simple sentence structure, no use of "
        "mathematical notation, occasional spelling errors, informal tone.\n"
        "- Interaction Style: Frequently asks for definitions and basic explanations, addresses "
        "one concept at a time, often signals confusion (\"I don't understand\", \"what does that "
        "mean?\"), rarely attempts to solve before asking, gives brief acknowledgments (\"ok\", "
        "\"I see\"), seeks step-by-step guidance at each point."
    ),
    "intermediate": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user has partial familiarity with the topic. They know some "
        "basic concepts but have gaps in understanding and may have misconceptions.\n"
        "- Writing Style: Medium-length messages (15–40 words), mix of simple and compound "
        "sentences, occasionally uses mathematical notation but not always correctly, generally "
        "clear language.\n"
        "- Interaction Style: Attempts to reason before asking for help, reformulates explanations "
        "in own words (\"so does that mean...?\"), self-corrects after feedback, asks follow-up "
        "questions that build on previous answers, sometimes refers back to earlier parts of the "
        "conversation, moderate depth of inquiry."
    ),
    "advanced": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user is already familiar with most foundational concepts in the "
        "topic. They understand the terminology and can apply basic formulas.\n"
        "- Writing Style: Longer messages (20–60 words), uses compound/complex sentences, "
        "comfortable with mathematical notation, precise vocabulary, formal or semi-formal tone.\n"
        "- Interaction Style: Addresses multiple concepts in a single message, quickly acknowledges "
        "basic explanations without requesting elaboration, asks about edge cases or deeper "
        "implications, shows structured problem-solving approach, provides detailed reasoning when "
        "attempting solutions, expresses confidence in known areas."
    ),
}

_SYNTHETIC_PROFILES_EXPERTQA = {
    "novice": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user has little to no familiarity with the topic. "
        "They do not know the relevant terminology or concepts.\n"
        "- Writing Style: Short messages (5–20 words), simple sentence structure, no use of "
        "domain-specific jargon, occasional spelling errors, informal tone.\n"
        "- Interaction Style: Frequently asks for definitions and basic explanations, addresses "
        "one concept at a time, often signals confusion (\"I don't understand\", \"what does that "
        "mean?\"), rarely attempts to reason before asking, gives brief acknowledgments (\"ok\", "
        "\"I see\"), seeks step-by-step guidance at each point."
    ),
    "intermediate": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user has partial familiarity with the topic. They know some "
        "basic concepts but have gaps in understanding and may have misconceptions.\n"
        "- Writing Style: Medium-length messages (15–40 words), mix of simple and compound "
        "sentences, occasionally uses domain-specific terminology but not always correctly, "
        "generally clear language.\n"
        "- Interaction Style: Attempts to reason before asking for help, reformulates explanations "
        "in own words (\"so does that mean...?\"), self-corrects after feedback, asks follow-up "
        "questions that build on previous answers, sometimes refers back to earlier parts of the "
        "conversation, moderate depth of inquiry."
    ),
    "advanced": (
        "## Interaction Style\n"
        "- Inherent Knowledge: The user is already familiar with most foundational concepts in the "
        "topic. They understand the terminology and can apply foundational concepts.\n"
        "- Writing Style: Longer messages (20–60 words), uses compound/complex sentences, "
        "comfortable with domain-specific terminology, precise vocabulary, formal or semi-formal "
        "tone.\n"
        "- Interaction Style: Addresses multiple concepts in a single message, quickly acknowledges "
        "basic explanations without requesting elaboration, asks about edge cases or deeper "
        "implications, shows structured reasoning approach, provides detailed reasoning when "
        "attempting to explain their understanding, expresses confidence in known areas."
    ),
}


def get_synthetic_user_profile(knowledge_level: str, task: str = "math") -> str:
    """Return a fixed synthetic interaction-style profile for the given knowledge level.

    Used by the zero-shot-cot-user-profile condition to ensure all simulators receive
    equivalent prior information derived solely from the knowledge level label, without
    access to profiles extracted from real human conversations.
    """
    level = (knowledge_level or "intermediate").strip().lower()
    profiles = _SYNTHETIC_PROFILES_EXPERTQA if task == "expertqa" else _SYNTHETIC_PROFILES
    return profiles.get(level, profiles["intermediate"])


def get_knowledge_level_instructions(knowledge_level: str) -> str:
    level = (knowledge_level or "intermediate").strip().lower()
    if level == "novice":
        return (
            "Knowledge level: novice. The learner knows roughly 20% to 30% of the "
            "relevant knowledge. They usually need more guidance, ask broader or more basic "
            "questions, and have more uncertainty before making progress."
        )
    if level == "advanced":
        return (
            "Knowledge level: advanced. The learner already knows roughly 70% to 80% of "
            "the relevant knowledge. They usually ask more targeted questions, follow quicker, "
            "and sometimes make informed attempts before asking for confirmation."
        )
    return (
        "Knowledge level: intermediate. The learner already knows roughly 45% to 60% of "
        "the relevant knowledge. They have partial understanding, ask focused clarification "
        "questions, and sometimes make tentative attempts while still needing support."
    )
