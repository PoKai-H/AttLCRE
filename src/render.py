from __future__ import annotations

from typing import Any

def render_context(sample: dict[str, Any], include_roles: bool = False) -> str:
    """
    Convert structured dialogue into plain text context.
    """
    lines: list[str] = []

    for turn in sample["dialogue"]:
        speaker = turn["speaker"]
        text = turn["text"].strip()

        if include_roles:
            role = turn.get("role", "unknown").upper()
            lines.append(f"[{role}][{speaker}] {text}")
        else:
            lines.append(f"[{speaker}] {text}")

    return "\n".join(lines)

def render_for_ranking(
    sample: dict[str, Any],
    include_roles: bool = False,
) -> list[dict[str, Any]]:
    """
    Expand one structured sample into multiple candidate-level rows.
    """
    context = render_context(sample, include_roles=include_roles)
    correct_index = sample["correct_index"]

    rows: list[dict[str, Any]] = []

    for candidate_index, candidate_text in enumerate(sample["candidates"]):
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "context": context,
                "candidate": candidate_text,
                "label": 1 if candidate_index == correct_index else 0,
                "candidate_index": candidate_index,
                "gold_index": correct_index,
                "metadata": sample.get("metadata", {})
            }
        )
    return rows

def build_rows(
    samples: list[dict[str, Any]],
    include_roles: bool = False,
) -> list [dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.extend(render_for_ranking(sample, include_roles=include_roles))
    return rows
