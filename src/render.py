from __future__ import annotations

from typing import Any


def render_turn(
    turn: dict[str, Any],
    include_speaker: bool = True,
    include_roles: bool = False,
) -> str:
    text = turn["text"].strip()
    speaker = turn.get("speaker", "A")
    role = turn.get("role", "unknown").upper()

    prefix = []
    if include_roles:
        prefix.append(f"[{role}]")
    if include_speaker:
        prefix.append(f"[{speaker}]")

    if prefix:
        return "".join(prefix) + f" {text}"
    return text


def render_context(
    sample: dict[str, Any],
    include_speaker: bool = True,
    include_roles: bool = False,
    render_mode: str = "full",
    local_k: int = 3,
) -> str:
    dialogue = sample["dialogue"]

    if render_mode == "candidate_only":
        # only keep the query turn
        dialogue = [dialogue[-1]]

    elif render_mode == "local_only":
        # keep only the last k turns
        dialogue = dialogue[-local_k:]

    elif render_mode == "full":
        pass

    else:
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    return "\n".join(
        render_turn(
            turn,
            include_speaker=include_speaker,
            include_roles=include_roles,
        )
        for turn in dialogue
    )


def render_for_ranking(
    sample: dict[str, Any],
    include_speaker: bool = True,
    include_roles: bool = False,
    render_mode: str = "full",
    local_k: int = 3,
) -> list[dict[str, Any]]:
    context = render_context(
        sample,
        include_speaker=include_speaker,
        include_roles=include_roles,
        render_mode=render_mode,
        local_k=local_k,
    )
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
                "metadata": sample.get("metadata", {}),
            }
        )
    return rows


def build_rows(
    samples: list[dict[str, Any]],
    include_speaker: bool = True,
    include_roles: bool = False,
    render_mode: str = "full",
    local_k: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.extend(
            render_for_ranking(
                sample,
                include_speaker=include_speaker,
                include_roles=include_roles,
                render_mode=render_mode,
                local_k=local_k,
            )
        )
    return rows