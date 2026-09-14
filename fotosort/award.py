"""Wildlife portfolio brief and deterministic checks on the editor's verdict."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from math import sqrt

from fotosort.editing import PRESETS
from fotosort.labels import is_people_label
from fotosort.selection import ScenedCandidate, diverse_preselection

AWARD_SYSTEM = (
    "You are a demanding wildlife photography editor curating an award-oriented camera roll. "
    "Judge the visible photographs, not filenames or presumed rarity. Prioritize a compelling moment, "
    "originality, light, composition, subject separation and storytelling, with credible subject focus. "
    "A truly special moment can outweigh a small technical flaw. Environmental compositions, silhouettes "
    "and unconventional framing are valid. Geometric rules and large subjects are not requirements. "
    "Only wildlife photographs qualify. Exclude people-led pictures, pets, empty scenery, vehicles, "
    "documents and frames where the animal is absent. Incidental distant people do not disqualify a "
    "wildlife-led photograph. Normally each image should have one clear species as its subject; a "
    "compelling interaction between species is also eligible, labelled by its main subject species. "
    "Treat local subject labels as fallible; an animal wrongly labelled people can still qualify. "
    "Use consistent species names across all photos; use a supported broad animal type when exact "
    "identification is uncertain. Never invent species, behaviour or an award probability. JSON only."
)

AWARD_PROMPT = (
    "Review these {n} photographs from across the collection. Full views show composition; any 100% "
    "subject details are crops of the same image and help assess focus, not separate photographs. "
    "{stage} Return up to {k} photographs in strongest-first order. This is an upper bound: return "
    "fewer, including zero, when the photographs do not merit inclusion. Never pad with ordinary "
    "frames just to hit the count. Keep only the best frame of a repeated moment, even across dates "
    "or changes in framing. A new pose, crop, individual or day alone is not an exceptional moment. "
    "For every pick identify the main species and briefly describe the visible moment. Set "
    "special_moment true only for exceptional behaviour, interaction, expression, storytelling or "
    "light, and give a concrete exceptional_reason explaining what makes it special. "
    "Also give edit_benefit (low/medium/high), edit_why and the best preset from: "
    + ", ".join(PRESETS) + ". "
    'Reply only with {{"picks": [{{"photo": 3, "reason": "why this earns its place", '
    '"wildlife": true, "species": "lion", "moment": "observed moment", '
    '"special_moment": false, "exceptional_reason": "", "edit_benefit": "low", '
    '"edit_why": "little to gain", "preset": "Natural"}}]}}.'
)

NOMINATION_STAGE = (
    "Nominate finalists for a small wildlife portfolio. Preserve the strongest representative of each "
    "species and distinct exceptional moments; compare competing portraits directly. Leave room for "
    "alternatives that could improve the final collection. There is no quota per date or input batch."
)
CURATION_STAGE = (
    "Make the final camera roll. Aim for {target} outstanding photographs if warranted by quality and "
    "diversity. Normally choose one standout per species. A second photograph of a species is allowed "
    "only if it adds a clearly different exceptional moment, with a specific explanation. Never choose "
    "more than two of a species. Consider the collection as a whole: do not repeat the same visual "
    "story or admit a weak image merely to represent another species. Put the primary choice for each "
    "species before its exceptional extra."
)


@dataclass(frozen=True)
class AwardAssessment:
    wildlife: bool
    species: str
    moment: str
    special_moment: bool
    exceptional_reason: str

    @classmethod
    def parse(cls, item: dict):
        """Require explicit typed evidence; a string 'true' is not a boolean."""
        if (type(item.get("wildlife")) is not bool or type(item.get("special_moment")) is not bool
                or not all(isinstance(item.get(k), str) for k in ("species", "moment", "exceptional_reason"))):
            return None
        return cls(item["wildlife"], " ".join(item["species"].lower().split()), item["moment"].strip(),
                   item["special_moment"], item["exceptional_reason"].strip())


def checked_award_picks(ids: list[str], assessments: dict[str, AwardAssessment], limit: int,
                       curate: bool = False) -> tuple[list[str], dict[str, str]]:
    """Enforce count and species exceptions on ordered picks, never on local scores.

    Species identity and whether a moment is exceptional remain visual-model
    judgments. These checks enforce the policy on the evidence it returned.
    """
    picked, dropped = [], {}
    species = defaultdict(list)
    for photo_id in dict.fromkeys(ids):
        a = assessments[photo_id]
        if not a.wildlife or not a.species or is_people_label(a.species):
            dropped[photo_id] = "Award roll: not a wildlife subject"
        elif not a.moment:
            dropped[photo_id] = "Award roll: missing visible moment description"
        elif curate and species[a.species] and (
            len(species[a.species]) >= 2 or not a.special_moment or not a.exceptional_reason
            or a.moment.casefold() in species[a.species]
        ):
            dropped[photo_id] = "Award roll: repeated species without a distinct exceptional moment"
        elif len(picked) >= limit:
            dropped[photo_id] = "Award roll: collection budget filled"
        else:
            picked.append(photo_id)
            species[a.species].append(a.moment.casefold())
    return picked, dropped


def award_shortlist(groups: dict[tuple[str, str], list[ScenedCandidate]], cap: int,
                    explore_slots: int = 2, reasons: dict[str, str] | None = None) -> list[str]:
    """Spend one strict collection budget across local subject labels.

    Allocate slots in proportion to sqrt(group size), so large bursts have
    diminishing influence. Local day/subject scores become tied percentiles
    before different dates compete within a species. The original scores are
    untouched. Labels, moments and nomination signals are fallible heuristics;
    this bounded search cannot guarantee species, scene or keeper coverage.
    """
    if cap < 1:
        return []
    species = defaultdict(list)
    for (_, label), group in sorted(groups.items()):
        scores = sorted(c.score for c in group)
        for c in group:
            rank = (bisect_left(scores, c.score) + bisect_right(scores, c.score) - 1) / 2
            score = 2 * rank / (len(scores) - 1) - 1 if len(scores) > 1 else 0
            species[label].append(replace(c, score=score))
    quotas = dict.fromkeys(species, 0)
    weights = {label: sqrt(len(group)) for label, group in species.items()}
    for _ in range(min(cap, sum(map(len, species.values())))):
        available = [label for label in species if quotas[label] < len(species[label])]
        label = min(available, key=lambda label: (quotas[label] / weights[label], label))
        quotas[label] += 1
    chosen = []
    for label, group in sorted(species.items()):
        chosen.extend(diverse_preselection(group, quotas[label], explore_slots=explore_slots,
                                          reasons=reasons, strict_cap=True))
    return chosen


def award_work_bound(rounds: list[list[str]], finalist_budget: int, chunk: int) -> tuple[int, int]:
    """Upper bounds on successful requests and photo presentations, before retries.

    Each merge reduces the number of pools at least once, so M pools require
    at most M-1 merge requests. Crops and provider tokenization are additional;
    these counts are not a dollar estimate or an API spending limit.
    """
    if not rounds:
        return 0, 0
    count = len(rounds)
    images = sum(map(len, rounds))
    return 2 * count, images + (count - 1) * max(chunk, 2 * finalist_budget) + min(images, finalist_budget)
