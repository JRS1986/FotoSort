"""Pick the best, most varied shots for one day across all subject buckets."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class Candidate:
    id: str
    score: float
    bucket: str
    emb: np.ndarray  # L2-normalised CLIP embedding (kept for labels; not used for duplicates)
    sig: np.ndarray | None = None  # zero-mean unit-norm thumbnail (see quality.signature)
    dino: np.ndarray | None = None  # L2-normalised DINOv2 embedding (visual sameness)
    digest: str = field(default="", kw_only=True)
    aesthetic: float | None = field(default=None, kw_only=True)
    composition: float | None = field(default=None, kw_only=True)
    editorial: float | None = field(default=None, kw_only=True)
    focus: float | None = field(default=None, kw_only=True)
    subject_dino: np.ndarray | None = field(default=None, kw_only=True)
    moment: int = field(default=-1, kw_only=True)
    distinctiveness: float | None = field(default=None, kw_only=True)
    uncertainty: float = field(default=0.0, kw_only=True)
    burst: int = field(default=-1, kw_only=True)
    capture_time: float | None = field(default=None, kw_only=True)
    burst_signals: dict[str, float] = field(default_factory=dict, kw_only=True)
    interaction_dino: np.ndarray | None = field(default=None, kw_only=True)


def is_duplicate(c: Candidate, picked: list[Candidate], dup_sim: float, dup_pixel: float,
                 reframe_sim: float = 0.98) -> bool:
    """Check matching views, including reframed versions within a known scene.

    A matching background alone must not veto a different pose. Exact source
    copies are always duplicates. CLIP is only a legacy fallback without DINO.
    Strong whole/subject DINO agreement in the same scene and bucket can bypass
    thumbnail correlation: panning or zooming changes pixel alignment. A zero
    reframe threshold disables this extra route. It never filters judge input.
    """
    for p in picked:
        if c.digest and c.digest == p.digest:
            return True
        full_sim = crop_sim = None
        if c.dino is not None and p.dino is not None:
            full_sim = float(np.dot(c.dino, p.dino))
            similar = full_sim >= dup_sim
        else:
            similar = float(np.dot(c.emb, p.emb)) >= dup_sim
        if c.sig is not None and p.sig is not None:
            similar = similar and float(np.dot(c.sig, p.sig)) >= dup_pixel
        if c.subject_dino is not None and p.subject_dino is not None:
            crop_sim = float(np.dot(c.subject_dino, p.subject_dino))
            similar = similar and crop_sim >= dup_sim
        same_scene = (getattr(c, "scene", -1) >= 0 and getattr(c, "scene", -1) == getattr(p, "scene", -1)
                      and c.bucket == p.bucket)
        reframed = (reframe_sim > 0 and same_scene and full_sim is not None and crop_sim is not None
                    and full_sim >= max(dup_sim, reframe_sim) and crop_sim >= dup_sim)
        if similar or reframed:
            return True
    return False


def select_day(
    cands: list[Candidate],
    budgets: dict[str, int],
    dup_sim: float,
    dup_pixel: float,
    min_score: float = -0.5,
    reframe_sim: float = 0.98,
) -> list[str]:
    """Greedy selection over a whole day.

    Candidates are visited best-first. One is taken when its score is at least
    `min_score`, its bucket still has budget, and it is not a near-duplicate
    (matching visual embedding and thumbnail framing) of any
    photo already taken that day, in any bucket. So a weak photo is never
    picked just because its subject has free slots, and the same colony shot
    labelled "seagull" and "cormorant" cannot get in twice.
    """
    picked: list[Candidate] = []
    count: dict[str, int] = defaultdict(int)
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if c.score < min_score:
            break
        if count[c.bucket] >= budgets.get(c.bucket, 0):
            continue
        if is_duplicate(c, picked, dup_sim, dup_pixel, reframe_sim):
            continue
        picked.append(c)
        count[c.bucket] += 1
    return [c.id for c in picked]


@dataclass(eq=False)
class ScenedCandidate(Candidate):
    scene: int = -1


def exact_twins(cands: list[Candidate]) -> tuple[list[Candidate], dict[str, str]]:
    """Keep the best-scoring representative of each verified identical source."""
    kept, twins, seen = [], {}, {}
    for c in sorted(cands, key=lambda c: (-c.score, c.id)):
        if c.digest and c.digest in seen:
            twins[c.id] = seen[c.digest]
            continue
        kept.append(c)
        if c.digest:
            seen[c.digest] = c.id
    return kept, twins


def build_shortlist(cands: list[ScenedCandidate], cap: int) -> list[str]:
    """What the judge gets to see for one bucket: the best frame of every scene
    (burst) first, then the next-best frames by score, up to `cap`. Only
    verified identical files are dropped before the judge."""
    kept, _ = exact_twins(cands)
    seen_scenes: set[int] = set()
    out: list[ScenedCandidate] = []
    for c in kept:  # pass 1: one per scene
        if c.scene not in seen_scenes:
            out.append(c)
            seen_scenes.add(c.scene)
    for c in kept:  # pass 2: fill by score
        if c not in out:
            out.append(c)
    return [c.id for c in out[:max(0, cap)]]


def chunk_for_tournament(cands: list[ScenedCandidate], chunk: int = 24) -> tuple[list[list[str]], dict[str, str]]:
    """Split one bucket into judge-sized chunks so every distinct frame is seen.
    Byte-identical sources are folded onto their best-scoring twin first; the
    rest are ordered by scene, then by score, and cut into chunks of at most
    `chunk` frames, balanced so the last chunk is not a stub.
    Returns (chunks of ids, {dropped id: id of the twin that stands for it})."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    kept, twins = exact_twins(cands)
    if any(c.burst >= 0 for c in kept):
        # Pack complete bursts where possible, without adding requests or
        # repeating frames. If bins cannot hold a whole burst, split only that
        # burst across the remaining spaces. Oversized bursts also must split.
        blocks = defaultdict(list)
        for c in kept:
            blocks[(c.scene, c.burst, "" if c.burst >= 0 else c.id)].append(c)
        n_chunks = -(-len(kept) // chunk)
        bins = [[] for _ in range(n_chunks)]
        for block in sorted(blocks.values(), key=lambda b: (-len(b), min(c.id for c in b))):
            block.sort(key=lambda c: (c.capture_time if c.capture_time is not None else float("inf"), c.id))
            fitting = [i for i, group in enumerate(bins) if chunk - len(group) >= len(block)]
            if fitting:
                i = min(fitting, key=lambda i: (chunk - len(bins[i]), i))
                bins[i].extend(block)
            else:
                remaining = list(block)
                for group in sorted(bins, key=len):
                    count = min(chunk - len(group), len(remaining))
                    group.extend(remaining[:count])
                    del remaining[:count]
        return [[c.id for c in group] for group in bins if group], twins
    kept.sort(key=lambda c: (c.scene, -c.score))
    if not kept:
        return [], twins
    n_chunks = max(1, -(-len(kept) // chunk))
    size, extra = divmod(len(kept), n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        end = start + size + (i < extra)
        chunks.append([c.id for c in kept[start:end]])
        start = end
    return chunks, twins


def diverse_preselection(cands: list[ScenedCandidate], cap: int, quality_share: float | None = None,
                         explore_slots: int = 2, reasons: dict[str, str] | None = None,
                         *, strict_cap: bool = False) -> list[str]:
    """Reserve quality, burst alternatives, then remaining visual coverage.

    Normally scene coverage can expand the cap. With strict_cap, scenes instead
    compete for roughly 35% of slots, leaving room for alternative nominations.
    Up to `explore_slots` of the existing
    slots consider novelty/uncertainty without a quality penalty. No candidate
    is excluded for a low score. Similarity updates need O(N*D), not O(N*N), memory.
    With temporal burst assignments, quality and bursts each receive a target
    40%; unused burst slots return to coverage. Without them, preserve the
    established independent-nomination/coverage policy. Burst signals never
    change global scores.
    """
    if not cands or cap <= 0:
        return []
    ranked, _ = exact_twins(cands)
    use_bursts = any(c.burst >= 0 for c in ranked)
    if quality_share is None:
        quality_share = .4 if use_bursts else .5
    n_scenes = len({c.scene for c in ranked})
    cap = cap if strict_cap else max(cap, n_scenes)
    reasons = reasons if reasons is not None else {}
    if len(ranked) <= cap:
        reasons.update({c.id: "All distinct frames fit" for c in ranked})
        return [c.id for c in ranked]
    scene_slots = min(n_scenes, max(1, round(cap * .35))) if strict_cap else n_scenes
    n_explore = min(max(0, explore_slots), max(0, cap - scene_slots - 1))
    n_quality = min(cap - n_explore, max(scene_slots, int(round(cap * quality_share))))
    scores = np.array([c.score for c in ranked])
    scale = max(float(np.median(np.abs(scores - np.median(scores)))) * 1.4826, 1.0)
    quality = 1 / (1 + np.exp(-np.clip((scores - np.median(scores)) / scale, -30, 30)))
    # Missing crops use only the whole frame; lack of a detection is not novelty.
    use_dino = any(c.dino is not None for c in ranked)
    vectors = [c.dino if use_dino else c.emb for c in ranked]
    whole_shape = next(v.shape for v in vectors if v is not None)
    has_whole = np.array([v is not None for v in vectors])
    whole = np.stack([v if v is not None else np.zeros(whole_shape) for v in vectors])
    crop_shape = next((c.subject_dino.shape for c in ranked if c.subject_dino is not None), None)
    has_crop = np.array([c.subject_dino is not None for c in ranked])
    crops = np.stack([c.subject_dino if c.subject_dino is not None else np.zeros(crop_shape)
                      for c in ranked]) if crop_shape else None
    context_shape = next((c.interaction_dino.shape for c in ranked if c.interaction_dino is not None), None)
    has_context = np.array([c.interaction_dino is not None for c in ranked])
    contexts = np.stack([c.interaction_dino if c.interaction_dino is not None else np.zeros(context_shape)
                         for c in ranked]) if context_shape else None
    closest = np.where(has_whole | has_crop | has_context, -1.0, 1.0)
    burst_closest = closest.copy()
    available = np.ones(len(ranked), dtype=bool)
    moments = [(c.scene, c.moment) for c in ranked]
    covered = set()
    chosen = []
    burst_members = defaultdict(list)
    for i, candidate in enumerate(ranked):
        if candidate.burst >= 0:
            burst_members[(candidate.scene, candidate.burst)].append(i)

    def similarities(i):
        sim = np.where(has_whole & has_whole[i], whole @ whole[i], -1.0)
        comparable = has_whole & has_whole[i]
        if crops is not None and has_crop[i]:
            crop_sim = crops @ crops[i]
            sim = np.where(has_crop, np.where(comparable, np.minimum(sim, crop_sim), crop_sim), sim)
            comparable |= has_crop
        if contexts is not None and has_context[i]:
            context_sim = contexts @ contexts[i]
            sim = np.where(has_context, np.where(comparable, np.minimum(sim, context_sim), context_sim), sim)
        return np.clip(sim, -1, 1)

    def add(i, reason):
        chosen.append(ranked[i].id)
        reasons[ranked[i].id] = reason
        available[i] = False
        covered.add(moments[i])
        sim = similarities(i)
        np.maximum(closest, np.clip(sim, -1, 1), out=closest)
        peers = burst_members.get((ranked[i].scene, ranked[i].burst), [])
        burst_closest[peers] = np.maximum(burst_closest[peers], sim[peers])

    scenes = set()
    for i, c in enumerate(ranked):
        if len(scenes) >= scene_slots:
            break
        if c.scene not in scenes:
            add(i, "Scene representative")
            scenes.add(c.scene)

    # A winner on an independent signal can lose in the combined score.
    # Reserve nominations before general coverage, inside the same allocation.
    for signal in ("aesthetic", "focus", "editorial", "composition", "distinctiveness"):
        if len(chosen) >= n_quality:
            break
        nominees = [i for i, c in enumerate(ranked) if getattr(c, signal) is not None
                    and np.isfinite(getattr(c, signal))]
        if not nominees:
            continue
        values = [getattr(ranked[i], signal) for i in nominees]
        if max(values) - min(values) <= 1e-6:
            continue
        best = max(nominees, key=lambda i: getattr(ranked[i], signal)
                   * (quality[i] if signal == "distinctiveness" else 1))
        if available[best]:
            add(best, f"Independent {signal} nomination")

    # This is a real allocation, not just a limit on independent nominations.
    for i in range(len(ranked)) if use_bursts else []:
        if len(chosen) >= n_quality:
            break
        if available[i]:
            add(i, "Protected quality allocation")

    bursts = [peers for peers in burst_members.values() if len(peers) > 1]
    n_burst = min(round(cap * .4), cap - len(chosen) - n_explore)
    for _ in range(n_burst):
        eligible = [peers for peers in bursts if available[peers].any()]
        if not eligible:
            break
        # Spread opportunities across capture sequences, accounting for frames
        # already admitted by quality. Dense bursts do not win every iteration.
        peers = min(eligible, key=lambda ps: (sum(not available[i] for i in ps) / len(ps),
                                             -max(quality[i] for i in ps), ranked[ps[0]].id))
        seen = [i for i in peers if not available[i]]
        alternatives = [i for i in peers if available[i]]
        if not seen:
            add(alternatives[0], "Burst representative")
            continue
        best_story = None
        # Alternate semantic challenges with visual pose challenges. No single
        # extreme expression can monopolise every slot in a capture sequence.
        if len(seen) % 3 == 1:
            for signal in ("expression", "interaction", "gesture"):
                values = {i: ranked[i].burst_signals.get(signal) for i in peers}
                values = {i: v for i, v in values.items() if v is not None and np.isfinite(v)}
                if len(values) < 2 or max(values.values()) - min(values.values()) < .005:
                    continue  # small CLIP differences are not evidence of a changed expression
                prior = max((values[i] for i in seen if i in values), default=None)
                if prior is None:
                    continue
                for i in alternatives:
                    if i in values and values[i] > prior:
                        gain = (values[i] - prior) / (max(values.values()) - min(values.values()))
                        nominee = (gain, quality[i], -i, i, signal)
                        if best_story is None or nominee[:3] > best_story[:3]:
                            best_story = nominee
        if best_story is not None:
            add(best_story[3], f"Burst alternative: {best_story[4]}")
        elif len(seen) % 3 == 0:
            add(alternatives[0], "Burst quality alternative")
        else:
            utility = (.35 + .65 * quality) * np.sqrt((1 - burst_closest) / 2)
            i = max(alternatives, key=lambda i: (utility[i], quality[i], -i))
            add(i, "Burst pose alternative" if utility[i] > 1e-5 else "Burst quality alternative")

    # Give unseen moments a soft preference. Multiplying novelty by quality
    # prevents low-quality pans/shake from dominating every diversity slot.
    while len(chosen) < cap - n_explore:
        novelty = np.sqrt((1 - closest) / 2)
        uncovered = np.array([m not in covered for m in moments])
        utility = quality * (0.15 + 0.85 * novelty + 0.3 * uncovered * novelty)
        i = int(np.argmax(np.where(available, utility, -np.inf)))
        add(i, "Moment representative" if uncovered[i] else "Quality and visual coverage")

    # Signal disagreement and label uncertainty are heuristics, not calibrated
    # probabilities. Exploration has a strict allowance and never grows the cap.
    disagreement = np.zeros(len(ranked))
    for signal in ("aesthetic", "focus", "editorial", "composition"):
        values = np.array([getattr(c, signal) if getattr(c, signal) is not None else np.nan for c in ranked])
        valid = np.isfinite(values)
        if valid.any() and np.ptp(values[valid]) > 1e-6:
            ranks = np.zeros(len(ranked))
            ranks[valid] = (values[valid] - values[valid].min()) / np.ptp(values[valid])
            disagreement = np.maximum(disagreement, np.where(valid, np.abs(ranks - quality), 0))
    uncertain = np.array([np.clip(c.uncertainty, 0, 1) for c in ranked])
    while len(chosen) < cap:
        utility = 0.7 * (1 - closest) / 2 + 0.2 * disagreement + 0.1 * uncertain
        i = int(np.argmax(np.where(available, utility, -np.inf)))
        add(i, "Exploration: visual novelty or signal uncertainty")
    return chosen


def dedup_by_score(cands: list[Candidate], k: int, dup_sim: float, dup_pixel: float,
                   reframe_sim: float = 0.98) -> list[str]:
    """Best-first top-k without near-duplicates: the judge-less fallback."""
    picked: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if len(picked) >= k:
            break
        if not is_duplicate(c, picked, dup_sim, dup_pixel, reframe_sim):
            picked.append(c)
    return [c.id for c in picked]
