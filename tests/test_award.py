"""Collection-wide wildlife curation, without model calls or real photo uploads."""
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fotosort.award import AWARD_SYSTEM, AwardAssessment, award_shortlist, award_work_bound, checked_award_picks
from fotosort.cli import Photo, apply_report, main, parse_args, score_and_reject, select_photos, write_report
from fotosort.judge import Judge, Verdict, parse_verdict
from fotosort.scan import find_images
from fotosort.selection import ScenedCandidate, diverse_preselection


def assessment(species="lion", special=False, moment="watching from the grass", why="", wildlife=True):
    return AwardAssessment(wildlife, species, moment, special, why)


def test_award_options_are_explicit_and_standard_defaults_stay_the_same():
    standard = parse_args(["."])
    assert (standard.mode, standard.judge_coverage, standard.highlights, standard.report) == (
        "standard", "preselect", "Highlights", "fotosort_report.csv")
    award = parse_args([".", "--mode", "award-roll", "--judge"])
    assert (award.roll_size, award.judge_coverage, award.highlights, award.report) == (
        12, "preselect", "AwardRoll", "award_roll_report.csv")
    assert award.award_candidates == 480
    explicit = parse_args([".", "--mode", "award-roll", "--judge", "--judge-coverage", "preselect",
                           "--roll-size", "15", "--highlights", "Portfolio", "--report", "p.csv"])
    assert (explicit.roll_size, explicit.judge_coverage, explicit.highlights, explicit.report) == (
        15, "preselect", "Portfolio", "p.csv")
    assert not parse_args([".", "--mode", "award-roll", "--apply-report"]).judge
    for flags in ([], ["--judge", "--include", "a"], ["--judge", "--roll-size", "9"],
                  ["--judge", "--roll-size", "16"], ["--judge", "--award-candidates", "0"],
                  ["--judge", "--award-candidates", "11"]):
        with pytest.raises(SystemExit):
            parse_args([".", "--mode", "award-roll", *flags])


def test_species_exceptions_need_visible_evidence_and_never_exceed_two():
    assessments = {
        "portrait": assessment(),
        "ordinary": assessment(moment="a different pose"),
        "vague": assessment(special=True, moment="an interaction"),
        "same": assessment(special=True, why="great light"),
        "action": assessment(special=True, moment="leaping across water", why="airborne with a reflection"),
        "third": assessment(special=True, moment="a hunt", why="intense action"),
        "bird": assessment("eagle"),
        "human": assessment("people"),
        "empty": assessment("landscape", wildlife=False),
    }
    picked, dropped = checked_award_picks(list(assessments), assessments, 12, curate=True)
    assert picked == ["portrait", "action", "bird"]
    assert set(dropped) == {"ordinary", "vague", "same", "third", "human", "empty"}
    assert checked_award_picks(list(assessments), assessments, 1, curate=True)[0] == ["portrait"]
    assert checked_award_picks([], {}, 12, curate=True) == ([], {})


def test_award_metadata_parsing_requires_typed_evidence_and_normalizes_species():
    item = dict(photo=2, reason="compelling", wildlife=True, species="  Cape  Fur Seal ",
                moment="playing", special_moment=True, exceptional_reason="pups interacting")
    verdict = parse_verdict(json.dumps({"picks": [item]}), ["a", "b"])
    assert verdict.picks == ["b"]
    assert verdict.awards["b"] == assessment("cape fur seal", True, "playing", "pups interacting")
    item["wildlife"] = "true"
    assert parse_verdict(json.dumps({"picks": [item]}), ["a", "b"]).awards is None


class AwardOracle:
    def __init__(self, evidence, ranks=None):
        self.evidence, self.ranks = evidence, ranks or {}
        self.calls = []

    def judge(self, paths, label, day, k, boxes, final=False):
        self.calls.append((list(map(str, paths)), k, self.award_final))
        ids = sorted(map(str, paths), key=lambda p: self.ranks.get(p, int(Path(p).stem)))
        return Verdict(ids, dict.fromkeys(ids, "strong wildlife photograph"), awards=self.evidence)


def photos_in(tmp_path, n):
    photos = []
    for i in range(n):
        path = tmp_path / f"{i:03}.jpg"
        path.touch()
        vec = np.array([1., 0.])  # even strong visual similarity must reach the judge
        photos.append(Photo(path, str(i), label="bird", time=datetime(2026, 9, 1) + timedelta(days=i % 3),
                            emb=vec, dino=vec, score=100 - i, scene=0))
    return photos


@pytest.mark.parametrize("n,chunk,target", [(81, 8, 12), (51, 1, 10), (45, 24, 15), (1, 24, 12)])
def test_full_collection_has_one_budget_and_every_distinct_frame_is_reviewed(tmp_path, n, chunk, target):
    photos = photos_in(tmp_path, n)
    evidence = {str(p.path): assessment(f"species {i}") for i, p in enumerate(photos)}
    # Deliberately reverse relative local scores: only the visual verdict can
    # compare different dates and groups for the final roll.
    ranks = {str(p.path): n - i for i, p in enumerate(photos)}
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge", "--judge-chunk", str(chunk),
                       "--roll-size", str(target), "--max-per-group", "1", "--judge-coverage", "full"])
    judge = AwardOracle(evidence, ranks)
    select_photos(photos, args, judge)
    assert {p.key for p in photos if p.selected} == {str(i) for i in range(max(0, n - target), n)}
    assert all(p.shortlisted for p in photos)
    assert sum(final for _, _, final in judge.calls) == 1
    assert len(judge.calls[-1][0]) <= 2 * target and judge.calls[-1][1] <= target
    assert all(len(paths) <= max(chunk, 4 * target) for paths, _, _ in judge.calls)
    winners = sorted((p for p in photos if p.selected), key=lambda p: p.award_rank)
    assert winners[0] is photos[-1]


def test_people_are_excluded_but_mislabelled_animals_and_special_moments_survive(tmp_path):
    photos = photos_in(tmp_path, 6)
    photos[0].label = "people"  # local mistake: the image is actually a lion
    values = [assessment(), assessment(moment="ordinary second portrait"),
              assessment(special=True, moment="cub greeting its mother", why="tender interaction"),
              assessment("people", wildlife=False), assessment("landscape", wildlife=False), assessment("eagle")]
    evidence = {str(p.path): a for p, a in zip(photos, values, strict=True)}
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge"])
    select_photos(photos, args, AwardOracle(evidence))
    assert [p.key for p in photos if p.selected] == ["0", "2", "5"]
    assert photos[0].judge_label == "lion" and photos[1].decision.startswith("Award roll: repeated species")
    assert all(p.shortlisted for p in photos)
    report = tmp_path / args.report
    write_report(photos, report, tmp_path)
    with report.open() as stream:
        rows = {r["relative_file"]: r for r in csv.DictReader(stream)}
    assert rows["002.jpg"]["award_rank"] == "2"
    assert rows["002.jpg"]["award_exceptional_reason"] == "tender interaction"
    assert rows["002.jpg"]["award_special_moment"] == "1"


def test_exact_copies_collapse_across_dates_and_sources_are_preserved(tmp_path):
    photos = photos_in(tmp_path, 3)
    photos[0].digest = photos[1].digest = "verified identical bytes"
    photos[2].source = tmp_path / "developed.dng"
    photos[2].source.touch()
    judge = AwardOracle({str(p.path): assessment(f"species {i}") for i, p in enumerate(photos)})
    select_photos(photos, parse_args([str(tmp_path), "--mode", "award-roll", "--judge"]), judge)
    assert not photos[1].shortlisted and "identical copy" in photos[1].decision
    assert judge.sources[str(photos[2].path)] == photos[2].source


def test_preselection_uses_one_global_cap_instead_of_group_share_or_scene_floors(tmp_path):
    photos = photos_in(tmp_path, 180)
    judge = AwardOracle({str(p.path): assessment(f"species {i}") for i, p in enumerate(photos)})
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge", "--judge-coverage", "preselect",
                       "--award-candidates", "24", "--preselect-min", "1000", "--preselect-share", "1"])
    for i, p in enumerate(photos):
        p.scene = i
    select_photos(photos, args, judge)
    assert sum(p.shortlisted for p in photos) == 24
    assert {i for paths, _, _ in judge.calls for i in paths} == {str(p.path) for p in photos if p.shortlisted}
    assert all(p.decision for p in photos)


@pytest.mark.parametrize("failure", ["network", "metadata"])
def test_award_failure_cannot_fall_back_to_relative_scores(tmp_path, monkeypatch, failure):
    photos = photos_in(tmp_path, 2)
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge"])
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class BadJudge:
        def judge(self, paths, *args, **kwargs):
            return Verdict([], {}, error="network") if failure == "network" else Verdict([str(paths[0])], {})

    with pytest.raises(SystemExit, match="no portfolio was exported"):
        select_photos(photos, args, BadJudge())
    assert not any(p.selected for p in photos)


def test_award_prompt_uses_wildlife_brief_and_empty_verdict_does_not_force_a_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    for path in paths:
        Image.new("RGB", (16, 16)).save(path)
    judge = Judge(mode="award-roll")
    assert judge.system == AWARD_SYSTEM
    prompts = []

    def ask(images, prompt):
        prompts.append(prompt)
        return '{"picks": []}'

    judge._ask_openai = ask
    assert not judge.judge(paths, "people", "all dates", 12).picks
    assert len(prompts) == 1 and "Nominate finalists" in prompts[0]
    assert "family" not in prompts[0] and "at least" not in prompts[0]
    judge.award_final = True
    assert not judge.judge(paths, "wildlife", "all dates", 12, final=True).picks
    assert "Make the final camera roll" in prompts[1] and "Aim for 12" in prompts[1]
    judge._ask_openai = lambda *_: '{"picks": [{"photo": 1, "reason": "nice"}]}'
    assert judge.judge(paths, "wildlife", "all dates", 12).error


@pytest.mark.parametrize("local", [False, True])
def test_award_api_payload_has_its_own_system_and_room_for_structured_finalists(monkeypatch, local):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    judge = Judge(mode="award-roll", model="test-model", base_url="http://localhost:1/v1" if local else None)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"picks":[]}'))])

    judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert judge._ask_openai([], "nominate") == '{"picks":[]}'
    assert requests[0]["messages"][0]["content"] == AWARD_SYSTEM
    assert requests[0]["max_tokens" if local else "max_completion_tokens"] == 10000


def test_award_report_can_be_applied_without_rejudging_and_output_is_not_scanned(tmp_path):
    photos = photos_in(tmp_path, 1)
    photo = replace(photos[0], selected=True, judge_label="lion", award_rank=1, edit_band="low")
    write_report([photo], tmp_path / "award_roll_report.csv", tmp_path)
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--apply-report", "--copy"])
    assert apply_report(photos, tmp_path, args) == 0
    assert (tmp_path / "AwardRoll" / "000.jpg").exists()
    assert find_images(tmp_path, True, set()) == [tmp_path / "000.jpg"]


def test_main_wires_award_judge_and_exports_only_winners_without_renaming_species(tmp_path, monkeypatch):
    originals = {}
    for i in range(3):
        path = tmp_path / f"{i:03}.jpg"
        Image.new("RGB", (16, 16), (20 + i, 30, 40)).save(path)
        originals[path] = path.read_bytes()

    def features(photos, root, args):
        for p in photos:
            p.emb = p.dino = np.array([1., 0.])
            p.label = "people"  # all local classifications are wrong
        return SimpleNamespace(text_embeddings=lambda *a, **kw: np.array([[1., 0.]]))

    class FakeJudge(AwardOracle):
        model = "test"
        usage = {"input": 0, "output": 0}

        def __init__(self, *args):
            assert args[-1] == "award-roll"
            super().__init__({str(p): assessment("lion") for p in originals})

        def name_subjects(self, *args):
            pytest.fail("award species must come from the curation verdict")

    monkeypatch.setattr("fotosort.cli.compute_features", features)
    monkeypatch.setattr("fotosort.cli.assign_scenes_and_labels", lambda *args: None)
    monkeypatch.setattr("fotosort.cli.editing_advice", lambda *args: None)
    monkeypatch.setattr("fotosort.judge.Judge", FakeJudge)
    assert main([str(tmp_path), "--mode", "award-roll", "--judge", "--copy", "--layout", "species"]) == 0
    assert list((tmp_path / "AwardRoll" / "lion").glob("*.jpg")) == [tmp_path / "AwardRoll/lion/000.jpg"]
    assert {p: p.read_bytes() for p in originals} == originals
    with (tmp_path / "award_roll_report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(r["selected"] == "1" for r in rows) == 1
    assert all(r["shortlisted"] == "1" for r in rows)


@pytest.mark.parametrize("cap", [120, 480])
def test_thousands_of_scenes_share_a_fixed_budget_and_small_species_remain_eligible(cap):
    groups = {}
    for day in range(3):
        for species in range(10):
            groups[(str(day), f"animal {species}")] = [
                ScenedCandidate(f"{day}/{species}/{i}", float(i), f"animal {species}", np.array([1., 0.]),
                                scene=day * 2000 + species * 200 + i, moment=i)
                for i in range(200)
            ]
    groups[("0", "rare bird")] = [ScenedCandidate("rare", -100, "rare bird", np.array([0., 1.]), scene=7000)]
    reasons = {}
    chosen = award_shortlist(groups, cap, reasons=reasons)
    assert len(chosen) == len(set(chosen)) == len(reasons) == cap
    assert "rare" in chosen
    assert {c.bucket for group in groups.values() for c in group if c.id in chosen} == {
        "rare bird", *[f"animal {i}" for i in range(10)]}
    assert award_shortlist(dict(reversed(list(groups.items()))), cap) == chosen
    assert groups[("0", "animal 0")][-1].score == 199  # percentile conversion does not mutate original scores


def test_cross_date_shortlisting_uses_within_group_ranks_and_preserves_ties():
    groups = {(str(day), "lion"): [ScenedCandidate(f"{day}/{i}", i // 2, "lion", np.array([1., 0.]),
                                                    scene=day, moment=i) for i in range(20)] for day in range(2)}
    expected = award_shortlist(groups, 12)
    transformed = {key: [replace(c, score=c.score * (100 if key[0] == "0" else .001) + 1000)
                         for c in group] for key, group in groups.items()}
    assert award_shortlist(transformed, 12) == expected
    assert award_shortlist({}, 12) == [] and award_shortlist(groups, 0) == []


def test_strict_scene_budget_leaves_room_for_focus_and_moment_alternatives():
    frames = [ScenedCandidate(str(i), 2 - i * .01, "lion", np.array([1., 0.]), scene=i, moment=i,
                              focus=0, distinctiveness=0) for i in range(100)]
    frames[-1].focus = 9
    frames[-2].distinctiveness = .9
    reasons = {}
    chosen = diverse_preselection(frames, 24, strict_cap=True, reasons=reasons)
    assert len(chosen) == 24 and {"99", "98"} <= set(chosen)
    assert reasons["99"] == "Independent focus nomination"
    assert reasons["98"] == "Independent distinctiveness nomination"
    assert len(diverse_preselection(frames, 24)) == 100  # standard mode still guarantees scene coverage


def test_comparison_bound_includes_finalists_and_small_groups_pack_together(tmp_path):
    photos = photos_in(tmp_path, 480)
    for i, p in enumerate(photos):
        p.label = f"species {i}"
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge"])
    judge = AwardOracle({str(p.path): assessment(p.label) for p in photos})
    select_photos(photos, args, judge)
    initial = [paths for paths, _, _ in judge.calls[:20]]
    assert all(len(paths) == 24 for paths in initial)
    requests, presentations = award_work_bound(initial, 24, 24)
    assert (requests, presentations) == (40, 1416)
    assert len(judge.calls) <= requests
    assert sum(len(paths) for paths, _, _ in judge.calls) <= presentations
    assert award_work_bound([], 24, 24) == (0, 0)


def test_full_coverage_explicitly_bypasses_candidate_cap(tmp_path):
    photos = photos_in(tmp_path, 65)
    judge = AwardOracle({str(p.path): assessment(f"species {i}") for i, p in enumerate(photos)})
    args = parse_args([str(tmp_path), "--mode", "award-roll", "--judge", "--judge-coverage", "full",
                       "--award-candidates", "12"])
    select_photos(photos, args, judge)
    assert all(p.shortlisted for p in photos)


def test_candidate_plan_matches_execution_without_claiming_photos_were_judged(tmp_path):
    photos = photos_in(tmp_path, 100)
    flags = [str(tmp_path), "--mode", "award-roll", "--award-candidates", "24"]
    plan = parse_args([*flags, "--award-plan"])
    select_photos(photos, plan)
    planned = {str(p.path) for p in photos if p.decision == "Planned for award judge (not sent)"}
    assert len(planned) == 24 and not any(p.shortlisted or p.selected for p in photos)
    assert plan.report == "award_roll_plan.csv"
    judge = AwardOracle({str(p.path): assessment("lion") for p in photos})
    select_photos(photos, parse_args([*flags, "--judge"]), judge)
    assert {str(p.path) for p in photos if p.shortlisted} == planned
    for flags in (["--award-plan"], ["--mode", "award-roll", "--award-plan", "--copy"],
                  ["--mode", "award-roll", "--award-plan", "--apply-report"]):
        with pytest.raises(SystemExit):
            parse_args([str(tmp_path), *flags])


def test_plan_and_paid_run_use_identical_quality_eligibility(tmp_path):
    photos = photos_in(tmp_path, 3)
    photos[0].clip_low = 1
    photos[1].label = "document or screenshot"
    flags = [str(tmp_path), "--mode", "award-roll"]
    score_and_reject(photos, parse_args([*flags, "--award-plan"]))
    plan_rejects = [p.reject for p in photos]
    score_and_reject(photos, parse_args([*flags, "--judge"]))
    assert plan_rejects == [p.reject for p in photos] == ["", "label", ""]


def test_main_plan_never_constructs_a_judge_even_when_judge_flag_is_present(tmp_path, monkeypatch):
    Image.new("RGB", (16, 16)).save(tmp_path / "000.jpg")

    def features(photos, *args):
        for p in photos:
            p.emb = np.array([1., 0.])
            p.label = "lion"
        return SimpleNamespace(text_embeddings=lambda *a, **kw: np.array([[1., 0.]]))

    monkeypatch.setattr("fotosort.cli.compute_features", features)
    monkeypatch.setattr("fotosort.cli.assign_scenes_and_labels", lambda *args: None)
    monkeypatch.setattr("fotosort.judge.Judge", lambda *a, **kw: pytest.fail("plan must not initialize a judge"))
    assert main([str(tmp_path), "--mode", "award-roll", "--award-plan", "--judge"]) == 0
    assert (tmp_path / "award_roll_plan.csv").exists()
    assert not (tmp_path / "AwardRoll").exists() and not (tmp_path / "award_roll_report.csv").exists()
