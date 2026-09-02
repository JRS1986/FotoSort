from fotosort.cli import pick_budget
from fotosort.labels import bucket_name


def test_pick_budget_grows_with_group_size_and_caps():
    assert pick_budget(6, 5, 40) == 5
    assert pick_budget(80, 5, 40) == 7
    assert pick_budget(454, 5, 40) == 15
    assert pick_budget(454, 5, 0) == 5


def test_people_labels_share_a_bucket():
    assert bucket_name("child") == "people" and bucket_name("man") == "people"
    assert bucket_name("zebra") == "zebra"
