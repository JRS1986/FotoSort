"""Default zero-shot subject labels. Tuned for a Southern Africa trip but generic
enough for most travel photography. Override with --labels FILE (one per line)."""

DEFAULT_LABELS = [
    "lion", "leopard", "cheetah", "elephant", "rhinoceros", "buffalo", "giraffe",
    "zebra", "hippopotamus", "wildebeest", "kudu antelope", "impala antelope", "eland antelope",
    "bontebok antelope", "gemsbok oryx", "springbok", "warthog", "hyena", "wild dog", "jackal", "baboon",
    "vervet monkey",
    "meerkat", "mongoose", "rock hyrax (dassie)", "crocodile", "tortoise", "lizard", "snake",
    "whale", "dolphin", "shark", "seal", "penguin",
    "bird", "eagle", "vulture", "ostrich", "flamingo", "owl", "hornbill", "seagull", "cormorant",
    "heron", "ibis", "pelican", "guinea fowl", "weaver bird", "sunbird", "duck or goose",
    "butterfly", "insect", "spider",
    "landscape", "sunset or sunrise", "beach and ocean", "mountains", "waterfall",
    "savanna grassland", "night sky", "tree", "flower",
    "city street", "building", "vehicle", "safari jeep", "boat", "airplane",
    "group of people", "portrait of a person", "person standing in a landscape", "child", "selfie",
    "man", "woman", "boy", "girl", "family posing for a photo", "tourist taking a photo",
    "food and drink", "wine",
    "document or screenshot", "indoor room",
]


# Labels that share one bucket, so people are not split into man/woman/child/... groups
# and get a sensible number of picks per day.
LABEL_GROUPS = {
    "group of people": "people", "portrait of a person": "people", "person standing in a landscape": "people",
    "child": "people", "selfie": "people", "man": "people", "woman": "people", "boy": "people",
    "girl": "people", "family posing for a photo": "people", "tourist taking a photo": "people",
}


def bucket_name(label: str) -> str:
    return LABEL_GROUPS.get(label, label)


PEOPLE_WORDS = ("person", "people", "child", "selfie", "diver", "portrait", "family", "tourist")


def is_people_label(label: str) -> bool:
    """Labels whose CLIP probability counts as evidence of a person in the frame."""
    low = label.lower()
    return bucket_name(label) == "people" or any(w in low for w in PEOPLE_WORDS)


def load_labels(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_LABELS
    with open(path) as f:
        labels = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return labels or DEFAULT_LABELS
