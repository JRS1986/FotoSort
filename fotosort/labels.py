"""Default zero-shot subject labels. Tuned for a Southern Africa trip but generic
enough for most travel photography. Override with --labels FILE (one per line)."""

DEFAULT_LABELS = [
    "lion", "leopard", "cheetah", "elephant", "rhinoceros", "buffalo", "giraffe",
    "zebra", "hippopotamus", "wildebeest", "kudu antelope", "impala antelope",
    "springbok", "warthog", "hyena", "wild dog", "jackal", "baboon", "vervet monkey",
    "meerkat", "mongoose", "crocodile", "tortoise", "lizard", "snake",
    "whale", "dolphin", "shark", "seal", "penguin",
    "bird", "eagle", "vulture", "ostrich", "flamingo", "owl", "hornbill",
    "butterfly", "insect", "spider",
    "landscape", "sunset or sunrise", "beach and ocean", "mountains", "waterfall",
    "savanna grassland", "night sky", "tree", "flower",
    "city street", "building", "vehicle", "safari jeep", "boat", "airplane",
    "group of people", "portrait of a person", "selfie", "food and drink", "wine",
    "document or screenshot", "indoor room",
]


def load_labels(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_LABELS
    with open(path) as f:
        labels = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return labels or DEFAULT_LABELS
