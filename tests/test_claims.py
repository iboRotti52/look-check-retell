from lcr.claims import extract, reference_classes
from lcr.vocab import CLASSES, KIND, SURFACE_TO_CLASS


def classes(caption):
    return [c.cls for c in extract(caption)]


def test_basic_extraction():
    assert classes("A dog playing with a ball in the grass") == ["dog", "ball", "grass"]


def test_person_terms_collapse_to_one_class():
    # gender / age words are one class and are deduplicated per caption
    assert classes("a man and a woman watch a little girl") == ["person"]
    for w in ("man", "woman", "boy", "girl", "child", "people"):
        assert SURFACE_TO_CLASS[w] == "person"


def test_synonym_groups():
    assert classes("a boy rides a bike") == ["person", "bicycle"]
    assert classes("a bicycle leans on a wall") == ["bicycle", "wall"]


def test_compound_modifier_is_not_a_claim():
    assert classes("a man holds a water bottle") == ["person", "bottle"]


def test_plural_and_tshirt():
    assert classes("two dogs near a man in a red t-shirt") == ["dog", "person", "shirt"]


def test_verb_reading_is_not_a_claim():
    assert "swing" not in classes("two children swing their arms")


def test_reference_union():
    refs = ["A dog runs.", "A puppy on the grass", "An animal outside"]
    assert reference_classes(refs) == {"dog", "grass"}


def test_vocabulary_is_consistent():
    assert set(KIND.values()) == {"person", "thing", "stuff"}
    assert set(SURFACE_TO_CLASS.values()) == set(CLASSES)
