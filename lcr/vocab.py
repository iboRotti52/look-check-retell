"""Fixed claim vocabulary V.

V was curated by hand from the nouns that occur at least 30 times in the
Flickr8k *training* captions (see scripts/01_data_stats.py for the frequency
list). Each class has a canonical name, a `kind`, and its surface forms
(synonym group). Kinds:

  person : any human term; checked for presence only, gender/age never scored
  thing  : countable objects that a box detector can localise
  stuff  : amorphous regions / scenes (water, grass, beach ...). They are still
           claims, but a box-level check is ill-posed for them, so the pilot
           reports them separately.

Deliberately excluded: body parts (parts of an already-checked person/animal),
sport names that are ambiguous with the ball ("football", "basketball"),
generic placeholders ("object", "something", "animal", "vehicle", "food"),
and abstract/positional nouns ("front", "top", "air", "background").
"""
from __future__ import annotations

PERSON = (
    "person people man men woman women boy girl child children kid baby guy lady player "
    "skateboarder snowboarder rider adult skier surfer toddler biker climber hiker cyclist "
    "bicyclist racer motorcyclist wrestler teenager teen friend spectator runner cowboy bride "
    "groom onlooker worker goalie officer cheerleader mother father jockey crowd couple team "
    "family someone swimmer dancer athlete gentleman female male infant youngster youth "
    "performer musician policeman fisherman boarder kayaker golfer gymnast catcher pitcher "
    "batter hikers group band audience"
).split()

THINGS: dict[str, list[str]] = {
    # animals
    "dog": "dog puppy pup greyhound poodle retriever terrier collie labrador dachshund pug husky "
           "spaniel beagle bulldog chihuahua doberman rottweiler dalmatian hound puppies".split(),
    "horse": ["horse", "pony"], "bird": ["bird", "seagull", "pigeon"], "duck": ["duck"],
    "cat": ["cat", "kitten"], "cow": ["cow", "bull", "cattle", "calf"], "sheep": ["sheep", "lamb"],
    "goat": ["goat"], "fish": ["fish"], "camel": ["camel"], "elephant": ["elephant"],
    # vehicles and boards
    "bicycle": ["bicycle", "bike"], "motorcycle": ["motorcycle", "motorbike", "dirtbike"],
    "car": ["car", "jeep", "van"], "truck": ["truck"], "bus": ["bus"], "train": ["train"],
    "boat": ["boat", "sailboat", "ship"], "canoe": ["canoe"], "kayak": ["kayak"], "raft": ["raft"],
    "skateboard": ["skateboard"], "snowboard": ["snowboard"], "surfboard": ["surfboard"],
    "ski": ["ski"], "scooter": ["scooter"], "unicycle": ["unicycle"], "cart": ["cart", "wagon"],
    "sled": ["sled", "sledge"], "stroller": ["stroller"], "airplane": ["airplane", "plane"],
    # play / sports objects
    "ball": ["ball"], "frisbee": ["frisbee", "disc"], "toy": ["toy"], "stick": ["stick"],
    "rope": ["rope"], "racket": ["racket", "racquet"], "bat": ["bat"], "kite": ["kite"],
    "swing": ["swing"], "slide": ["slide"], "ramp": ["ramp"], "trampoline": ["trampoline"],
    "hurdle": ["hurdle"], "hoop": ["hoop"], "net": ["net"], "balloon": ["balloon"],
    "bubble": ["bubble"], "cone": ["cone"], "tire": ["tire"], "wheel": ["wheel"],
    "tube": ["tube"], "ladder": ["ladder"], "bone": ["bone"],
    # clothing and wearables
    "shirt": ["shirt", "tshirt", "jersey"], "jacket": ["jacket", "coat"],
    "sweater": ["sweater", "sweatshirt", "hoodie"], "hat": ["hat", "cap"], "helmet": ["helmet"],
    "dress": ["dress"], "shorts": ["shorts"], "pants": ["pants", "jeans", "trousers"],
    "suit": ["suit"], "uniform": ["uniform"], "costume": ["costume"], "vest": ["vest"],
    "skirt": ["skirt"], "scarf": ["scarf"], "bikini": ["bikini"],
    "swimsuit": ["swimsuit", "wetsuit"], "glove": ["glove"], "boot": ["boot"], "shoe": ["shoe"],
    "glasses": ["glasses", "sunglasses"], "goggles": ["goggles"], "mask": ["mask"],
    "backpack": ["backpack"], "bag": ["bag", "purse"], "headband": ["headband"],
    "bandanna": ["bandanna", "bandana"], "collar": ["collar"], "leash": ["leash"],
    "harness": ["harness"], "muzzle": ["muzzle"],
    # hand-held and furniture
    "camera": ["camera"], "guitar": ["guitar"], "microphone": ["microphone"],
    "umbrella": ["umbrella"], "bottle": ["bottle"], "cup": ["cup", "mug"], "plate": ["plate"],
    "book": ["book"], "newspaper": ["newspaper"], "phone": ["phone", "cellphone"],
    "cigarette": ["cigarette"], "bucket": ["bucket"], "box": ["box"], "blanket": ["blanket"],
    "towel": ["towel"], "flag": ["flag"], "sign": ["sign"], "bench": ["bench"],
    "chair": ["chair"], "table": ["table"], "couch": ["couch", "sofa"], "bed": ["bed"],
    "tent": ["tent"], "hose": ["hose"], "sprinkler": ["sprinkler"],
    # structures
    "fence": ["fence"], "wall": ["wall"], "building": ["building"], "house": ["house"],
    "window": ["window"], "door": ["door"], "bridge": ["bridge"], "pole": ["pole"],
    "fountain": ["fountain"], "statue": ["statue"], "tunnel": ["tunnel"],
    "stairs": ["stairs", "stair", "staircase", "steps"], "railing": ["railing", "rail"],
    "dock": ["dock", "pier"], "pool": ["pool"], "log": ["log"], "tree": ["tree"],
    "flower": ["flower"], "bush": ["bush"], "branch": ["branch"], "leaf": ["leaf", "leaves"],
}

STUFF: dict[str, list[str]] = {
    "water": ["water"], "grass": ["grass"], "snow": ["snow"], "field": ["field", "meadow"],
    "beach": ["beach"], "street": ["street", "road"], "rock": ["rock", "stone", "boulder"],
    "mountain": ["mountain", "hill"], "sand": ["sand"], "ocean": ["ocean", "sea"],
    "lake": ["lake"], "river": ["river"], "wave": ["wave"], "sidewalk": ["sidewalk", "pavement"],
    "path": ["path", "trail"], "track": ["track"], "dirt": ["dirt"], "mud": ["mud"],
    "sky": ["sky"], "forest": ["forest", "woods"], "park": ["park"],
    "yard": ["yard", "backyard", "lawn"], "ice": ["ice"], "pond": ["pond"],
    "puddle": ["puddle"], "stream": ["stream", "creek"], "waterfall": ["waterfall"],
    "cliff": ["cliff"], "shore": ["shore"], "floor": ["floor"], "city": ["city"],
    "desert": ["desert"], "garden": ["garden"], "fire": ["fire"], "cloud": ["cloud"],
    "court": ["court"], "stage": ["stage"], "playground": ["playground"],
}


def _build():
    surface_to_class: dict[str, str] = {}
    kind: dict[str, str] = {"person": "person"}
    for w in PERSON:
        surface_to_class[w] = "person"
    for table, k in ((THINGS, "thing"), (STUFF, "stuff")):
        for canon, forms in table.items():
            kind[canon] = k
            for f in forms:
                assert f not in surface_to_class, f"duplicate surface form: {f}"
                surface_to_class[f] = canon
    return surface_to_class, kind


SURFACE_TO_CLASS, KIND = _build()
CLASSES: list[str] = sorted(KIND)                # canonical class names, fixed order
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
