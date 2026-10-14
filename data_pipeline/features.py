from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LEXICON_VERSION = "1"

_INGREDIENTS = """
# proteins
beef
chicken
wings: wing
pork
bacon
ham
turkey
sausage
pepperoni
pancetta
steak
sirloin
ribeye
filet mignon
meatball
meat sauce
plant-based patty
# seafood
fish
cod
haddock
tuna
shrimp
scampi
calamari
mussel
sushi
# dairy and eggs
egg
cheese
cheddar
mozzarella
parmesan
pecorino
ricotta
feta
swiss cheese: swiss
butter
cream
ice cream
crema
milk
oat milk
yogurt
# bread, grains, pasta
rice
noodles: noodle
tortilla
taco shell
bun
bread
toast
roll
flatbread
english muffin: muffin
pancake
dumpling
spaghetti
fettuccine
penne
linguine
macaroni
angel hair
pasta
oats: oat
granola
quinoa
breadcrumb
batter
chips
fries: french fries
# vegetables, fruit, herbs, spices
tomato
tomato sauce
lettuce
greens
onion
mushroom
avocado
spinach
cabbage
cilantro
pineapple
pickle
cucumber
olive
olive oil
peppers: pepper
black pepper
sweet potato
potato
vegetables: vegetable
beans: bean
black beans: black bean
chickpea
edamame
tofu
falafel
hummus
tahini
basil
parsley
oregano
herbs: herb
garlic
ginger
lemon
lime
chili
chili flakes
salsa
pesto
pine nuts: pine nut
nuts: nut
berries: berry
banana
strawberry
acai
# sweet things, drinks, sauces
chocolate
vanilla
caramel
honey
syrup
sugar
brown sugar
powdered sugar
tea
tapioca pearls: tapioca pearl
coffee
espresso
cold brew
foam: microfoam
water
white wine
beer
soy sauce
soy
tartar sauce
vinaigrette
wasabi
sesame
"""

_CHARACTERISTICS = """
spicy: spice | fiery | heat | kick | chili | hot wing | hotter | arrabbiata
sweet: sugar | syrup | caramel | honey | glazed | glaze | chocolate | frosting | sweet | fizzy
fresh: freshly | fresh | bright
hearty: filling | substantial | satisfying | hearty | stacked | hungriest | indulgence | loaded | full plate | full english breakfast
buttery: butter | buttery
cheesy: cheesy | melty | melted | stretchy | bubbling mozzarella | cheddar
rich: indulgent | decadent | rich
bold: bolder | boldest | punchy | fiery | intense
cold: iced | chilled | frozen | cold | melts
creamy: velvety | silky | custardy | crema | cream sauce | creamy
smoky: smoked | smokey | smoky
tender: tenderly | tender
earthy
savory: savoury | umami | seasoned
light: lighter | lightly | refreshing | light
crispy: crisp | crisped | crispier | crispiest | crunchy | crunch | crunchier | crunchiest | breadcrumb crust | golden battered
"""

# A trigger word inside one of these phrases does NOT count for that characteristic.
# ("sweet potato" is a vegetable, not a sweet dish.)
CHARACTERISTIC_BLOCKERS = {
    "sweet": ["sweet potato"],
}

# Site tags that correspond to one of our characteristics (used to measure accuracy).
TAG_TO_CHARACTERISTIC = {
    "cheesy": "cheesy", "fresh": "fresh", "earthy": "earthy", "savory": "savory",
    "hearty": "hearty", "smoky": "smoky", "crispy": "crispy", "spicy": "spicy",
    "bold": "bold", "lighter": "light", "light": "light", "tender": "tender",
    "buttery": "buttery", "rich": "rich", "creamy": "creamy", "sweet": "sweet",
    "cold": "cold",
}

# Ingredients that contradict a "vegetarian" tag.
MEAT_AND_FISH = {
    "beef", "chicken", "pork", "bacon", "ham", "turkey", "sausage", "pepperoni", "pancetta",
    "steak", "sirloin", "ribeye", "filet mignon", "meatball", "meat sauce", "fish", "cod",
    "haddock", "tuna", "shrimp", "scampi", "calamari", "mussel",
}


def _parse(block: str) -> dict[str, list[str]]:
    """'name: alt1 | alt2' lines -> {name: [name, alt1, alt2]}. '#' lines are comments."""
    out: dict[str, list[str]] = {}
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(":")
        name = name.strip()
        forms = [name] + [f.strip() for f in rest.split("|") if f.strip()]
        out[name] = forms
    return out


INGREDIENTS = _parse(_INGREDIENTS)
CHARACTERISTICS = _parse(_CHARACTERISTICS)


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"

MODEL_NAME = "en_core_web_sm"
NEGATORS = {"no", "not", "without", "never", "zero", "non"}
CLAUSE_BREAKS = {",", ".", ";", ":", "!", "?"}
NEGATION_WINDOW = 2  # how many tokens back we look for a negator


def load_nlp():
    import spacy
    try:
        # We only need tokens, lemmas and part of speech - not the slow parts.
        return spacy.load(MODEL_NAME, disable=["ner", "parser"])
    except OSError:
        raise SystemExit(f"spaCy model missing. Run:  python -m spacy download {MODEL_NAME}")


def plural_variants(phrase: str) -> set[str]:
    """Generate more reasonable English singular and plural variants."""
    variants = {phrase}

    words = phrase.split()
    last_word = words[-1]

    if last_word.endswith(("s", "x", "z", "ch", "sh")):
        plural = last_word + "es"
    elif last_word.endswith("y") and len(last_word) > 1 and last_word[-2] not in "aeiou":
        plural = last_word[:-1] + "ies"
    else:
        plural = last_word + "s"

    variants.add(" ".join(words[:-1] + [plural]))
    return variants



class FeatureExtractor:
    def __init__(self, nlp=None):
        self.nlp = nlp or load_nlp()
        self.ingredient_patterns = self._compile(INGREDIENTS, plurals=True)
        self.characteristic_patterns = self._compile(CHARACTERISTICS, plurals=False)
        self.blockers = {
            name: self._compile({name: phrases}, plurals=True)
            for name, phrases in CHARACTERISTIC_BLOCKERS.items()
        }

    # ---- setup ----
    def _compile(self, lexicon: dict[str, list[str]], plurals: bool):
        """-> list of (word tuple, canonical name), longest phrases first."""
        patterns = set()
        for canonical, forms in lexicon.items():
            for form in forms:
                for variant in (plural_variants(form) if plurals else {form}):
                    words = tuple(t.lower_ for t in self.nlp.make_doc(variant))
                    patterns.add((words, canonical))
        return sorted(patterns, key=lambda p: (-len(p[0]), p[1], p[0]))

    # ---- matching ----
    @staticmethod
    def _negated(tokens, start: int) -> bool:
        for j in range(start - 1, max(start - 1 - NEGATION_WINDOW, -1), -1):
            if tokens[j].text in CLAUSE_BREAKS:
                return False
            if tokens[j].lower_ in NEGATORS:
                return True
        return False

    def _match(self, tokens, forms, patterns, blocked=None) -> list[str]:
        """Return canonical names in order of first appearance.
        Longest match wins within each characteristic.
        Different characteristics may share the same tokens.
        """
        used_by_canonical: dict[str, set[int]] = {}
        hits: list[tuple[int, str]] = []

        for words, canonical in patterns:
            size = len(words)
            used = used_by_canonical.setdefault(canonical, set())

            for i in range(len(tokens) - size + 1):
                positions = range(i, i + size)

                if any(pos in used for pos in positions):
                    continue

                if not all(words[k] in forms[i + k] for k in range(size)):
                    continue

                if blocked and any(
                        i + k in blocked.get(canonical, ())
                        for k in range(size)
                ):
                    continue

                if self._negated(tokens, i):
                    continue

                used.update(positions)
                hits.append((i, canonical))

        ordered: list[str] = []
        for _, canonical in sorted(hits):
            if canonical not in ordered:
                ordered.append(canonical)

        return ordered

    def _features_from_doc(self, doc) -> dict:
        tokens = list(doc)
        # For ingredients we also accept the lemma, but only for nouns:
        # "toasted" has the lemma "toast", and that must NOT count as the ingredient toast.
        ingredient_forms = [
            {t.lower_} | ({t.lemma_.lower()} if t.pos_ in ("NOUN", "PROPN") else set()) for t in tokens
        ]
        characteristic_forms = [{t.lower_, t.lemma_.lower()} for t in tokens]

        blocked: dict[str, set[int]] = {}
        for name, patterns in self.blockers.items():
            for words, _ in patterns:
                size = len(words)
                for i in range(len(tokens) - size + 1):
                    if all(words[k] in ingredient_forms[i + k] for k in range(size)):
                        blocked.setdefault(name, set()).update(range(i, i + size))

        return {
            "ingredients": self._match(tokens, ingredient_forms, self.ingredient_patterns),
            "characteristics": self._match(tokens, characteristic_forms, self.characteristic_patterns, blocked),
        }

    # ---- public API ----
    @staticmethod
    def _text(item: dict) -> str:
        return f"{item['name']}. {item.get('description_clean') or ''}"

    def extract(self, item: dict) -> dict:
        return self._features_from_doc(self.nlp(self._text(item)))

    def extract_many(self, items: list[dict]) -> list[dict]:
        docs = self.nlp.pipe([self._text(i) for i in items], batch_size=64)
        return [self._features_from_doc(d) for d in docs]



def evaluate(items: list[dict]) -> dict:
    """Compare extraction with the site's own tags and look for contradictions."""
    expected = found = 0
    missed: list[str] = []
    no_ingredients, no_characteristics, conflicts = [], [], []
    for item in items:
        for tag in item.get("tags", []):
            trait = TAG_TO_CHARACTERISTIC.get(tag)
            if trait:
                expected += 1
                if trait in item["characteristics"]:
                    found += 1
                else:
                    missed.append(f"{item['id']}: tag '{tag}' not found in text")
        if not item["ingredients"]:
            no_ingredients.append(item["id"])
        if not item["characteristics"]:
            no_characteristics.append(item["id"])
        if "vegetarian" in item.get("tags", []):
            meat = sorted(set(item["ingredients"]) & MEAT_AND_FISH)
            if meat:
                conflicts.append(f"{item['id']}: tagged vegetarian but mentions {', '.join(meat)}")
    return {
        "tag_traits_expected": expected,
        "tag_traits_found": found,
        "tag_recall": round(found / expected, 3) if expected else None,
        "missed_tag_traits": missed,
        "items_without_ingredients": no_ingredients,
        "items_without_characteristics": no_characteristics,
        "vegetarian_conflicts": conflicts,
        "avg_ingredients": round(sum(len(i["ingredients"]) for i in items) / len(items), 2) if items else 0,
        "avg_characteristics": round(sum(len(i["characteristics"]) for i in items) / len(items), 2) if items else 0,
    }


# ---------- files / command line ----------

def latest_clean() -> Path | None:
    files = sorted(CLEAN_DIR.glob("*_clean.json"))
    return files[-1] if files else None


def get_item_changes(
    current_items: list[dict],
    previous_items: list[dict],
) -> tuple[list[dict], list[str]]:
    """Return new or modified items and IDs of removed items."""

    previous_by_id = {
        item["id"]: item
        for item in previous_items
    }

    current_by_id = {
        item["id"]: item
        for item in current_items
    }

    changed_items = []

    fields_to_compare = ("name", "price", "description_clean")

    for item in current_items:
        previous = previous_by_id.get(item["id"])

        # New item
        if previous is None:
            changed_items.append(item)
            continue

        # Existing item whose important fields changed
        if any(
            item.get(field) != previous.get(field)
            for field in fields_to_compare
        ):
            changed_items.append(item)

    removed_ids = sorted(
        set(previous_by_id) - set(current_by_id)
    )

    return changed_items, removed_ids


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else latest_clean()
    if path is None or not path.exists():
        print(f"No input file found. Pass one: python features.py path/to/x_clean.json  (looked in {CLEAN_DIR})")
        return 2

    menu = json.loads(path.read_text(encoding="utf-8"))
    items = menu.get("items")
    if not isinstance(items, list) or not items:
        print("FAILED: the file has no items.")
        return 1

    extractor = FeatureExtractor()
    enriched = [{**item, **features} for item, features in zip(items, extractor.extract_many(items))]

    # Validate extracted features before saving.
    valid_characteristics = set(CHARACTERISTICS)
    valid_ingredients = set(INGREDIENTS)

    validation_errors = []

    for item in enriched:
        ingredients = item.get("ingredients")
        characteristics = item.get("characteristics")

        if not isinstance(ingredients, list):
            validation_errors.append(
                f"{item['id']}: ingredients must be a list"
            )
        else:
            if len(ingredients) != len(set(ingredients)):
                validation_errors.append(
                    f"{item['id']}: duplicate ingredients found"
                )

            for ingredient in ingredients:
                if ingredient not in valid_ingredients:
                    validation_errors.append(
                        f"{item['id']}: unknown ingredient '{ingredient}'"
                    )

        if not isinstance(characteristics, list):
            validation_errors.append(
                f"{item['id']}: characteristics must be a list"
            )
        else:
            if len(characteristics) != len(set(characteristics)):
                validation_errors.append(
                    f"{item['id']}: duplicate characteristics found"
                )

            for characteristic in characteristics:
                if characteristic not in valid_characteristics:
                    validation_errors.append(
                        f"{item['id']}: unknown characteristic '{characteristic}'"
                    )

    if validation_errors:
        print("Feature validation failed:")
        for error in validation_errors:
            print(" -", error)
        return 1

    print("Feature validation passed.")
    report = evaluate(enriched)
    inspect_missed_items(enriched, report)

    print(f"Input        : {path}")
    print(f"Items        : {len(enriched)}")
    print(f"Avg per item : {report['avg_ingredients']} ingredients, {report['avg_characteristics']} characteristics")
    print(f"Tag check    : {report['tag_traits_found']}/{report['tag_traits_expected']} "
          f"site tags confirmed by the text (recall {report['tag_recall']})")
    for key, label in (("items_without_ingredients", "No ingredients found"),
                       ("items_without_characteristics", "No characteristics found")):
        if report[key]:
            print(f"{label}: {', '.join(report[key])}")
    for line in report["vegetarian_conflicts"]:
        print(f"  CONFLICT {line}")
    for line in report["missed_tag_traits"][:15]:
        print(f"  missed   {line}")
    if len(report["missed_tag_traits"]) > 15:
        print(f"  ... and {len(report['missed_tag_traits']) - 15} more")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / (path.stem.removesuffix("_clean") + "_features.json")
    import spacy
    out.write_text(json.dumps({
        "features_at": datetime.now(timezone.utc).isoformat(),
        "source_file": path.name,
        "lexicon_version": LEXICON_VERSION,
        "spacy_version": spacy.__version__,
        "model": f"{MODEL_NAME} {extractor.nlp.meta['version']}",
        "items": enriched,
        "report": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved        : {out}")
    return 0


def inspect_missed_items(items: list[dict], report: dict) -> None:
    missed_ids = {
        line.split(": tag", 1)[0]
        for line in report["missed_tag_traits"]
    }

    for item in items:
        if item["id"] in missed_ids:
            print(f"\n--- {item['id']} ---")
            print("Name:", item["name"])
            print("Description:", item.get("description", ""))
            print("Tags:", item.get("tags", []))
            print("Extracted characteristics:", item.get("characteristics", []))


if __name__ == "__main__":
    sys.exit(main(sys.argv))