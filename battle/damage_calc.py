"""Gen 3 (FireRed/LeafGreen) damage calculator.

Implements the damage formula as documented for Generation III, including the
integer truncation at each step -- the truncation is not cosmetic, it changes
results by 1-2 HP regularly and matters for KO thresholds.

Typical use from the AI player:

    data = GameData.load()
    attacker = Pokemon(species="IVYSAUR", level=17, types=("Grass", "Poison"),
                       stats={"atk": 28, "def": 30, "spa": 36, "spd": 35, "spe": 33},
                       max_hp=48, current_hp=48, ability="Overgrow")
    defender = Pokemon(species="PIDGEY", level=5, types=("Normal", "Flying"),
                       stats={"atk": 10, "def": 9, "spa": 9, "spd": 10, "spe": 10},
                       max_hp=19, current_hp=19, ability="Keen Eye")
    result = calculate(data, attacker, defender, data.move("VINE WHIP"))
    print(result.summary())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DATA_DIR = Path(__file__).parent

# In Gen 3 a move's damage category follows its type, not the move itself.
PHYSICAL_TYPES = frozenset(
    {"Normal", "Fighting", "Flying", "Ground", "Rock", "Bug", "Ghost", "Poison", "Steel"}
)
SPECIAL_TYPES = frozenset(
    {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}
)

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# The in-game move names FRLG displays differ from the canonical names in
# moves.json for a handful of moves. Keys are normalized forms.
MOVE_ALIASES = {
    "vicegrip": "visegrip",          # VICEGRIP -> Vise Grip
    "faintattack": "feintattack",    # FAINT ATTACK -> Feint Attack
    "hijumpkick": "highjumpkick",
    "smellingsalt": "smellingsalts",
    "selfdestruct": "selfdestruct",
    "ancientpower": "ancientpower",
    "thunderpunch": "thunderpunch",
    "sandattack": "sandattack",
    "doubleedge": "doubleedge",
    "softboiled": "softboiled",
}

# Type-boosting held items: 1.1x to matching-type moves in Gen 3.
TYPE_BOOST_ITEMS = {
    "silk scarf": "Normal",
    "charcoal": "Fire",
    "mystic water": "Water",
    "miracle seed": "Grass",
    "magnet": "Electric",
    "never-melt ice": "Ice",
    "nevermeltice": "Ice",
    "black belt": "Fighting",
    "poison barb": "Poison",
    "soft sand": "Ground",
    "sharp beak": "Flying",
    "twisted spoon": "Psychic",
    "silver powder": "Bug",
    "hard stone": "Rock",
    "spell tag": "Ghost",
    "dragon fang": "Dragon",
    "black glasses": "Dark",
    "metal coat": "Steel",
    "sea incense": "Water",
    "odd incense": "Psychic",
    "rock incense": "Rock",
    "wave incense": "Water",
    "rose incense": "Grass",
}

# Pinch abilities: 1.5x base power for the matching type at <= 1/3 max HP.
PINCH_ABILITIES = {
    "overgrow": "Grass",
    "blaze": "Fire",
    "torrent": "Water",
    "swarm": "Bug",
}

HIGH_CRIT_MOVES = frozenset(
    {
        "karatechop", "razorleaf", "crabhammer", "slash", "aeroblast",
        "crosschop", "aircutter", "blazekick", "leafblade", "poisontail",
        "razorwind", "skyattack",
    }
)

# Moves that deal a fixed amount of damage, bypassing the formula entirely.
FIXED_DAMAGE_MOVES = frozenset(
    {"seismictoss", "nightshade", "dragonrage", "sonicboom", "superfang", "psywave"}
)

OHKO_MOVES = frozenset({"fissure", "horndrill", "guillotine", "sheercold"})

# Moves whose stored power of 1 is a placeholder for a runtime calculation.
VARIABLE_POWER_MOVES = frozenset(
    {
        "lowkick", "return", "frustration", "flail", "reversal", "magnitude",
        "hiddenpower", "endeavor", "spitup", "present", "weatherball",
        "furycutter", "rollout", "iceball", "eruption", "waterspout",
        "beatup", "triplekick",
    }
)

# Type effectiveness is forced to neutral for these regardless of the chart.
TYPELESS_MOVES = frozenset({"struggle", "futuresight", "beatup", "doomdesire"})


def normalize(name: str) -> str:
    """Fold a move or species name to a lookup key.

    Handles the game's uppercase display forms, hyphen/space/period differences
    and the mojibake gender symbols in pokedex.json.
    """
    key = name.lower().strip()
    key = key.replace("♀", "f").replace("♂", "m")
    key = re.sub(r"[^a-z0-9]", "", key)
    return MOVE_ALIASES.get(key, key)


class DataError(Exception):
    """Raised when a lookup against the JSON data files fails."""


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    name: str
    type: str
    power: int | None
    accuracy: int | None
    category: str
    tm: str | None = None
    effect: str = ""

    @property
    def key(self) -> str:
        return normalize(self.name)

    @property
    def is_physical(self) -> bool:
        return self.category == "Physical"

    @property
    def is_special(self) -> bool:
        return self.category == "Special"

    @property
    def is_status(self) -> bool:
        return self.category == "Status"

    @property
    def never_misses(self) -> bool:
        # moves.json uses both null and 0 for "cannot miss".
        return not self.accuracy

    @classmethod
    def from_json(cls, raw: Mapping) -> "Move":
        mtype = raw["type"]
        if raw.get("power") is None and raw.get("category") == "Status":
            category = "Status"
        elif mtype in PHYSICAL_TYPES:
            category = "Physical"
        elif mtype in SPECIAL_TYPES:
            category = "Special"
        else:
            category = raw.get("category", "Status")
        return cls(
            name=raw["name"],
            type=mtype,
            power=raw.get("power"),
            accuracy=raw.get("accuracy"),
            category=category,
            tm=raw.get("tm"),
            effect=raw.get("effect", ""),
        )


@dataclass
class GameData:
    """Loaded pokedex, move list and type chart."""

    moves: dict[str, Move]
    species_by_id: dict[int, dict]
    species_by_name: dict[str, dict]
    type_chart: dict[str, dict[str, float]]
    types: tuple[str, ...]

    @classmethod
    def load(cls, directory: Path | str = DATA_DIR) -> "GameData":
        directory = Path(directory)
        moves_raw = _read_json(directory / "moves.json")
        dex_raw = _read_json(directory / "pokedex.json")
        chart_raw = _read_json(directory / "typeChart.json")

        moves = {}
        for entry in moves_raw:
            move = Move.from_json(entry)
            moves[move.key] = move

        by_id, by_name = {}, {}
        for entry in dex_raw:
            by_id[entry["id"]] = entry
            key = normalize(entry["name"])
            # IDs 29/32 (Nidoran female/male) collide because pokedex.json has
            # mojibake gender symbols. First one wins; prefer lookup by id.
            by_name.setdefault(key, entry)

        return cls(
            moves=moves,
            species_by_id=by_id,
            species_by_name=by_name,
            type_chart=chart_raw["chart"],
            types=tuple(chart_raw["types"]),
        )

    def move(self, name: str) -> Move:
        key = normalize(name)
        if key not in self.moves:
            raise DataError(f"unknown move: {name!r} (normalized {key!r})")
        return self.moves[key]

    def species(self, ident: int | str) -> dict:
        if isinstance(ident, int):
            if ident not in self.species_by_id:
                raise DataError(f"unknown species id: {ident}")
            return self.species_by_id[ident]
        key = normalize(ident)
        if key not in self.species_by_name:
            raise DataError(f"unknown species: {ident!r}")
        return self.species_by_name[key]

    def effectiveness(self, attack_type: str, defend_type: str) -> float:
        return self.type_chart.get(attack_type, {}).get(defend_type, 1.0)


def _read_json(path: Path):
    # The JSON files were saved as UTF-8 but contain mojibake sequences from an
    # earlier bad decode; utf-8 still parses them, the damage is in the strings.
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Battle entities
# --------------------------------------------------------------------------


@dataclass
class Pokemon:
    """A battler, built from the state the emulator exposes."""

    species: str
    level: int
    types: tuple[str, ...]
    stats: Mapping[str, int]           # atk, def, spa, spd, spe (already final)
    max_hp: int
    current_hp: int
    ability: str = ""
    item: str = ""
    status: str = ""                   # burn, poison, toxic, paralysis, sleep, freeze
    stages: dict[str, int] = field(default_factory=dict)
    friendship: int = 70
    weight_kg: float = 0.0             # only needed for Low Kick
    stockpile: int = 0
    consecutive_hits: int = 0          # Rollout / Ice Ball / Fury Cutter counter
    flash_fire_active: bool = False
    charged: bool = False

    def __post_init__(self):
        self.types = tuple(t for t in self.types if t)
        base = {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "acc", "eva")}
        base.update(self.stages)
        self.stages = base

    @property
    def hp_fraction(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp else 0.0

    def stat(self, key: str, *, ignore_negative=False, ignore_positive=False) -> int:
        """Effective stat after stage modifiers."""
        raw = self.stats[key]
        stage = self.stages.get(key, 0)
        if ignore_negative and stage < 0:
            stage = 0
        if ignore_positive and stage > 0:
            stage = 0
        if stage >= 0:
            numerator, denominator = 2 + stage, 2
        else:
            numerator, denominator = 2, 2 - stage
        return max(1, raw * numerator // denominator)


@dataclass
class Field:
    """Field and side conditions relevant to damage."""

    weather: str = ""                  # rain, sun, sandstorm, hail
    reflect: bool = False              # on the *defender's* side
    light_screen: bool = False
    is_double: bool = False
    cloud_nine: bool = False           # Cloud Nine / Air Lock on the field


@dataclass
class Context:
    """Per-use modifiers the caller can toggle."""

    critical: bool = False
    helping_hand: bool = False
    double_damage: bool = False        # Facade on a statused user, Pursuit on a switch, etc.
    spread_move: bool = False          # halves damage in doubles
    magnitude_power: int | None = None
    hidden_power_type: str | None = None
    hidden_power_power: int | None = None
    target_switching: bool = False
    target_paralyzed: bool = False
    damaged_this_turn: bool = False


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class DamageResult:
    move: Move
    rolls: tuple[int, ...]             # 16 damage rolls, lowest to highest
    effectiveness: float
    stab: bool
    category: str
    power: int
    attack: int
    defense: int
    critical: bool
    defender_hp: int
    notes: tuple[str, ...] = ()

    @property
    def min_damage(self) -> int:
        return self.rolls[0]

    @property
    def max_damage(self) -> int:
        return self.rolls[-1]

    @property
    def average_damage(self) -> float:
        return sum(self.rolls) / len(self.rolls)

    @property
    def guaranteed_ko(self) -> bool:
        return self.min_damage >= self.defender_hp

    @property
    def possible_ko(self) -> bool:
        return self.max_damage >= self.defender_hp

    @property
    def ko_chance(self) -> float:
        """Probability this single hit KOs, assuming it hits."""
        if not self.rolls:
            return 0.0
        return sum(1 for r in self.rolls if r >= self.defender_hp) / len(self.rolls)

    @property
    def min_percent(self) -> float:
        return 100.0 * self.min_damage / self.defender_hp if self.defender_hp else 0.0

    @property
    def max_percent(self) -> float:
        return 100.0 * self.max_damage / self.defender_hp if self.defender_hp else 0.0

    def summary(self) -> str:
        if self.category == "Status":
            return f"{self.move.name}: status move, no damage"
        label = {0: "immune", 0.25: "double resisted", 0.5: "not very effective",
                 1: "neutral", 2: "super effective", 4: "double super effective"}
        eff = label.get(self.effectiveness, f"x{self.effectiveness}")
        if self.max_damage == 0:
            return f"{self.move.name}: no damage [{eff}]"
        parts = [
            f"{self.move.name}: {self.min_damage}-{self.max_damage} "
            f"({self.min_percent:.1f}-{self.max_percent:.1f}%) [{eff}]"
        ]
        if self.stab:
            parts.append("STAB")
        if self.critical:
            parts.append("crit")
        if self.guaranteed_ko:
            parts.append("guaranteed OHKO")
        elif self.possible_ko:
            parts.append(f"OHKO {self.ko_chance:.0%}")
        else:
            hits = -(-self.defender_hp // max(1, int(self.average_damage)))
            parts.append(f"~{hits}HKO")
        if self.notes:
            parts.extend(self.notes)
        return " | ".join(parts)


# --------------------------------------------------------------------------
# Power / stat resolution
# --------------------------------------------------------------------------


def _flail_power(hp_fraction: float) -> int:
    scaled = int(hp_fraction * 48)
    if scaled < 1:
        return 200
    if scaled < 4:
        return 150
    if scaled < 9:
        return 100
    if scaled < 17:
        return 80
    if scaled < 33:
        return 40
    return 20


def _low_kick_power(weight_kg: float) -> int:
    for threshold, power in ((10, 20), (25, 40), (50, 60), (100, 80), (200, 100)):
        if weight_kg < threshold:
            return power
    return 120


def resolve_power(
    data: GameData,
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    fieldstate: Field,
    ctx: Context,
) -> tuple[int, list[str]]:
    """Return the effective base power plus any notes about assumptions made."""
    key = move.key
    notes: list[str] = []
    power = move.power or 0

    if key == "return":
        power = max(1, attacker.friendship * 2 // 5)
    elif key == "frustration":
        power = max(1, (255 - attacker.friendship) * 2 // 5)
    elif key in ("flail", "reversal"):
        power = _flail_power(attacker.hp_fraction)
    elif key in ("eruption", "waterspout"):
        power = max(1, int(150 * attacker.hp_fraction))
    elif key == "lowkick":
        if attacker.weight_kg or defender.weight_kg:
            power = _low_kick_power(defender.weight_kg)
        else:
            power = 50
            notes.append("Low Kick: target weight unknown, assumed 50 BP")
    elif key == "magnitude":
        power = ctx.magnitude_power or 71   # expected value of the table
        if ctx.magnitude_power is None:
            notes.append("Magnitude: averaged power 71")
    elif key == "hiddenpower":
        power = ctx.hidden_power_power or 50
        if ctx.hidden_power_power is None:
            notes.append("Hidden Power: IVs unknown, assumed 50 BP")
    elif key == "spitup":
        power = 100 * max(1, attacker.stockpile)
    elif key in ("rollout", "iceball"):
        power = (move.power or 30) * (2 ** min(attacker.consecutive_hits, 4))
    elif key == "furycutter":
        power = (move.power or 10) * (2 ** min(attacker.consecutive_hits, 4))
    elif key == "triplekick":
        power = 10
        notes.append("Triple Kick: per-strike power, hits 1-3 times with rising power")
    elif key in VARIABLE_POWER_MOVES:
        notes.append(f"{move.name}: variable power not modelled, using {power}")

    # Pinch abilities boost base power in Gen 3 (not the attack stat).
    pinch_type = PINCH_ABILITIES.get(attacker.ability.lower())
    if pinch_type and move.type == pinch_type and attacker.hp_fraction <= 1 / 3:
        power = power * 3 // 2
        notes.append(f"{attacker.ability} active")

    # Charge doubles the next Electric move.
    if attacker.charged and move.type == "Electric":
        power *= 2

    # Type-boosting held items: 1.1x.
    boost_type = TYPE_BOOST_ITEMS.get(attacker.item.lower())
    if boost_type and move.type == boost_type:
        power = power * 110 // 100

    if attacker.item.lower() == "sea incense" and move.type == "Water":
        power = power * 105 // 100

    return max(1, power), notes


def resolve_attack_defense(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    critical: bool,
) -> tuple[int, int, list[str]]:
    """Effective A and D, with Gen 3 crit stage rules applied."""
    notes: list[str] = []
    if move.is_physical:
        atk_key, def_key = "atk", "def"
    else:
        atk_key, def_key = "spa", "spd"

    # A crit ignores the attacker's negative stages and the defender's positive ones.
    attack = attacker.stat(atk_key, ignore_negative=critical)
    defense = defender.stat(def_key, ignore_positive=critical)

    ability = attacker.ability.lower()
    if ability in ("huge power", "pure power") and move.is_physical:
        attack *= 2
    elif ability == "hustle" and move.is_physical:
        attack = attack * 3 // 2
    elif ability == "guts" and attacker.status and move.is_physical:
        attack = attack * 3 // 2
        notes.append("Guts active")

    item = attacker.item.lower()
    if item == "choice band" and move.is_physical:
        attack = attack * 3 // 2
    elif item == "thick club" and normalize(attacker.species) in ("cubone", "marowak"):
        attack *= 2
    elif item == "light ball" and normalize(attacker.species) == "pikachu" and move.is_special:
        attack *= 2

    if defender.ability.lower() == "marvel scale" and defender.status and move.is_physical:
        defense = defense * 3 // 2

    return max(1, attack), max(1, defense), notes


def type_effectiveness(
    data: GameData, move: Move, defender: Pokemon, ignore_immunity: bool = False
) -> float:
    if move.key in TYPELESS_MOVES:
        return 1.0
    multiplier = 1.0
    for defend_type in defender.types:
        multiplier *= data.effectiveness(move.type, defend_type)

    ability = defender.ability.lower()
    if ability == "levitate" and move.type == "Ground":
        multiplier = 0.0
    elif ability == "wonder guard" and multiplier <= 1.0:
        multiplier = 0.0
    elif ability == "volt absorb" and move.type == "Electric":
        multiplier = 0.0
    elif ability == "water absorb" and move.type == "Water":
        multiplier = 0.0
    elif ability == "flash fire" and move.type == "Fire":
        multiplier = 0.0
    elif ability == "thick fat" and move.type in ("Fire", "Ice"):
        multiplier *= 0.5
    return multiplier


# --------------------------------------------------------------------------
# The formula
# --------------------------------------------------------------------------


def calculate(
    data: GameData,
    attacker: Pokemon,
    defender: Pokemon,
    move: Move | str,
    fieldstate: Field | None = None,
    ctx: Context | None = None,
) -> DamageResult:
    """Run the Gen 3 damage formula and return every possible roll."""
    if isinstance(move, str):
        move = data.move(move)
    fieldstate = fieldstate or Field()
    ctx = ctx or Context()
    notes: list[str] = []

    effectiveness = type_effectiveness(data, move, defender)

    if move.is_status:
        return DamageResult(
            move=move, rolls=(0,) * 16, effectiveness=effectiveness, stab=False,
            category=move.category, power=0, attack=0, defense=0,
            critical=ctx.critical, defender_hp=defender.current_hp,
            notes=("status move, deals no damage",),
        )

    if effectiveness == 0:
        return DamageResult(
            move=move, rolls=(0,) * 16, effectiveness=0.0, stab=False,
            category=move.category, power=0, attack=0, defense=0,
            critical=ctx.critical, defender_hp=defender.current_hp,
            notes=("immune",),
        )

    # Moves that skip the formula entirely.
    if move.key in FIXED_DAMAGE_MOVES:
        return _fixed_damage(move, attacker, defender, effectiveness, ctx)
    if move.key in OHKO_MOVES:
        rolls = (defender.current_hp,) * 16
        note = "OHKO move" if defender.level <= attacker.level else "fails: target outlevels user"
        if defender.level > attacker.level:
            rolls = (0,) * 16
        return DamageResult(
            move=move, rolls=rolls, effectiveness=effectiveness, stab=False,
            category=move.category, power=0, attack=0, defense=0,
            critical=False, defender_hp=defender.current_hp, notes=(note,),
        )

    power, power_notes = resolve_power(data, move, attacker, defender, fieldstate, ctx)
    notes.extend(power_notes)
    attack, defense, ad_notes = resolve_attack_defense(move, attacker, defender, ctx.critical)
    notes.extend(ad_notes)

    # ---- base damage, truncating at each division ----
    damage = (2 * attacker.level) // 5 + 2
    damage = damage * power * attack
    damage = damage // defense
    damage = damage // 50

    # Burn halves physical damage unless the attacker has Guts.
    if (
        attacker.status == "burn"
        and move.is_physical
        and attacker.ability.lower() != "guts"
    ):
        damage //= 2
        notes.append("burned")

    # Screens do not apply to critical hits.
    if not ctx.critical:
        screened = (move.is_physical and fieldstate.reflect) or (
            move.is_special and fieldstate.light_screen
        )
        if screened:
            if fieldstate.is_double:
                damage = damage * 2 // 3
            else:
                damage //= 2
            notes.append("reflect" if move.is_physical else "light screen")

    if ctx.spread_move and fieldstate.is_double:
        damage //= 2

    # A physical move that has been reduced to 0 here floors at 1.
    if damage == 0 and move.is_physical:
        damage = 1

    weather = "" if fieldstate.cloud_nine else fieldstate.weather
    if weather == "rain":
        if move.type == "Water":
            damage = damage * 3 // 2
        elif move.type == "Fire":
            damage //= 2
    elif weather == "sun":
        if move.type == "Fire":
            damage = damage * 3 // 2
        elif move.type == "Water":
            damage //= 2
    if weather and weather != "sun" and move.key == "solarbeam":
        damage //= 2

    if attacker.flash_fire_active and move.type == "Fire":
        damage = damage * 3 // 2
        notes.append("Flash Fire boosted")

    damage += 2

    # ---- multiplicative stage ----
    if move.key == "spitup":
        damage *= max(1, attacker.stockpile)

    if ctx.critical:
        damage *= 2

    if _double_damage_applies(move, ctx):
        damage *= 2
        notes.append("double damage condition met")

    if ctx.helping_hand:
        damage = damage * 3 // 2

    stab = move.type in attacker.types
    if stab:
        damage = damage * 3 // 2

    # Type effectiveness is applied one defending type at a time, truncating.
    for defend_type in defender.types:
        multiplier = data.effectiveness(move.type, defend_type)
        if move.key in TYPELESS_MOVES:
            continue
        if multiplier == 2:
            damage *= 2
        elif multiplier == 0.5:
            damage //= 2
        elif multiplier == 0:
            damage = 0
    if defender.ability.lower() == "thick fat" and move.type in ("Fire", "Ice"):
        damage //= 2

    # ---- the 85-100 random factor ----
    rolls = []
    for factor in range(85, 101):
        roll = damage * factor // 100
        if roll < 1 and effectiveness > 0:
            roll = 1
        rolls.append(roll)

    return DamageResult(
        move=move,
        rolls=tuple(rolls),
        effectiveness=effectiveness,
        stab=stab,
        category=move.category,
        power=power,
        attack=attack,
        defense=defense,
        critical=ctx.critical,
        defender_hp=defender.current_hp,
        notes=tuple(notes),
    )


def _double_damage_applies(move: Move, ctx: Context) -> bool:
    if ctx.double_damage:
        return True
    key = move.key
    if key == "pursuit" and ctx.target_switching:
        return True
    if key == "smellingsalts" and ctx.target_paralyzed:
        return True
    if key == "revenge" and ctx.damaged_this_turn:
        return True
    return False


def _fixed_damage(
    move: Move, attacker: Pokemon, defender: Pokemon, effectiveness: float, ctx: Context
) -> DamageResult:
    key = move.key
    if key in ("seismictoss", "nightshade"):
        rolls = (attacker.level,) * 16
    elif key == "dragonrage":
        rolls = (40,) * 16
    elif key == "sonicboom":
        rolls = (20,) * 16
    elif key == "superfang":
        rolls = (max(1, defender.current_hp // 2),) * 16
    elif key == "psywave":
        # 0.5x to 1.5x the user's level, in 1/10 increments.
        rolls = tuple(max(1, attacker.level * n // 100) for n in range(50, 151, 10))
    else:
        rolls = (0,) * 16
    return DamageResult(
        move=move, rolls=tuple(sorted(rolls)), effectiveness=effectiveness, stab=False,
        category=move.category, power=0, attack=0, defense=0, critical=False,
        defender_hp=defender.current_hp, notes=("fixed damage move",),
    )


# --------------------------------------------------------------------------
# Helpers for the AI layer
# --------------------------------------------------------------------------


def critical_hit_rate(move: Move, attacker: Pokemon, focus_energy: bool = False) -> float:
    """Gen 3 critical hit probability."""
    if attacker.ability.lower() in ("battle armor", "shell armor"):
        return 0.0
    stage = 0
    if move.key in HIGH_CRIT_MOVES:
        stage += 1
    if focus_energy:
        stage += 2          # Focus Energy is +2 stages in Gen 3
    if attacker.item.lower() in ("scope lens", "razor claw"):
        stage += 1
    if attacker.item.lower() == "lucky punch" and normalize(attacker.species) == "chansey":
        stage += 2
    if attacker.item.lower() == "stick" and normalize(attacker.species) == "farfetchd":
        stage += 2
    return {0: 1 / 16, 1: 1 / 8, 2: 1 / 4, 3: 1 / 3}.get(min(stage, 4), 1 / 2)


def accuracy_chance(move: Move, attacker: Pokemon, defender: Pokemon) -> float:
    """Probability the move connects, accounting for accuracy/evasion stages."""
    if move.never_misses:
        return 1.0
    stage = max(-6, min(6, attacker.stages.get("acc", 0) - defender.stages.get("eva", 0)))
    if stage >= 0:
        modifier = (3 + stage) / 3
    else:
        modifier = 3 / (3 - stage)
    return min(1.0, (move.accuracy / 100) * modifier)


def expected_damage(
    data: GameData,
    attacker: Pokemon,
    defender: Pokemon,
    move: Move | str,
    fieldstate: Field | None = None,
    ctx: Context | None = None,
) -> float:
    """Average damage per turn, weighted by accuracy and crit rate.

    This is the number to rank moves by -- it folds in the chance of missing and
    the chance of critting, which raw min/max rolls do not.
    """
    if isinstance(move, str):
        move = data.move(move)
    ctx = ctx or Context()

    normal_ctx = Context(**{**ctx.__dict__, "critical": False})
    crit_ctx = Context(**{**ctx.__dict__, "critical": True})

    normal = calculate(data, attacker, defender, move, fieldstate, normal_ctx)
    crit = calculate(data, attacker, defender, move, fieldstate, crit_ctx)

    crit_rate = critical_hit_rate(move, attacker)
    hit_rate = accuracy_chance(move, attacker, defender)
    blended = normal.average_damage * (1 - crit_rate) + crit.average_damage * crit_rate
    return blended * hit_rate


def rank_moves(
    data: GameData,
    attacker: Pokemon,
    defender: Pokemon,
    moves: Iterable[Move | str],
    fieldstate: Field | None = None,
    ctx: Context | None = None,
) -> list[tuple[Move, float, DamageResult]]:
    """Order the attacker's moves by expected damage, highest first."""
    ranked = []
    for entry in moves:
        move = data.move(entry) if isinstance(entry, str) else entry
        result = calculate(data, attacker, defender, move, fieldstate, ctx)
        ranked.append((move, expected_damage(data, attacker, defender, move, fieldstate, ctx), result))
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked


def pokemon_from_state(data: GameData, state: Mapping) -> Pokemon:
    """Build a Pokemon from a dict shaped like the emulator's battle state.

    Expects keys: species, level, hp, max_hp, stats (atk/def/spa/spd/spe) and
    optionally types, ability, item, status, stages. Types are filled in from
    the pokedex when absent.
    """
    types = state.get("types")
    if not types:
        types = data.species(state["species"])["types"]
    return Pokemon(
        species=state["species"],
        level=int(state["level"]),
        types=tuple(types),
        stats={k: int(v) for k, v in state["stats"].items()},
        max_hp=int(state["max_hp"]),
        current_hp=int(state.get("hp", state["max_hp"])),
        ability=state.get("ability", ""),
        item=state.get("item", ""),
        status=state.get("status", ""),
        stages=dict(state.get("stages", {})),
        friendship=int(state.get("friendship", 70)),
    )


if __name__ == "__main__":
    game = GameData.load()

    ivysaur = Pokemon(
        species="IVYSAUR", level=17, types=("Grass", "Poison"),
        stats={"atk": 28, "def": 30, "spa": 36, "spd": 35, "spe": 33},
        max_hp=48, current_hp=48, ability="Overgrow",
    )
    pidgey = Pokemon(
        species="PIDGEY", level=5, types=("Normal", "Flying"),
        stats={"atk": 10, "def": 9, "spa": 9, "spd": 10, "spe": 10},
        max_hp=19, current_hp=19, ability="Keen Eye",
    )

    print(f"{ivysaur.species} Lv{ivysaur.level} vs {pidgey.species} Lv{pidgey.level}\n")
    for move, score, result in rank_moves(
        game, ivysaur, pidgey, ["VINE WHIP", "TACKLE", "LEECH SEED", "SLEEP POWDER"]
    ):
        print(f"  {score:6.2f} exp  {result.summary()}")

    print("\nIncoming:")
    for move, score, result in rank_moves(game, pidgey, ivysaur, ["TACKLE", "SAND-ATTACK"]):
        print(f"  {score:6.2f} exp  {result.summary()}")
