"""Is this party strong enough to beat that trainer yet?

The damage calculator answers "what does this move do right now". This answers
the question the player actually asks before walking into a gym: *should I go
in, or should I train first?*

It does that by running the calculator over every pairing of your party against
a stored enemy team and asking two different questions:

  coverage — for each of their Pokemon, does anything of yours beat it 1-on-1?
             This is the question that matters when you're willing to switch.
  sweep    — can ONE of your Pokemon walk through the whole team without
             fainting? HP carries between their Pokemon but theirs is restored,
             which is what makes a gym harder than the sum of its parts.

Both are decided on expected damage per turn (accuracy- and crit-weighted, from
damage_calc.expected_damage), turned into turns-to-KO, and compared with speed
breaking the tie. Nothing here is a simulation of the real battle: it assumes
best move every turn, no items, no status luck, no switching mid-fight. It is
deliberately a *lower* bound on a competent player, so "ready" means ready.

Enemy teams live in trainers.json, in exactly the shape GAME_STATE reports a
Pokemon - so recording a new one is a copy, not a transcription:

    python battle/matchup.py capture brock --name "Leader BROCK"

Usage:
    python battle/matchup.py list                # known trainers
    python battle/matchup.py brock               # assess the live party
    python battle/matchup.py brock --level 18    # what-if at a higher level
    python battle/matchup.py capture <id>        # record the foe you're fighting

    from matchup import Roster, assess
    report = assess(gameData, party, roster.team("brock"))
    report["verdict"]        # 'ready' | 'risky' | 'not_ready'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "mGBA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from damage_calc import (  # noqa: E402
    Context,
    DataError,
    Field,
    GameData,
    Pokemon,
    expected_damage,
)
from live_calc import to_pokemon  # noqa: E402

TRAINER_FILE = HERE / "trainers.json"

# A sweep that ends on fumes is not a plan, it's a coin flip - one crit or one
# unlucky roll and the run is over. Below this fraction of max HP we call the
# fight winnable but risky rather than ready.
COMFORTABLE_HP = 0.25

# Turns of daylight between KOing them and being KOed. A margin of 1 means a
# single miss erases the lead, so it counts as risky, not ready.
COMFORTABLE_MARGIN = 2

# Cap on the "train this many levels" search, so an unwinnable matchup returns
# "not with this party" instead of promising level 60.
MAX_LEVEL_SEARCH = 15


# --------------------------------------------------------------------------
# Combatants
# --------------------------------------------------------------------------


@dataclass
class Combatant:
    """A Pokemon plus the raw state dict it came from (which holds its moves)."""

    raw: dict
    mon: Pokemon

    @classmethod
    def build(cls, raw: dict, healed: bool = False) -> "Combatant":
        mon = to_pokemon(raw)
        if healed:
            mon.current_hp = mon.max_hp
        return cls(raw=raw, mon=mon)

    @property
    def name(self) -> str:
        nick = self.raw.get("nickname") or self.mon.species
        return nick if nick.upper() == self.mon.species.upper() else \
            f"{nick} ({self.mon.species})"

    @property
    def label(self) -> str:
        return f"Lv{self.mon.level} {self.name}"

    def moves(self) -> list:
        return [m["name"] for m in self.raw.get("moves", []) if m.get("name")]


def _effectiveSpeed(mon: Pokemon) -> int:
    speed = mon.stat("spe")
    if mon.status == "paralysis":
        speed //= 4
    return max(1, speed)


# --------------------------------------------------------------------------
# One pairing
# --------------------------------------------------------------------------


def bestAttack(data: GameData, attacker: Combatant, defender: Combatant):
    """(moveName, expectedDamagePerTurn) for the attacker's best damaging move."""
    best, bestScore = None, 0.0
    for name in attacker.moves():
        try:
            score = expected_damage(data, attacker.mon, defender.mon, name,
                                    Field(), Context())
        except DataError:
            continue        # a move that isn't in moves.json yet
        if score > bestScore:
            best, bestScore = name, score
    return best, bestScore


def _turns(hp: int, perTurn: float):
    """Turns to remove `hp`, or None if this attacker can never get there."""
    if perTurn <= 0:
        return None
    return max(1, math.ceil(hp / perTurn))


def duel(data: GameData, you: Combatant, foe: Combatant) -> dict:
    """Who wins a straight 1-on-1, and by how many turns."""
    yourMove, yourDps = bestAttack(data, you, foe)
    theirMove, theirDps = bestAttack(data, foe, you)

    toWin = _turns(foe.mon.current_hp, yourDps)
    toLose = _turns(you.mon.current_hp, theirDps)
    youFirst = _effectiveSpeed(you.mon) >= _effectiveSpeed(foe.mon)

    if toWin is None:
        wins = False
    elif toLose is None:
        wins = True
    else:
        # Moving first means you get to land the killing blow on the tying turn.
        wins = toWin <= toLose if youFirst else toWin < toLose

    return {
        "you": you.label, "foe": foe.label,
        "yourMove": yourMove, "yourDps": yourDps, "turnsToWin": toWin,
        "theirMove": theirMove, "theirDps": theirDps, "turnsToLose": toLose,
        "youFirst": youFirst, "wins": wins,
        "margin": (toLose - toWin) if (toWin is not None and toLose is not None)
                  else (99 if toWin is not None else -99),
    }


def sweep(data: GameData, you: Combatant, team: list) -> dict:
    """Send one Pokemon through the whole team and see if it comes out alive.

    Their HP resets between fights and yours doesn't - which is the whole reason
    a gym is harder than its hardest single Pokemon.
    """
    hp = you.mon.current_hp
    maxHp = you.mon.max_hp
    log = []
    for foe in team:
        working = Combatant(raw=you.raw, mon=_clone(you.mon, hp))
        result = duel(data, working, foe)
        if result["turnsToWin"] is None:
            log.append(f"can't damage {foe.label}")
            return {"survives": False, "hpLeft": hp, "failedOn": foe.label,
                    "log": log}

        # Moving first saves you exactly one turn of incoming damage.
        turnsTaken = result["turnsToWin"] - (1 if result["youFirst"] else 0)
        damage = result["theirDps"] * max(0, turnsTaken)
        hp -= damage
        log.append(f"{foe.label}: {result['turnsToWin']} turn(s) with "
                   f"{result['yourMove']}, you take ~{damage:.0f} -> "
                   f"{max(0, hp):.0f}/{maxHp} HP")
        if hp <= 0:
            return {"survives": False, "hpLeft": 0, "failedOn": foe.label,
                    "log": log}
    return {"survives": True, "hpLeft": hp, "failedOn": None, "log": log}


def _clone(mon: Pokemon, hp: float) -> Pokemon:
    """Copy a Pokemon at a different current HP (Pokemon is a mutable dataclass)."""
    clone = Pokemon(
        species=mon.species, level=mon.level, types=mon.types,
        stats=dict(mon.stats), max_hp=mon.max_hp,
        current_hp=max(0, int(round(hp))), ability=mon.ability, item=mon.item,
        status=mon.status, stages=dict(mon.stages), friendship=mon.friendship)
    return clone


# --------------------------------------------------------------------------
# Whole-team assessment
# --------------------------------------------------------------------------


def assess(data: GameData, party: list, team: list,
           healed: bool = True) -> dict:
    """Verdict on a party against an enemy team.

    `party` and `team` are lists of GAME_STATE-shaped Pokemon dicts. `healed`
    assumes a Pokemon Center visit first, which is the right assumption for
    "should I go and fight this" - the answer shouldn't flip because you're
    two potions down right now.
    """
    yours = [Combatant.build(p, healed=healed) for p in party
             if p.get("hp", 0) > 0 or healed]
    theirs = [Combatant.build(e, healed=True) for e in team]

    if not yours:
        return {"verdict": "not_ready", "summary": "you have no Pokemon",
                "coverage": [], "sweepers": [], "levels": None}
    if not theirs:
        return {"verdict": "unknown", "summary": "no team recorded for that "
                "trainer", "coverage": [], "sweepers": [], "levels": None}

    # Coverage: their Pokemon, and your best answer to each.
    coverage = []
    for foe in theirs:
        duels = [duel(data, you, foe) for you in yours]
        winners = [d for d in duels if d["wins"]]
        best = max(winners or duels, key=lambda d: d["margin"])
        coverage.append({"foe": foe.label, "covered": bool(winners),
                         "best": best,
                         "answers": [d["you"] for d in winners]})

    # Sweep: can one Pokemon do the whole job?
    sweepers = []
    for you in yours:
        run = sweep(data, you, theirs)
        run["who"] = you.label
        run["hpFraction"] = run["hpLeft"] / max(1, you.mon.max_hp)
        sweepers.append(run)
    sweepers.sort(key=lambda r: (not r["survives"], -r["hpFraction"]))

    verdict, summary = _verdict(coverage, sweepers, yours, theirs)
    return {"verdict": verdict, "summary": summary, "coverage": coverage,
            "sweepers": sweepers, "healed": healed,
            "yourBestLevel": max(y.mon.level for y in yours),
            "theirAceLevel": max(t.mon.level for t in theirs)}


def _verdict(coverage, sweepers, yours, theirs):
    uncovered = [c["foe"] for c in coverage if not c["covered"]]
    best = sweepers[0] if sweepers else None
    aceLevel = max(t.mon.level for t in theirs)
    yourLevel = max(y.mon.level for y in yours)

    if uncovered and not (best and best["survives"]):
        return "not_ready", (
            f"nothing in your party beats {', '.join(uncovered)} one-on-one "
            f"(their ace is Lv{aceLevel}, your best is Lv{yourLevel})")

    if best and best["survives"] and best["hpFraction"] >= COMFORTABLE_HP:
        thin = [c["foe"] for c in coverage
                if c["covered"] and c["best"]["margin"] < COMFORTABLE_MARGIN]
        if thin:
            return "risky", (
                f"{best['who']} can win alone, but {', '.join(thin)} is close - "
                f"one miss or crit could flip it")
        return "ready", (f"{best['who']} beats the whole team and ends around "
                         f"{best['hpFraction']:.0%} HP")

    if best and best["survives"]:
        return "risky", (f"{best['who']} can just barely sweep, ending near "
                         f"{best['hpFraction']:.0%} HP - heal up and bring "
                         f"potions, or train a little more")

    if not uncovered:
        return "risky", ("no single Pokemon can sweep, but every one of theirs "
                         "loses to something of yours - you will have to switch "
                         "and you will take chip damage")

    return "not_ready", (f"{', '.join(uncovered)} beats everything you have "
                         f"(their ace is Lv{aceLevel}, your best is Lv{yourLevel})")


# --------------------------------------------------------------------------
# "How much more training?"
# --------------------------------------------------------------------------


def projectToLevel(raw: dict, level: int) -> dict:
    """Estimate what a Pokemon's stats look like at a different level.

    Gen 3 stats are (2*Base + IV + EV/4) * L / 100 + 5 (+ level and 10 for HP),
    so the level-dependent part scales linearly and everything unknown about the
    Pokemon - its base stats, IVs, EVs - cancels out of the ratio. Nature is the
    one thing this smears, since it multiplies the whole stat rather than the
    scaled part, so treat the result as an estimate and not a promise.
    """
    out = dict(raw)
    old = max(1, int(raw.get("level", 1)))
    if level == old:
        return out
    ratio = level / old

    for key in ("attack", "defense", "sp_attack", "sp_defense", "speed"):
        if key in raw:
            out[key] = max(1, int(round((raw[key] - 5) * ratio + 5)))
    if "max_hp" in raw:
        base = raw["max_hp"] - old - 10
        out["max_hp"] = max(1, int(round(base * ratio + level + 10)))
        out["hp"] = out["max_hp"]
    out["level"] = level
    return out


def levelsNeeded(data: GameData, party: list, team: list,
                 limit: int = MAX_LEVEL_SEARCH):
    """Levels of training until the verdict turns 'ready', or None if never.

    Only the strongest Pokemon is levelled in the projection: that is what
    actually happens when you go and grind, and it keeps the answer honest
    rather than assuming a whole rebuilt team.
    """
    if not party:
        return None
    lead = max(party, key=lambda p: p.get("level", 0))
    others = [p for p in party if p is not lead]
    for extra in range(1, limit + 1):
        projected = [projectToLevel(lead, lead.get("level", 1) + extra)] + others
        if assess(data, projected, team)["verdict"] == "ready":
            return extra
    return None


# --------------------------------------------------------------------------
# The stored roster
# --------------------------------------------------------------------------


@dataclass
class Roster:
    """trainers.json: known enemy teams, in GAME_STATE shape."""

    path: Path = TRAINER_FILE
    data: dict = dc_field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = TRAINER_FILE) -> "Roster":
        raw = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"matchup: {path.name} is not valid JSON ({exc}); "
                      f"treating the roster as empty.")
        return cls(path=path, data=raw)

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2) + "\n",
                             encoding="utf-8")

    def ids(self) -> list:
        return sorted(self.data)

    def get(self, trainerId: str):
        key = (trainerId or "").strip().lower()
        if key in self.data:
            return self.data[key]
        for known in self.data:
            if known.lower() == key or known.lower().replace("_", " ") == key:
                return self.data[known]
        return None

    def team(self, trainerId: str) -> list:
        entry = self.get(trainerId)
        return list(entry.get("party", [])) if entry else []

    def record(self, trainerId: str, party: list, name: str = "",
               where: str = "", note: str = ""):
        entry = self.data.setdefault(trainerId.strip().lower(), {})
        entry.update({k: v for k, v in
                      (("name", name), ("where", where), ("note", note)) if v})
        entry["party"] = party
        self.save()
        return entry


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

VERDICT_LABEL = {"ready": "READY", "risky": "RISKY", "not_ready": "NOT READY",
                 "unknown": "UNKNOWN"}


def summarize(report: dict, trainerName: str = "", levels=None) -> str:
    """One or two lines - what goes in the model's prompt every turn."""
    label = VERDICT_LABEL.get(report["verdict"], report["verdict"])
    who = f" vs {trainerName}" if trainerName else ""
    line = f"READINESS{who}: {label} - {report['summary']}."
    if levels:
        line += (f" Estimate: about {levels} more level(s) on your strongest "
                 f"Pokemon would make this comfortable.")
    elif levels is None and report["verdict"] == "not_ready":
        # The projection scales stats, not movesets, and a level-up move is
        # exactly what turns a losing gym fight around (a Bulbasaur that can't
        # scratch ONIX at Lv7 learns VINE WHIP at Lv13 and 4x's it). Say so,
        # rather than claiming training is hopeless.
        line += (f" Training for {MAX_LEVEL_SEARCH} levels doesn't fix this on "
                 f"its own - though this estimate can't see moves you'd learn "
                 f"along the way, so a new move may change it. A Pokemon with a "
                 f"better type matchup would help.")
    return line


def describe(report: dict, trainerName: str = "", levels=None) -> str:
    """The full breakdown, for a human or for an on-demand `check` command."""
    lines = [summarize(report, trainerName, levels), ""]

    lines.append("THEIR TEAM, AND YOUR BEST ANSWER")
    for entry in report["coverage"]:
        best = entry["best"]
        mark = "ok  " if entry["covered"] else "LOSS"
        if best["turnsToWin"] is None:
            detail = "you have no move that damages it"
        else:
            detail = (f"{best['you']} needs {best['turnsToWin']} turn(s) with "
                      f"{best['yourMove']}; it needs "
                      f"{best['turnsToLose'] if best['turnsToLose'] else 'never'}"
                      f" with {best['theirMove'] or '-'}"
                      f" ({'you move' if best['youFirst'] else 'it moves'} first)")
        lines.append(f"  [{mark}] {entry['foe']}: {detail}")

    lines.append("")
    lines.append("RUNNING THE WHOLE TEAM WITH ONE POKEMON")
    for run in report["sweepers"][:3]:
        head = ("survives" if run["survives"]
                else f"faints to {run['failedOn']}")
        lines.append(f"  {run['who']}: {head} "
                     f"({run['hpFraction']:.0%} HP left)")
        for step in run["log"]:
            lines.append(f"      {step}")
    if report.get("healed"):
        lines.append("")
        lines.append("  (assumes you heal at a Pokemon Center first, and that "
                     "you pick the best move every turn with no items)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _liveState(host: str, port: int):
    from mgba_client import MGBAClient
    with MGBAClient(host=host, port=port) as client:
        return client.game_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trainer", nargs="?", default="list",
                        help="trainer id to assess, or 'list', or 'capture'")
    parser.add_argument("capture_id", nargs="?", default=None,
                        help="with 'capture': the id to store the foe under")
    parser.add_argument("--name", default="", help="display name when capturing")
    parser.add_argument("--where", default="", help="location when capturing")
    parser.add_argument("--note", default="", help="free note when capturing")
    parser.add_argument("--level", type=int, default=None,
                        help="what-if: project your strongest Pokemon to this level")
    parser.add_argument("--hurt", action="store_true",
                        help="assess at current HP instead of assuming a heal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    args = parser.parse_args()

    roster = Roster.load()

    if args.trainer == "list":
        if not roster.ids():
            print(f"No trainers recorded yet in {TRAINER_FILE.name}.")
            print("Start a fight and run: python battle/matchup.py capture <id>")
            return 0
        print(f"Known trainers in {TRAINER_FILE.name}:")
        for tid in roster.ids():
            entry = roster.data[tid]
            party = entry.get("party", [])
            team = ", ".join(f"Lv{p['level']} {p['species']}" for p in party)
            print(f"  {tid:<12} {entry.get('name', ''):<16} "
                  f"{entry.get('where', ''):<22} {team}")
        return 0

    if args.trainer == "capture":
        if not args.capture_id:
            print("Usage: python battle/matchup.py capture <id> [--name ...]")
            return 1
        state = _liveState(args.host, args.port)
        enemy = state.get("enemy_party") or []
        if not enemy:
            print("No enemy party in the game state - are you in a battle?")
            return 1
        entry = roster.record(args.capture_id, enemy, name=args.name,
                              where=args.where, note=args.note)
        print(f"Recorded {len(entry['party'])} Pokemon under "
              f"'{args.capture_id.lower()}':")
        for p in entry["party"]:
            print(f"  Lv{p['level']} {p['species']}  HP {p['max_hp']}  "
                  f"{', '.join(m['name'] for m in p.get('moves', []))}")
        return 0

    team = roster.team(args.trainer)
    if not team:
        print(f"No team recorded for {args.trainer!r}. "
              f"Known: {', '.join(roster.ids()) or '(none)'}")
        return 1

    state = _liveState(args.host, args.port)
    party = state.get("party") or []
    if not party:
        print("Your party is empty.")
        return 1
    if args.level is not None:
        lead = max(party, key=lambda p: p.get("level", 0))
        party = [projectToLevel(lead, args.level)] + [p for p in party if p is not lead]

    data = GameData.load()
    report = assess(data, party, team, healed=not args.hurt)
    entry = roster.get(args.trainer) or {}
    levels = (levelsNeeded(data, party, team)
              if report["verdict"] != "ready" else None)
    print()
    print(describe(report, entry.get("name") or args.trainer, levels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
