# Battle Tools

Battles in Pokemon are another great example of why you would create a tool for a model.  While there are numerous stratigies, esspically in competitive play, battles themselves are basically giant math formulas.  Rather than asking a model to try and generate or recall this every turn, it's far more memory efiecent to move as much as possible into external files and programs.  For example, [pokedex](./pokedex.json) contains the Generation III pokedex, [typeChart](./typeChart.json) contains the Generation III type chart, and [moves](./moves.json) contains every move in Fire Red and Leaf Green, and their effects.  

## Getting Data

Like with [location tracking](../locationTracking/README.md), the original goal was to try and use Optical Character Recognition(OCR) to do detections and processing of how the battle waas going.  This quickly failed, the gameboy advance has a very small screen, and thus all the games on it play natively at a small resolution.  While the game is pretty readable by human standards, OCR models would get confused over relatively import details, like HP or the name of the opposing Pokemon.  In theory this could have been fixed by pulling out all the alphabet sprites, and training an OCR model on those, but for DEFCON the faster and easier thing to do was to just fall back on using game state.  

Getting the data from the game state of the emulator actually solves a lot of problems down the line, since the game knows not only if we are fighting and if so what monster, but also all of the opponents stats.  This gives us a very accurate damage calculation, which is about the only win we get.  Movesets in the early Pokemon games were not great, esspically for your starters.  Most really powerful moves are gated behind a TM, and as of a week before DEFCON, I plan to leave item management to the AI as a hopefully ammusing comparision to all of the tools being designed for the rest of the game.  So who knows if anything will ever learn hyper beam.  As such, we actually need the battle AI to be pretty smart, and able to plan around the use of status effect moves in addtion to damaging ones.

## Am I Strong Enough Yet?

The damage calculator answers "what does this move do right now", which is the wrong question when you are standing outside a gym.  [matchup.py](./matchup.py) answers the one the player actually asks - *should I go in, or should I train first?* - by running the calculator over every pairing of your party against a stored enemy team.

It asks two different questions, because they have different answers.  **Coverage** is per Pokemon: for each of theirs, does anything of yours beat it one-on-one?  That is the question that matters if you are happy to switch.  **Sweep** is the harder one: can a single Pokemon walk through the whole team without fainting?  Their Pokemon are restored between fights and yours is not, which is exactly what makes a gym harder than its hardest individual Pokemon.  Both are decided on expected damage per turn (accuracy- and crit-weighted), turned into turns-to-KO, with speed breaking the tie.

Nothing in there is a simulation of the real fight.  It assumes the best move every turn, no items, no status luck and no switching mid-fight, which makes it a deliberate *under*estimate of a competent player - so a `ready` verdict means ready.  When the verdict is not ready it also estimates how many levels of training would fix it, by scaling stats with the Gen III stat formula.  That estimate cannot see moves learned on the way up, which matters more than it sounds: a Lv7 BULBASAUR genuinely cannot beat Brock, right up until it learns VINE WHIP at Lv13 and hits for 4x.

Enemy teams live in [trainers.json](./trainers.json), stored in exactly the shape `GAME_STATE` reports a Pokemon, so recording a new one is a copy rather than a transcription:

```
python battle/matchup.py capture brock --name "Leader BROCK" --where "Pewter City Gym"
python battle/matchup.py brock             # assess your live party
python battle/matchup.py brock --level 18  # what-if, at a higher level
python battle/matchup.py list
```

This is also what makes "train until you are strong enough" a checkable objective rather than a vibe - see [objectives.py](../objectives.py), where a `ready_for` condition asks this module and advances the player when it says yes.  