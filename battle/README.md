# Battle Tools

Battles in Pokemon are another great example of why you would create a tool for a model.  While there are numerous stratigies, esspically in competitive play, battles themselves are basically giant math formulas.  Rather than asking a model to try and generate or recall this every turn, it's far more memory efiecent to move as much as possible into external files and programs.  For example, [pokedex](./pokedex.json) contains the Generation III pokedex, [typeChart](./typeChart.json) contains the Generation III type chart, and [moves](./moves.json) contains every move in Fire Red and Leaf Green, and their effects.  

## Getting Data

Like with [location tracking](../locationTracking/README.md), the original goal was to try and use Optical Character Recognition(OCR) to do detections and processing of how the battle waas going.  This quickly failed, the gameboy advance has a very small screen, and thus all the games on it play natively at a small resolution.  While the game is pretty readable by human standards, OCR models would get confused over relatively import details, like HP or the name of the opposing Pokemon.  In theory this could have been fixed by pulling out all the alphabet sprites, and training an OCR model on those, but for DEFCON the faster and easier thing to do was to just fall back on using game state.  

Getting the data from the game state of the emulator actually solves a lot of problems down the line, since the game knows not only if we are fighting and if so what monster, but also all of the opponents stats.  This gives us a very accurate damage calculation, which is about the only win we get.  Movesets in the early Pokemon games were not great, esspically for your starters.  Most really powerful moves are gated behind a TM, and as of a week before DEFCON, I plan to leave item management to the AI as a hopefully ammusing comparision to all of the tools being designed for the rest of the game.  So who knows if anything will ever learn hyper beam.  As such, we actually need the battle AI to be pretty smart, and able to plan around the use of status effect moves in addtion to damaging ones.

## Damage Calculation

There are a lot of Pokemon damage calculators tailored made for everything from nuzlocks to competitive play.  However, because the goal of this project was to run live at DEFCON, where WiFi is fickle at best, I wanted to make sure the AI player had their own local damage calculator to use.  For this initial pass it's just doing the basics of how many turns will it take for move X to knock out Y, vs how soon can they knock out your own pokemon.  Eventually I want to better represent stalling tactics, like starting a long fight with leech seed to recover HP over multiple turns, or using sleep powder to stall a stronger opponent, but that will likely happen after the conference.  


### Damage Formula

TODO:  Grab the Gen III Damage calc and explain it here along with how we get stuff from game state

## Battle Planning

The other nice thing about having the calculator as a local script is we can build around it for being able to tell the player AI when they are actually strong enough to likely win against the next gym battle or other important encounter.  [Trainers.json](./trainers.json) has a list of pulled trainers from the game that represent key encounters, while [matchup](./matchup.py) handles calculating how good or bad the current players team does against these extracted encounters.  This then gives us a way to inform the player AI that they need to go train more by either fighting trainers or wild pokemon.  

One thing we don't do that I would eventually like to is to have this toolset also recommend moves when a pokemon learns a new move or gains a new TM.  But that's pretty far off for now.  

