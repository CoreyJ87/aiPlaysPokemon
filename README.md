# AI Plays Pokemon

2026 has become the year of AI agents, with people publishing guides for controlling everything from sprinkler systems to crypto wallets with agentic models.  However, these guides tend to gloss over why and how you should build your tooling and harnesses, and instead recommend making everything an MCP server and to just throw data at increasingly more expensive models until you find one that works.  This solution works, but leads to two bad habits: everyone uses expensive cloud models to solve everything, and everyone removes all their privacy by exposing all of their personal data directly to these remote models.  It’s also not sustainable; no one wants to pay money every time they look at a light switch or read an email.  

We in the AI Village would like to provide a better example in this space, showing off both the logic of why and what to expose to a model, as well as how different harness and tooling designs can allow local edge models to work at the same level as frontier cloud platforms.  To do this, we decided to use the classic video games of Pokemon Fire Red and Leaf Green.  Both because it is a simple game that everyone can understand, and the author of this repo is a massive fan of the series.  

## Background

Pokemon actually has a long, if unintentional, history in the programming space.  While MissingNo was a glitch, it’s a great lesson in memory management.  And Twitch Plays Pokemon exposed millions of people to just how powerful and silly APIs can get.  Last year, both Anthropic and OpenAI even used the games as an informal benchmark, doing live streams with both of their models successfully completing the game.  Thus, there’s not only a lot of history in this space, but even a good comparison to grade ourselves against.  

## The Challenge and Set Up

Our goal is to defeat the Elite Four and become the Pokemon Champion in either Fire Red or Leaf Green using only locally hosted models and tools.  The goal is to see both how far we get and if we can do better than Anthropic and OpenAI.  

- We’ll be using Ollama as our LLM hosting environment, since it’s pretty easy to set up across multiple operating systems   
- Gemma4 will be the family of models we’ll use, but all the tools should be model-agnostic
  - We picked Gemma 4 because it has a wide selection of sizes for local models that can run on both Raspberry Pis and laptops 
- The model must be involved in playing the game
  - While this is a TAS, we don’t want everything to be hard-coded

