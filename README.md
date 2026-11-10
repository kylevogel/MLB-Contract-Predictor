# MLB Free-Agent Contract Predictor

Can a neural network predict the length and average annual value (AAV) of an MLB free-agent
or extension contract from a player's performance history? Final project for Artificial
Neural Networks and Deep Learning (Kyle Vogel, Alex Elias, Liam Scanlon).

Status: data collection and cleaning finished; baselines and model work next. See `DATA_SPEC.md`
for exactly what is in the dataset and `DECISIONS.md` for every data choice, who made it and the evidence.

## Data and credits
`data/processed/contracts.csv` has 1,904 post-rookie MLB contracts (free agents and extensions,
offseasons 2014-2025). Contract terms were collected from [Spotrac](https://www.spotrac.com),
checked against MLB.com, MLB Trade Rumors, Baseball-Reference, FanGraphs and news reports, and
joined to player IDs from the [Chadwick Bureau register](https://github.com/chadwickbureau/register).
The raw Spotrac pages are not in this repo (personal-use terms). The cleaned file is published
with the permission of our course instructor.

## AI use
Parts of the data pipeline were written with Claude (Anthropic). Mistakes the AI made and how we
caught them are logged in `DECISIONS.md`. Each PR states how AI was used.

## Repo layout
See `AGENTS.md`. Figures for the blog live in `blog_figures/`; this README will list the code
that generates each one.
