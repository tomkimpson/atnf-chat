# ATNF-Chat

A RAG-powered conversational interface for querying the ATNF Pulsar Catalogue.

## Overview

ATNF-Chat uses **Retrieval-Augmented Generation (RAG)** to provide accurate, grounded answers about pulsars. Unlike a simple LLM wrapper, every response is backed by real data retrieved from the ATNF Pulsar Catalogue.

**How it works:**
1. You ask a question in natural language (e.g., "What are the fastest spinning pulsars?")
2. The LLM translates your question into a structured query
3. The query executes against the live ATNF catalogue (~3800+ pulsars)
4. Results are returned to the LLM, which synthesizes a grounded response

This means you get the convenience of natural language with the accuracy of direct database queries - no hallucinated pulsar data.

## Features

- **Grounded Responses**: Every answer is backed by real catalogue data, not LLM training data
- **Natural Language Queries**: Ask questions in plain English
- **Live Data Access**: Queries the current ATNF catalogue via psrqpy
- **Interactive Visualizations**: P-Pdot diagrams, sky maps, and more with Plotly
- **Scientific Safety**: Explicit null handling and data quality warnings
- **Reproducible**: All queries exportable as validated Python code

## Why not just ask a frontier LLM?

Modern LLMs like ChatGPT and Claude have agent modes and code execution — so why use ATNF-Chat? Consider a concrete example:

> **"How many millisecond pulsars (i.e. pulsars with spin periods less than 30ms) have orbital periods less than 1 day?"**

**ATNF-Chat** translates this to a validated query (`P0 < 0.03 && PB < 1.0`) against the live catalogue and returns the exact answer: **174 pulsars** (ATNF v2.7.0, 4351 pulsars).

**A frontier LLM** without catalogue access will typically:

- **Hallucinate a number**, stated with false confidence
- **Cite stale literature**, e.g. ~70 confirmed "spider" pulsars from a 2019 survey or ~111 entries from the 2025 SpiderCat catalogue — both of which describe curated astrophysical classes, not the raw catalogue query result
- **Conflate categories**: "spider pulsars" (black widows and redbacks with confirmed companion ablation) are a *subset* of MSPs with PB < 1 day, but an LLM will often treat them as equivalent
- **Punt**, telling you to go query the catalogue yourself

Even with web search, the precise answer (174) does not appear in any published paper — it is a live database query whose result changes as new pulsars are discovered. This is the class of question where RAG over a structured catalogue provides value that a general-purpose LLM cannot replicate.



## License

MIT

## Acknowledgments

- [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/) - CSIRO
- [psrqpy](https://github.com/mattpitkin/psrqpy) - Python interface to the catalogue
- [Anthropic Claude](https://www.anthropic.com) - LLM powering the chat interface
