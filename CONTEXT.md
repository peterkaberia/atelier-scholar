# Atelier

An AI-powered academic research synthesizer: it plans database-specific search queries from a natural-language question, concurrently fetches candidate papers from PubMed, Europe PMC, OpenAlex, Semantic Scholar, Crossref, and arXiv, scores and deduplicates them (SPLADE similarity + citation count + recency + study-design evidence tier), extracts structured metadata per paper via an LLM, and synthesizes a cited, abstract-style research summary — presented through a Dash single-page app with a checkpointed, resumable LangGraph pipeline underneath.

## Language

**Session**:
A user's research conversation thread, anchored to an initial topic. Accumulates Papers (via a many-to-many link) and Queries over its lifetime.
_Avoid_: Thread, Conversation

**Query**:
A single turn within a Session — either the initial search or a follow-up (chat or new search) — that produces its own Synthesis and its own set of cited Papers. A Session has many Queries; there is no single "final" synthesis for a Session, only the latest Query's.
_Avoid_: Turn, Message (reserve "message" for raw chat_history entries, not the persisted record)

**Paper**:
A single deduplicated academic record, identified globally (across all Sessions) by its Fingerprint. Holds permanent AI-extracted fields (population, methods, results, outcomes, sample size, etc.) computed once and reused forever. Not scoped to a source database — PubMed, Europe PMC, OpenAlex, Semantic Scholar, Crossref, and arXiv results all normalize into the same Paper shape. Study type (evidence tier - systematic review, RCT, cohort, etc.) is classified from title/abstract on the fly wherever it's needed rather than persisted, since it's cheap and deterministic to recompute.
_Avoid_: Record (that's the in-memory dataclass form of a Paper before/during persistence), Source, Document

**Fingerprint**:
The global deduplication key for a Paper: `doi:...` if a DOI exists, else `pmid:...`, else `pmcid:...`, else `arxiv:...`, else a `{source}:{source_id}` fallback. The same paper found via two different engines collapses to one Fingerprint and one Paper row.

**Synthesis**:
The cited markdown research summary produced for one Query, built only from the Papers that Query actually cites. For a SEARCH-routed Query it's written in real abstract-style prose (Background / Evidence Synthesis / Key Findings / Conclusion); for a CHAT-routed follow-up it adapts to what was actually asked — answering a question, rewriting the user's own pasted text with citations woven in, or writing a requested deliverable (an abstract, an introduction) in that genre's actual conventions, not the SEARCH template. There is exactly one Synthesis per Query.
_Avoid_: Treating "abstract" as a distinct artifact from Synthesis — abstract-style prose is one of Synthesis's possible shapes, not a separate thing.

**Atelier Meter**:
A weighted yes/no agreement verdict, generated (only for yes/no-shaped Queries) alongside a Synthesis. Each cited Paper's stance is weighted by its evidence tier and citation count rather than counted equally — inspired by Consensus.app's Consensus Meter.
_Avoid_: Consensus Meter (Atelier's own name for this artifact is Atelier Meter; the DB column and some internal identifiers still say `consensus_meter` for migration-safety reasons, but nothing user-facing should).

**Paper Summary**:
A topic-focused AI summary of a single Paper (not a synthesis across many), shown when a user opens a Paper's detail view. Cached per (Paper, Session) since the same Paper viewed again in the same Session always resolves to the same Session topic and therefore the same summary.

**Screening**:
The implicit relevance gate inside AI extraction: a Paper only becomes part of a Query's cited set if the LLM marks `is_relevant: true` for it against that Query's topic. Not a separate pipeline stage — it happens inline during extraction, and applies uniformly regardless of how the Paper entered the system (searched or uploaded).
