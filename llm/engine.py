"""
Atelier AI Engine Module

This module houses the core Artificial Intelligence orchestrator for the application.
It uses the Factory Pattern to dynamically instantiate LangChain models on the fly
with task-specific temperatures, ensuring strictness for data extraction and 
creativity for synthesis and chat.
"""

import re
import json
import logging
import warnings
from typing import Dict, List, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import resolve_key, STUDY_TYPE_WEIGHTS
from core.exceptions import AIOutputError, AIProviderError, AtelierAIError
from core.utils import classify_study_type, citation_bonus, is_yes_no_question
from llm.utils import _common_llm_params, _is_set, resolve_model_config, get_model_choices

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==========================================
# 1. STANDALONE HELPERS
# ==========================================

def _get_val(obj: Any, key: str, default: Any = "") -> Any:
    """
    Safely extracts a value from an object, accommodating both Dictionary 
    formats (from SQLite) and Class Instances (from live requests).
    
    Args:
        obj (Any): The object to inspect.
        key (str): The attribute or dictionary key to retrieve.
        default (Any): The fallback value if the key does not exist.
        
    Returns:
        Any: The extracted value or the default.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_synthesis_entry(i: int, r: Any) -> str:
    """
    Builds one paper's entry in generate_copilot_synthesis's context.
    Previously this was just "[N] Title - one-line answer"; now it also
    surfaces the structured extraction fields already sitting on every
    record (population/methods/results/outcomes - extracted once during
    pipeline/nodes.py's extraction pass, not recomputed here) plus, when
    full text was available, a short topic-relevant excerpt
    (Record.top_passage) - real RAG grounding for the FINAL synthesis, not
    just per-paper extraction. Every field is optional; only lines with an
    actual value get included; falls back to the bare "title - answer" line
    if nothing structured is present at all.

    Also surfaces each paper's evidence tier (study type) and citation
    count, so the synthesis prompt's evidence-quality rule has something
    concrete to point at (e.g. "12 of 17 studies support this, particularly
    the higher-quality RCTs"). study_type is recomputed here via
    classify_study_type rather than trusted from the record - it's set
    during search/sparse_encoder.py's compute_scores for ranking purposes,
    but extract_node re-fetches records from the database afterwards
    (get_top_unprocessed_records), and study_type isn't a persisted column,
    so by the time a record reaches synthesis that field is back to empty.
    Recomputing is cheap (pure regex over already-available title/abstract)
    and gives the same deterministic answer either way.
    """
    title = _get_val(r, "title")
    answer = _get_val(r, "answer")
    lines = [f"[{i + 1}] {title} - {answer}"]

    study_type = _get_val(r, "study_type") or classify_study_type(title, _get_val(r, "abstract"))
    if study_type and study_type != "unspecified":
        lines.append(f"    Study Type: {study_type.title()}")

    citations = _get_val(r, "citation_count")
    if citations:
        lines.append(f"    Citations: {citations}")

    for label, key in (
        ("Population", "population"),
        ("Methods", "methods"),
        ("Results", "results"),
        ("Outcomes", "outcomes"),
    ):
        value = _get_val(r, key)
        if value and value != "-":
            lines.append(f"    {label}: {value}")

    excerpt = _get_val(r, "top_passage")
    if excerpt:
        lines.append(f'    Excerpt: "{excerpt}"')

    return "\n".join(lines) + "\n"


def parse_llm_json(raw_text: str) -> dict:
    """
    Aggressively cleans LLM output to extract JSON, stripping markdown formatting 
    and model "thinking" tags (e.g., DeepSeek's <think> tags).
    
    Args:
        raw_text (str): The raw string output from the LLM.
        
    Returns:
        dict: The parsed Python dictionary.
        
    Raises:
        AIOutputError: If no valid JSON structure can be found in the text.
    """
    # 1. Strip reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", " ", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 2. Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()
    
    # 3. Locate JSON boundaries
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    
    if start != -1 and end != -1:
        try: 
            return json.loads(cleaned[start:end])
        except Exception: 
            pass
            
    raise AIOutputError(f"LLM Output could not be parsed as JSON: {raw_text[:100]}...")


# ==========================================
# 2. THE AI ENGINE CLASS
# ==========================================

class AtelierAIEngine:
    """
    The central intelligence orchestrator for Atelier.
    
    This class manages all interactions with external LLM providers (OpenAI, 
    Anthropic, Groq, Google). It should be instantiated per-request within 
    Dash callbacks to ensure stateless, dynamic execution.
    """

    def __init__(self, model_choice: str):
        """
        Stores the user's requested model and validates its existence.
        
        Args:
            model_choice (str): The string identifier for the model (e.g., 'gpt-4o').
            
        Raises:
            ValueError: If the requested model is not found in the configuration.
        """
        self.model_choice = model_choice
        
        # Validate that the model exists in the configuration
        if not resolve_model_config(model_choice):
            supported = ", ".join(get_model_choices())
            raise ValueError(f"Unsupported model '{model_choice}'. Supported: {supported}")

    # Every model_choice is "provider:model" (see llm/utils.py). Maps a cloud
    # provider prefix to (its Settings/.env key name, the constructor kwarg
    # that provider's LangChain class expects the key under). Local/proxy
    # providers (ollama/lmstudio/openrouter) aren't here - resolve_model_config
    # already returns fully self-contained constructor_params for those, and
    # ollama/lmstudio need no key at all.
    _CLOUD_PROVIDER_KEY_PARAMS = {
        "anthropic": ("ANTHROPIC_API_KEY", "api_key"),
        "google": ("GOOGLE_API_KEY", "google_api_key"),
        "groq": ("GROQ_API_KEY", "api_key"),
        "openai": ("OPENAI_API_KEY", "api_key"),
    }

    def _provider(self) -> str:
        return self.model_choice.strip().lower().split(":", 1)[0]

    def _get_llm(self, temperature: float = 0.0) -> BaseChatModel:
        """
        Internal Factory: Instantiates the chosen LangChain model at the required temperature.

        Args:
            temperature (float): Creativity dial (0.0 for strict logic, 0.3+ for chat).

        Returns:
            BaseChatModel: An instantiated LangChain chat model ready for invocation.
        """
        config = resolve_model_config(self.model_choice)
        llm_class = config['class']
        model_specific_params = dict(config.get('constructor_params', {}))

        self._ensure_credentials()

        provider = self._provider()
        if provider in self._CLOUD_PROVIDER_KEY_PARAMS:
            # Resolve+inject the actual key fresh (Settings UI takes
            # priority over .env) rather than relying on the LangChain
            # class's own env-var auto-discovery, since a Settings-saved
            # key never touches os.environ.
            env_name, key_param = self._CLOUD_PROVIDER_KEY_PARAMS[provider]
            model_specific_params[key_param] = resolve_key(env_name)
        # ollama/lmstudio/openrouter: constructor_params is already
        # self-contained from resolve_model_config().

        # Merge unified parameters with the specific requested temperature
        all_params = {**_common_llm_params, **model_specific_params, "temperature": temperature}

        return llm_class(**all_params)

    def _ensure_credentials(self) -> None:
        """
        Validates that a usable API key is configured (via Settings UI or
        .env) for the chosen model's provider.

        Ollama and LM Studio are local servers with no API key requirement,
        so they're exempt.

        Raises:
            ValueError: If the provider needs a key and none is configured.
        """
        provider = self._provider()

        if provider in ("ollama", "lmstudio"):
            return

        if provider == "openrouter":
            env_name = "OPENROUTER_API_KEY"
        elif provider in self._CLOUD_PROVIDER_KEY_PARAMS:
            env_name, _ = self._CLOUD_PROVIDER_KEY_PARAMS[provider]
        else:
            raise ValueError(f"Unrecognized provider for model '{self.model_choice}'.")

        if not _is_set(resolve_key(env_name)):
            raise ValueError(
                f"'{self.model_choice}' selected but {env_name} is not configured "
                f"(set it in Settings or your .env file)."
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _execute(self, temperature: float, system_msg: str, user_inputs: dict, is_json: bool = True) -> Any:
        """
        Internal unified execution wrapper featuring automatic exponential backoff retries.
        
        Args:
            temperature (float): The desired creativity setting for this specific call.
            system_msg (str): The system prompt defining the AI's persona and rules.
            user_inputs (dict): A dictionary containing the target payload mapping to `{input_data}`.
            is_json (bool): Whether the function should attempt to aggressively parse JSON output.
            
        Returns:
            Any: A Python dictionary (if is_json=True) or a raw string (if is_json=False).
            
        Raises:
            AIOutputError: If JSON extraction fails, triggering a Tenacity retry.
            AIProviderError: If the external API is unreachable or rate-limited.
        """
        llm = self._get_llm(temperature)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input_data}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            response = chain.invoke(user_inputs)
            return parse_llm_json(response) if is_json else response
        except AIOutputError:
            raise  # Let Tenacity catch it and retry
        except Exception as e:
            raise AIProviderError(f"API Connection Error: {e}") from e

    # ==========================================
    # 3. PUBLIC API METHODS
    # ==========================================

    def plan_topic_queries(self, topic: str) -> dict:
        """
        Translates a natural language research topic into optimized boolean search strings.
        
        Args:
            topic (str): The user's natural language research question.
            
        Returns:
            dict: JSON object containing formatted queries for various databases.
        """
        sys_prompt = """
        You are an Expert Medical and Academic Literature Query Planner.
        Your task is to translate a user's natural language research topic into SIX search queries, one per database. Each database parses query syntax differently - reusing one boolean/field-tagged string across all six produces broken or literal-text searches on most of them. Follow each database's rules exactly.

        General Rules:
        1. Extract the core scientific, medical, or sociological concepts first; build every query from those concepts.
        2. DO NOT let geographic terms overly restrict the search. If a specific country is mentioned (e.g., "Kenya"), you MUST broadly expand the geography (e.g., "Kenya" OR "East Africa" OR "LMIC" OR "Sub-Saharan Africa") to maximize recall.
        3. Remove stop words and unnecessary conversational phrasing.
        4. If the topic is clearly outside a database's scope (e.g. a pure clinical-trial question has no reasonable arXiv angle), set that database's query to the string "null" rather than forcing an irrelevant search.
        5. Output ONLY a valid JSON object matching the Output Schema exactly.

        Per-Database Syntax Rules:

        - pubmed_query (NCBI E-utilities `esearch` term syntax):
          Use PubMed field tags: [tiab] for title/abstract keywords, [Mesh] for controlled-vocabulary MeSH terms. Combine with AND/OR/NOT and parentheses for grouping.
          Example: ("vitamin d"[tiab] OR "cholecalciferol"[Mesh]) AND ("respiratory tract infections"[Mesh] OR "respiratory infection"[tiab])

        - europe_pmc_query (Europe PMC REST search syntax):
          Use Europe PMC's OWN fielded syntax - TITLE:"...", ABSTRACT:"...", AUTH:"...". Do NOT reuse PubMed's [tiab]/[Mesh] tags here; Europe PMC does not understand them and will treat the brackets as literal text. Quote multi-word phrases. Combine with AND/OR.
          Example: (TITLE:"vitamin d" OR ABSTRACT:"vitamin d") AND (TITLE:"respiratory infection" OR ABSTRACT:"respiratory tract infection")

        - openalex_query (OpenAlex `search` parameter):
          OpenAlex's basic search is a full-text relevance search, NOT a boolean parser - operators like AND/OR/NOT and field tags like [tiab] are indexed as literal words and HURT relevance instead of filtering. Output a short, clean, natural-language keyword phrase only. No boolean operators, no field tags, no quotes.
          Example: vitamin d supplementation respiratory tract infections children

        - semantic_scholar_query (Semantic Scholar Graph Search `query` parameter):
          Semantic Scholar also does best with a concise natural-language keyword phrase (not PubMed-style field tags). Quote a phrase only when an exact multi-word term must stay together; avoid boolean operators otherwise.
          Example: vitamin D supplementation respiratory infection risk

        - crossref_query (Crossref `query.bibliographic` parameter):
          Also a relevance search, not a boolean parser - same rule as OpenAlex/Semantic Scholar: a short, clean, natural-language keyword phrase only. No boolean operators, no field tags.
          Example: vitamin D supplementation respiratory tract infection

        - arxiv_query (arXiv API `search_query` value, appended after "all:"):
          A concise natural-language keyword phrase. arXiv is preprints in physics, CS, math, quantitative biology/finance, and statistics - if the topic is purely clinical/medical with no computational, quantitative-methods, or bioinformatics angle, set this to "null" instead of forcing an unrelated search.
          Example: vitamin D supplementation infection risk meta-analysis

        Output Schema:
        {{
        "pubmed_query": "<PubMed [tiab]/[Mesh] boolean string, or \\"null\\">",
        "europe_pmc_query": "<Europe PMC TITLE:/ABSTRACT: boolean string, or \\"null\\">",
        "openalex_query": "<plain natural-language keyword phrase, no operators or tags, or \\"null\\">",
        "semantic_scholar_query": "<plain natural-language keyword phrase, no operators or tags, or \\"null\\">",
        "crossref_query": "<plain natural-language keyword phrase, no operators or tags, or \\"null\\">",
        "arxiv_query": "<plain natural-language keyword phrase, no operators or tags, or \\"null\\">"
        }}
        """
        try:
            return self._execute(temperature=0.0, system_msg=sys_prompt, user_inputs={"input_data": f"Topic: {topic}"})
        except AtelierAIError:
            logger.warning("Query planning failed. Falling back to keyword stripping.")
            cleaned = re.sub(r"[^\w\s]", "", topic)
            return {k: cleaned for k in [
                "pubmed_query", "europe_pmc_query", "openalex_query",
                "semantic_scholar_query", "crossref_query", "arxiv_query",
            ]}

    def extract_paper_data(self, topic: str, paper_json_str: str) -> dict:
        """
        Analyzes a single academic paper against the user's overarching research query.
        
        Args:
            topic (str): The user's overarching research question.
            paper_json_str (str): Stringified JSON representation of the paper's text.
            
        Returns:
            dict: Structured extraction data (e.g., 'is_relevant', 'population', 'results').
        """
        sys_prompt = """
        You are a rigorous Academic Data Extraction and Quality Control AI.
        Your task is to analyze a scientific paper's abstract (and full text, if provided) against a user's research query.

        Rules:
        1. Assess Relevance: Rate how directly this paper answers the user's query on a scale of 0 to 100. If it is entirely unrelated, set "is_relevant" to false.
        2. Extract Parameters: Identify the target population, study methodology, core results, measured outcomes, sample size (N), number of included studies (for meta-analyses), duration, and geographic location.
        3. If a parameter is not explicitly stated in the text, you MUST output "-". Do not guess.
        4. Provide a punchy, 1-sentence "answer" summarizing the paper's main takeaway regarding the user's query.
        5. Spin Check: If full text is provided, verify that the abstract does not exaggerate the findings.

        Output ONLY a valid JSON object matching this exact schema:
        {{
        "is_relevant": true,
        "ai_rank_score": 85,
        "answer": "<1-sentence direct answer>",
        "population": "<target demographic>",
        "methods": "<study design>",
        "results": "<main findings>",
        "outcomes": "<measured variables>",
        "sample_size": "<integer or '-'>",
        "study_count": "<integer or '-'>",
        "duration": "<timeframe or '-'>",
        "country": "<location or '-'>",
        "is_abstract_misleading": false,
        "fidelity_rationale": "<Reasoning or 'Accurate'>"
        }}
        """
        try:
            return self._execute(
                temperature=0.0,
                system_msg=sys_prompt,
                user_inputs={"input_data": f"Query: {topic}\nPaper Data: {paper_json_str}"}
            )
        except AtelierAIError as e:
            logger.warning(f"Paper extraction failed: {e}")
            return {"is_relevant": False, "ai_rank_score": 0, "answer": "Extraction Error"}

    def extract_papers_batch(self, topic: str, papers: List[dict]) -> List[dict]:
        """
        Batched version of extract_paper_data: analyzes several papers in ONE
        LLM call instead of one call per paper. The dominant cost of the
        whole pipeline is this extraction pass (see pipeline/nodes.py's
        extract_node) - one call per paper means 20 sequential round-trips,
        each paying a fixed per-request latency on top of generation time
        (especially pronounced against a local model). Batching a handful of
        papers per call cuts that round-trip count roughly batch-size-fold.

        Args:
            topic (str): The user's overarching research question.
            papers (List[dict]): Each item is {"paper_index": i, "title":
                ..., "content": ...} - paper_index is what lets results be
                matched back to their source paper below even if the model
                reorders, skips, or duplicates an entry in its response,
                which smaller/local models are more prone to than top-tier
                cloud ones.

        Returns:
            List[dict]: One extraction result per input paper, in the SAME
                order as `papers` (re-aligned via paper_index, not
                whatever order the model happened to return). Any paper
                the model didn't return a usable result for gets a graceful
                "Extraction Error" placeholder instead of silently
                vanishing from the batch.
        """
        sys_prompt = """
        You are a rigorous Academic Data Extraction and Quality Control AI.
        Your task is to analyze MULTIPLE scientific papers (abstracts and full text, if provided) against a user's research query, independently, and return one structured result per paper.

        Rules:
        1. Assess Relevance: Rate how directly EACH paper answers the user's query on a scale of 0 to 100. If entirely unrelated, set "is_relevant" to false for that paper.
        2. Extract Parameters: For EACH paper independently, identify the target population, study methodology, core results, measured outcomes, sample size (N), number of included studies (for meta-analyses), duration, and geographic location. Do not mix or blend data between papers.
        3. If a parameter is not explicitly stated in a given paper's text, you MUST output "-" for that paper. Do not guess, and do not borrow a value from a different paper in this batch.
        4. Provide a punchy, 1-sentence "answer" per paper summarizing its own main takeaway regarding the user's query.
        5. Spin Check: If full text is provided for a paper, verify its abstract does not exaggerate the findings.
        6. You MUST return exactly one result per input paper, each carrying its own "paper_index" copied verbatim from that paper's input.

        Output ONLY a valid JSON object matching this exact schema:
        {{
        "results": [
          {{
            "paper_index": 0,
            "is_relevant": true,
            "ai_rank_score": 85,
            "answer": "<1-sentence direct answer>",
            "population": "<target demographic>",
            "methods": "<study design>",
            "results": "<main findings>",
            "outcomes": "<measured variables>",
            "sample_size": "<integer or '-'>",
            "study_count": "<integer or '-'>",
            "duration": "<timeframe or '-'>",
            "country": "<location or '-'>",
            "is_abstract_misleading": false,
            "fidelity_rationale": "<Reasoning or 'Accurate'>"
          }}
        ]
        }}
        """
        papers_json_str = json.dumps(papers)
        try:
            parsed = self._execute(
                temperature=0.0,
                system_msg=sys_prompt,
                user_inputs={"input_data": f"Query: {topic}\nPapers: {papers_json_str}"}
            )
            raw_results = parsed.get("results", []) if isinstance(parsed, dict) else []
        except AtelierAIError as e:
            logger.warning(f"Batch paper extraction failed: {e}")
            raw_results = []

        by_index = {
            item.get("paper_index"): item
            for item in raw_results
            if isinstance(item, dict) and item.get("paper_index") is not None
        }

        return [
            by_index.get(p["paper_index"]) or {"is_relevant": False, "ai_rank_score": 0, "answer": "Extraction Error"}
            for p in papers
        ]

    def generate_copilot_synthesis(self, topic: str, valid_records: List[Any]) -> str:
        """
        Writes an objective literature review summarizing findings from multiple relevant papers.
        
        Args:
            topic (str): The user's original query.
            valid_records (List[Any]): A list of extracted paper records/dictionaries.
            
        Returns:
            str: Plain markdown synthesis with bare [1]/[2] bracket
                citations - NOT HTML-enriched. Interactive citation
                tooltips are rendered exactly once, at display time, by
                ui/layouts/feed.py's build_synthesis_body - baking HTML in
                here too used to double-process the same text every time
                it was displayed (this string gets persisted to the DB
                as-is and re-rendered on every future page load).
        """
        sys_prompt = f"""
        You are the Atelier Synthesizer, an expert at writing academic literature review abstracts.
        Your task is to read the provided paper extracts - each includes a one-line takeaway, structured study details (population/methods/results/outcomes), evidence-tier metadata (study type, citation count) and, where available, an actual excerpt from the paper's full text - and write the answer in the voice and shape of a published journal abstract: dense, structured, evidence-forward prose, not a loose bulleted breakdown.

        Structure the output with these four Markdown headers, in this order, every time:

        ### Background
        1 sentence of plain framing: what question the literature is being asked to answer.

        ### Evidence Synthesis
        The core of the abstract - a dense paragraph (not bullets) synthesizing what the body of evidence actually shows, written the way a journal abstract's Results section reads. This is where evidence strength/consistency gets characterized, not listed separately: name real counts and the strongest study types driving any agreement (e.g. "12 of 17 studies support this, particularly the systematic reviews and RCTs [2][5][9]"), and call out disagreement plainly, noting if it splits along evidence quality (e.g. "the higher-powered cohort studies find X, while the effect is weaker in smaller case series"). Never fabricate a count that isn't clearly supported by the provided papers - "most papers.../a minority of studies..." is fine when an exact number would be a guess.

        ### Key Findings
        The detailed breakdown, formatted however best fits what the papers actually discuss:
        - Distinct variables/causes → a Markdown table.
        - A debated issue → a "Pros & Cons" list.
        - A definitional/conceptual question → thematic bullet points.

        ### Conclusion
        1 sentence naming the main gap in the literature, if one is apparent from the provided papers.

        Rules:
        1. Base your answer ENTIRELY on the provided insights. Do not introduce outside knowledge.
        2. Prefer grounding specific claims (numbers, effect sizes, specific findings) in a paper's "Results"/"Excerpt" text over its one-line takeaway when both are available - the takeaway is a summary, the results/excerpt are the actual evidence.
        3. You MUST cite the source of every claim using bracketed numbers corresponding to the paper index (e.g., [1]).
        CRITICAL: If citing multiple papers, use separate brackets like [1][2]. DO NOT use comma-separated formats like [1, 2].
        4. COVERAGE REQUIREMENT: {len(valid_records)} papers are provided below, numbered [1] through [{len(valid_records)}]. Every single one MUST be cited somewhere across the whole answer - do not silently drop any of them, even minor or tangential ones. If a paper only weakly supports the topic, still work it in (a Key Findings table row, a brief caveat in Evidence Synthesis, or the Conclusion) rather than omitting its citation entirely. Before finishing, verify every number from [1] to [{len(valid_records)}] appears at least once.
        """

        context = "".join(_format_synthesis_entry(i, r) for i, r in enumerate(valid_records))

        try:
            # Returned as plain markdown with bare [1]/[2] bracket citations
            # - NOT HTML-enriched here. Citation-to-tooltip HTML rendering
            # happens exactly once, at render time, in
            # ui/layouts/feed.py's build_synthesis_body. Enriching here too
            # used to double-process the same text (this raw_md gets saved
            # to the DB as-is, then build_synthesis_body ran ITS OWN
            # citation regex over the ALREADY-HTML-enriched result on every
            # render) - a block-level <div> landing inline inside markdown
            # text via dangerously_allow_html confused Dash's markdown
            # parser, producing literal always-visible tooltip text instead
            # of a hidden-until-hover tooltip.
            return self._execute(
                temperature=0.1,
                system_msg=sys_prompt,
                user_inputs={"input_data": f"Topic: {topic}\nContext:\n{context}"},
                is_json=False
            )
        except AtelierAIError:
            return "Failed to generate synthesis due to an AI processing error. Please try again."

    def generate_atelier_meter(self, topic: str, valid_records: List[Any]) -> Optional[dict]:
        """
        The Atelier Meter: an agreement meter for yes/no-shaped research
        questions (e.g. "Does metformin reduce cardiovascular risk?"),
        inspired by Consensus.app's own "Consensus Meter" feature. The
        returned dict/persisted JSON shape and the DB column name
        (QueryModel.consensus_meter) keep their original names internally -
        only the user-facing label and these public symbol names changed -
        renaming the column would need its own migration against already-
        persisted data for no functional benefit.

        Two-layer gate before this does any real work:
        1. core.utils.is_yes_no_question - a free regex check on the query's
           opening word(s), done by the CALLER before this is even invoked,
           so the common case (a non-yes/no topic) never reaches an LLM call.
        2. The LLM's own "applicable" verdict below - is_yes_no_question is
           deliberately permissive/dumb (just matches "Does .../Is ...");
           the LLM catches cases that are grammatically a yes/no question but
           not actually answerable as one (e.g. "Is there a relationship
           between X and Y?" often doesn't cleanly resolve to yes/no).

        Classifies each paper's stance (not just relevance) then aggregates
        WEIGHTED by evidence quality in Python, not the LLM - mirrors
        Consensus.app's Consensus Meter 2.0 fix for the same problem (a case report and a systematic review
        don't get an equal vote). Uses the exact same STUDY_TYPE_WEIGHTS /
        citation_bonus already driving search ranking (search/sparse_encoder.py)
        for one consistent "quality" vocabulary across the app, rather than
        inventing a second weighting scheme.

        Returns:
            Optional[dict]: None if not applicable (either gate declines) or
                fewer than 2 papers actually took a yes/no/possibly stance
                (not enough signal for a meaningful meter). Otherwise:
                {"verdict": str, "yes_pct": float, "no_pct": float,
                 "possibly_pct": float, "counts": {"yes": int, "no": int, "possibly": int},
                 "positions": [{"paper_index": int, "stance": str}, ...]}
        """
        sys_prompt = """You are a strict yes/no research-question classifier.

        Task 1: Decide if the user's topic genuinely resolves to a yes/no verdict across the literature
        (e.g. "Does X reduce Y?", "Is X associated with Y?"). If the topic is too open-ended, comparative,
        or exploratory to have a real yes/no answer, set "applicable" to false and return an empty positions list.

        Task 2: If applicable, classify EVERY paper's own position on the question, using ONLY its takeaway/results/outcomes:
        - "yes": the paper's findings support a yes answer
        - "no": the paper's findings support a no answer
        - "possibly": mixed, conditional, or inconclusive findings on the question itself
        - "not_relevant": the paper doesn't actually address this specific question, despite being in the result set

        Return ONLY valid JSON matching this exact schema, with one entry per paper index provided:
        {{"applicable": true or false, "positions": [{{"paper_index": 1, "stance": "yes"}}, ...]}}
        """

        context = "".join(
            f"[{i + 1}] {_get_val(r, 'title')} - {_get_val(r, 'answer')}\n"
            f"    Results: {_get_val(r, 'results')}\n    Outcomes: {_get_val(r, 'outcomes')}\n"
            for i, r in enumerate(valid_records)
        )

        try:
            result = self._execute(
                temperature=0.0,
                system_msg=sys_prompt,
                user_inputs={"input_data": f"Question: {topic}\nPapers:\n{context}"},
            )
        except AtelierAIError as e:
            logger.warning(f"Atelier Meter classification failed, skipping meter: {e}")
            return None

        if not isinstance(result, dict) or not result.get("applicable"):
            return None

        positions = result.get("positions") or []
        weight_by_stance = {"yes": 0.0, "no": 0.0, "possibly": 0.0}
        count_by_stance = {"yes": 0, "no": 0, "possibly": 0}
        kept_positions = []

        for pos in positions:
            idx = pos.get("paper_index")
            stance = pos.get("stance")
            if stance not in weight_by_stance or not idx or not (1 <= idx <= len(valid_records)):
                continue
            rec = valid_records[idx - 1]
            study_type = _get_val(rec, "study_type") or classify_study_type(_get_val(rec, "title"), _get_val(rec, "abstract"))
            # Same evidence-tier weighting as search ranking, plus a +0.5
            # floor so an "unspecified" study type still casts a real (if
            # minimal) vote rather than a literal zero.
            weight = STUDY_TYPE_WEIGHTS.get(study_type.lower(), 0.0) + 0.5 + citation_bonus(_get_val(rec, "citation_count"))
            weight_by_stance[stance] += weight
            count_by_stance[stance] += 1
            kept_positions.append({"paper_index": idx, "stance": stance})

        total_weight = sum(weight_by_stance.values())
        if total_weight <= 0 or sum(count_by_stance.values()) < 2:
            return None

        pct = {k: round((v / total_weight) * 100, 1) for k, v in weight_by_stance.items()}

        if pct["yes"] >= 65:
            verdict = "Likely Yes"
        elif pct["no"] >= 65:
            verdict = "Likely No"
        elif pct["yes"] > pct["no"] and pct["yes"] >= 40:
            verdict = "Possibly Yes"
        elif pct["no"] > pct["yes"] and pct["no"] >= 40:
            verdict = "Possibly No"
        else:
            verdict = "Mixed"

        return {
            "verdict": verdict,
            "yes_pct": pct["yes"],
            "no_pct": pct["no"],
            "possibly_pct": pct["possibly"],
            "counts": count_by_stance,
            "positions": kept_positions,
        }

    def generate_paper_summary(self, topic: str, record: Any) -> str:
        """
        A focused summary of ONE paper specifically in relation to the
        user's topic - not a repeat of its 1-line extraction "answer", but
        a few sentences of plain prose grounded in whatever's actually
        available (results/outcomes/abstract), written for someone who just
        selected this one paper out of a results list and wants a real
        sense of it before opening the PDF.

        Returns:
            str: Plain markdown, no citation brackets (there's nothing to
                cite - this is about a single paper, not a synthesis across
                many).
        """
        title = _get_val(record, "title")
        abstract = _get_val(record, "abstract")
        results = _get_val(record, "results")
        outcomes = _get_val(record, "outcomes")
        population = _get_val(record, "population")
        methods = _get_val(record, "methods")
        answer = _get_val(record, "answer")

        sys_prompt = """You are Atelier's research assistant, summarizing a single paper for someone researching a specific topic.

        Write 3-5 sentences of plain prose (no headers, no bullet points, no citation brackets) covering:
        - What this paper specifically found or argues, in relation to the user's topic
        - The population/methods, if that context matters for interpreting the finding
        - How directly relevant this paper is to the topic - say plainly if it's only tangential

        Base this ENTIRELY on the provided paper data. If the data given is too thin to say much, say so plainly rather than padding with generic language.
        """

        context = (
            f"Title: {title}\n"
            f"Population: {population}\n"
            f"Methods: {methods}\n"
            f"Results: {results}\n"
            f"Outcomes: {outcomes}\n"
            f"Abstract: {abstract}\n"
            f"One-line takeaway: {answer}\n"
        )

        try:
            return self._execute(
                temperature=0.2,
                system_msg=sys_prompt,
                user_inputs={"input_data": f"Topic: {topic}\nPaper:\n{context}"},
                is_json=False,
            )
        except AtelierAIError:
            return "Couldn't generate a summary for this paper right now — please try again."

    def followup_chat(self, query: str, current_topic: str, processed_records: List[Any]) -> dict:
        """
        Intent Router: Evaluates whether a follow-up needs new external
        documents (SEARCH), can be answered directly from what's already
        loaded (CHAT), or needs active multi-step digging into the
        existing papers - comparing them, reading full text, asking a
        targeted question of one specific paper - without necessarily
        needing a whole new search (INVESTIGATE).

        Args:
            query (str): The user's new question.
            current_topic (str): The overarching timeline topic.
            processed_records (List[Any]): Papers currently loaded in the feed view.

        Returns:
            dict: JSON containing 'intent' (SEARCH/CHAT/INVESTIGATE) and an
                optimal 'standalone_query' (SEARCH only).
        """
        sys_prompt = """You are an intelligent routing agent for an academic research engine.

            Task 1: Classify the user's follow-up into exactly one of three routes:

            - 'CHAT': A direct question answerable in one pass from the existing papers' takeaways -
              a summary, a specific fact, a simple comparison already visible at a glance.
            - 'INVESTIGATE': Needs active, multi-step digging into the EXISTING papers to answer well -
              e.g. "compare how paper 3 and 7 measured X", "what does the full text of the review say about Y",
              "which of these papers actually addresses Z in depth". The papers likely already cover this,
              but answering it well takes more than one read-through of the takeaways.
            - 'SEARCH': The existing papers don't cover this at all - a new demographic, population, or
              question genuinely outside what's already been gathered.

            Judge from each paper's takeaway below, not just its title - a paper whose title doesn't
            mention the follow-up's subject may still directly address it (e.g. a title about "screening
            determinants" whose takeaway discusses income/employment).

            Default to CHAT when genuinely unsure between CHAT and INVESTIGATE (INVESTIGATE costs more time
            for marginal benefit on a question CHAT could already answer). Default to INVESTIGATE over SEARCH
            when unsure whether the existing papers cover it (SEARCH throws away existing context and makes
            the user wait through a whole new search; INVESTIGATE can still fall back to searching if it
            turns out to be needed, but tries the cheaper path first).

            Task 2: If SEARCH, combine the original topic and the follow-up into a single standalone search query.

            Return ONLY a valid JSON object matching this exact schema:
            {{"intent": "SEARCH" or "CHAT" or "INVESTIGATE", "standalone_query": "merged query here or null"}}
            """

        # Includes each paper's one-line takeaway (not just its title) so the
        # router can actually judge topical coverage - title-only context
        # meant a paper whose TITLE didn't mention e.g. "income" looked
        # irrelevant even when its extracted takeaway explicitly discussed
        # income as a factor, which was silently over-routing follow-ups to
        # SEARCH (discarding/de-emphasizing perfectly good existing context)
        # far more often than necessary.
        context_preview = "".join([
            f"- {_get_val(r, 'title')}: {_get_val(r, 'answer')}\n" for r in processed_records
        ])
        user_input = {"input_data": f"Original Topic: {current_topic}\nContext:\n{context_preview}\nFollow-up: {query}"}
        
        try:
            result = self._execute(temperature=0.0, system_msg=sys_prompt, user_inputs=user_input)
            return result if isinstance(result, dict) else {"intent": "CHAT", "standalone_query": None}
        except Exception as e:
            logger.warning(f"Router fallback triggered: {e}")
            return {"intent": "CHAT", "standalone_query": None}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
    def chat_with_literature(self, chat_history: List[Dict[str, str]], context_records: List[Any]) -> str:
        """
        Context-aware Conversational RAG. Answers deep-dive questions based on the 
        currently loaded paper feed. Bypasses standard templates to safely ingest user strings.
        
        Args:
            chat_history (List[Dict]): Historical role/content pairings.
            context_records (List[Any]): Papers currently loaded in the feed view.
            
        Returns:
            str: Plain markdown AI response with bare [1]/[2] citations -
                see generate_copilot_synthesis's docstring for why HTML
                citation enrichment isn't done here.
        """
        llm = self._get_llm(temperature=0.3)
        context_str = "".join(_format_synthesis_entry(i, r) for i, r in enumerate(context_records))

        sys_prompt = f"""
        You are Atelier's specialized Academic Research Assistant, working from the literature Context below.

        The user's message is a QUESTION, a REWRITE request, or a WRITTEN-DELIVERABLE request - infer which from their own wording, and handle it accordingly:

        1. QUESTIONS (e.g. "what about X?", "does this apply to Y?", "focus on Z"):
        - CLOSED-BOOK: your only source of truth is the provided Context. If it doesn't contain the answer, say so explicitly: "The provided literature does not contain information to answer this."
        - Maintain an objective, academic tone. Avoid hyperbole.

        2. REWRITE REQUESTS (the user has pasted their OWN text - a paragraph, draft, or claim - and wants it rewritten, strengthened, or expanded using the evidence you have, e.g. "rewrite this using the evidence", "improve this paragraph with citations"):
        - Preserve the user's own structure, voice, and intent as much as reasonable - you're strengthening their writing, not replacing it with something unrecognizable.
        - Weave in specific findings/citations from the Context wherever they genuinely support, complicate, or contradict a claim in the user's text.
        - If their text makes a claim the Context doesn't address, don't silently drop it or invent evidence for it - note briefly (e.g. a short bracketed aside) that the literature here doesn't cover that specific point.

        3. WRITTEN-DELIVERABLE REQUESTS (the user is asking you to WRITE something in a specific genre/format from scratch using the Context - e.g. "write a 300-word abstract", "write a summary paragraph", "write an introduction"):
        - Match the ACTUAL CONVENTIONS of the requested genre, not the structure of an earlier message in this conversation. An abstract in particular is DENSE FLOWING PROSE - one or a few unbroken paragraphs, written in past/present tense as a self-contained synopsis. It is NEVER broken into headers like "Background/Evidence Synthesis/Key Findings/Conclusion", and NEVER contains a Markdown table or bullet list - those belong to Atelier's separate full literature-review synthesis, a different deliverable from what's being asked for here. If the earlier conversation contains that format, do not imitate it just because it's nearby.
        - Respect any requested length (e.g. "300 words") as a real target, not a suggestion to ignore.
        - COVERAGE: draw on and cite as much of the full Context as genuinely fits the requested length and topic, not just the first few papers - {len(context_records)} papers are available below; a comprehensive deliverable like an abstract should engage with the breadth of the evidence, not a narrow subset of it.

        In ALL modes: you MUST cite claims grounded in the Context using the bracketed numbers from the context headers (e.g., [1]).
        CRITICAL: If citing multiple papers, use separate brackets like [1][2]. DO NOT use comma-separated formats like [1, 2].

        Context:
        {context_str}
        """

        messages = [SystemMessage(content=sys_prompt)]
        
        # Keep only the last 6 messages to preserve token window
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
            
        try:
            # Plain markdown with bare [1]/[2] citations - see
            # generate_copilot_synthesis's docstring for why HTML citation
            # enrichment belongs solely in build_synthesis_body at render
            # time, not baked in here.
            return llm.invoke(messages).content
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return "I encountered an error connecting to the AI. Please try asking your question again."