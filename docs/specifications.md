# Technical Specifications

Operational specifications extracted from dissertation methods chapter.
Claude Code should reference this file when generating modules.

## Phase 1: Benchmark Construction and Gold Standard Establishment

### 1.1 Report Identification and Sampling

**Source:** UNDP Independent Evaluation Office Evaluation Resource Center
**Filter settings (include):** Decentralized Country Programme Evaluation, Global Programme, Impact, Independent Country Programme Evaluation, Outcome, Portfolio Evaluation, Project, Regional Programme, Thematic, UNDAF
**Filter settings (exclude):** Other, Synthesis and Lessons
**Export:** Excel, 6,864 reports as of January 2026

**Sampling procedure:**
- Three-tab spreadsheet: Tab 1 (Main) = filtered reports with metadata (ID, title/URL, type, country, year, budget, management response, joint eval, GEF indicators); Tab 2 (Randomizer) = 200 random numbers via =RANDARRAY(200,1,0,6864,TRUE); Tab 3 (Sample) = matched metadata via =FILTER(Main!A:Z, ISNUMBER(MATCH(Main!A:A, Randomizer!A:A, 0)))
- 200-report oversample to accommodate ~85-95% fragment eligibility rate plus 4 calibration examples

### 1.2 Fragment Extraction

**Target:** 150 eligible fragments, processed sequentially from randomized list

**Eligibility criteria (all three required):**
1. Addresses a single evaluation criterion (relevance, effectiveness, efficiency, sustainability, impact, or coherence)
2. Interpretable without extensive external context
3. Constitutes evaluative synthesis, not purely descriptive or prescriptive content

**Exclusion rules:**
- Multiple independent criteria without clear separation
- Heavy reliance on external context
- Mainly descriptive or prescriptive content

**Extraction rules:**
- Navigate to conclusion section ("Conclusions" or "Evaluative Conclusions")
- Screen segments sequentially against three criteria
- When multiple eligible fragments exist in one report, random number generator selects one
- Typical fragment length: 1-5 paragraphs capturing complete evaluative reasoning for one criterion
- Manual extraction required (not automated)

**Recording:** Spreadsheet with metadata: criterion addressed, paragraph count, extraction date, notes
**Tracking:** Running counts of reports processed, reports excluded with reasons, fragments extracted toward 150 target

### 1.3 Calibration Example Selection

**Pool:** Continue extraction after 150 benchmark fragments until 10-15 additional eligible fragments identified
**Selection:** 4 fragments via purposive sampling

**Balance requirements:**
- 2 sound, 2 not sound
- Sound examples: vary strengths (argument structure, synthesis, qualification)
- Not sound examples: vary deficiencies (weak warrants/Domain 3, inadequate evidence or synthesis/Domains 2 and 4, over-generalization/Domain 6)
- Complexity range: single-paragraph straightforward to multi-paragraph moderate; avoid extreme boundary cases

**Documentation per example:** Fragment ID, source report, classification, primary distinguishing characteristic, complexity level, selection rationale

### 1.4 Expert Judgment and Gold Standard Establishment

**Calibration phase:**
- 10-15 practice fragments from outside corpus
- Per fragment: apply 6-domain checklist, classify sound/not sound, write 2-4 sentence rationale
- Reflect on: time taken, salient domains, trade-off weighing, consistency
- Continue until stable judgment patterns emerge
- Document boundary cases in calibration memo (Appendix E.1)

**Systematic assessment (150 fragments):**
- Per fragment: review text, verify inclusion criteria, apply 6-domain checklist holistically as orienting heuristics (not algorithmic rules), classify sound/not sound, write 2-4 sentence rationale identifying salient domains and trade-offs
- Flag as boundary case when: domains conflict, judgment rests on standards/degrees, reasonable disagreement possible

**Session management:**
- 15-20 fragments per session
- 1-hour breaks between sessions
- 5-10 working days total
- Session log (Appendix E.2): date, fragment range, observations

**RC1 — Expert Temporal Consistency:**
- Triggered at 50-fragment milestone
- 20 randomly selected fragments reassessed without viewing initial judgments
- Cohen's kappa and percent agreement calculated
- Target: κ ≥ 0.80
- Functions as diagnostic screen, not formal reliability study (small sample limits kappa precision)
- Discrepancies investigated: systematic inconsistency vs. boundary-case variation
- Documented in Appendix E.3

**Gold standard lock:**
- Confirm all 150 fragments have classifications and rationales
- Create read-only timestamped copy
- Sign pre-commitment declaration (Appendix E.4): "Judgments established [date]. Will not be revised based on model outputs."
- Purpose: prevents criterion contamination and circular influence

**Formal templates:** Appendices E.1 (calibration memo), E.2 (session log), E.3 (consistency check), E.4 (lock declaration)

## Phase 2: Model Evaluation and Performance Assessment

### 2.1 Prompt Construction

**Two conditions:**
- **Zero-shot:** Task instructions + abbreviated 6-domain checklist reference + response format specs (Appendix B)
- **Few-shot:** Same as zero-shot + 4 calibration examples with expert classifications and rationales (Appendix C)

**Consistency rules:**
- Identical prompt structure across all models and fragments within each condition
- Only the target fragment text varies between assessments
- Crossed design: zero-shot tests operationalization of abstract principles; few-shot tests whether worked examples enhance reasoning

### 2.2 Model Selection and Configuration

**Closed models (3):** Anthropic Claude Opus 4.6, OpenAI GPT 5.2, Google Gemini 3 Pro
**Open models (3):** DeepSeek V3.2 Thinking, Moonshot AI Kimi K2 Thinking, Z.AI GLM 4.7
**Note:** Selections subject to change to ensure top 3 reasoning SOTA for each architecture category

**Standardized parameters:**
- Temperature: 0 (maximum determinism)
- Max output tokens: 500
- Expected input: ~750 tokens (prompt + fragment)
- Expected output: ~250 tokens (classification + 2-4 sentence rationale)
- Runs per fragment-model-prompt combination: 3 (for consistency analysis)
- API versions documented at time of execution

**Primary vs. comparison models:**
- GPT 5.2: full diagnostic treatment (performance + failure mode coding)
- 5 comparison models: performance assessment only

**Scale:**
- Total API calls: 5,400 (150 fragments × 6 models × 2 prompts × 3 runs)
- Estimated cost: $168-173 (closed ~$158, open ~$9-15)
- Total tokens: ~5.4 million
- Selection basis: Vellum leaderboard (closed), LiveBench (open), as of January 20, 2026

### 2.3 Model Execution and Response Capture

**Execution order:**
- Fragments processed sequentially in randomized order (different from gold-standard sequence)
- All 6 models evaluate each fragment under both conditions before proceeding to next fragment
- 3 independent runs per model-prompt combination with 5-second delays between runs
- Each run is a completely independent API call (no memory, no conversation history, no accumulated context)

**Error handling:**
- Automatic retry on timeout, service unavailability, or rate limit exceeded
- Up to 3 retries with exponential backoff: 10s, 30s, 90s
- If all retries fail: log error, continue execution, re-attempt failed calls in separate session
- Prevents wasted compute on voided batches

**Data capture (per API call):**
- Fragment ID
- Model family
- Prompt condition (zero-shot / few-shot)
- Run number (1-3)
- Classification output (sound / not sound)
- Rationale text
- Timestamp
- API latency (seconds)
- Input token count
- Output token count
- API version
- Error flag

**Response persistence:** Each successful response immediately written to database (preserves data if execution interrupted)

**Session management:**
- Batches of 500-1,000 calls per session (~14-28 fragments with all combinations)
- Sessions spread across multiple days
- Technical issues, retry outcomes, progress logged in Appendix F.1
- Implementation code in Appendix G

### 2.4 Automated Response Scoring

**4-step sequential pipeline applied to each of 5,400 responses:**

**Step 1 — Internal Coherence Check (tiered):**

*Tier 1: Rule-based keyword matching*
- Coherent signal: "sound" classification + strength indicators (affirmations of reasoning quality, satisfied checkpoints, positive evaluative language) OR "not sound" classification + weakness indicators (violated/missing checkpoints, reasoning flaws, negative evaluative language)
- Confident signal (indicators substantially outweigh opposite) → assign automatically
- Mixed, balanced, or insufficient indicators → route to Tier 2

*Tier 2: Lightweight LLM screen (Claude Haiku, temperature 0)*
- Input: classification + rationale text ONLY (no fragment, no gold standard, no study metadata)
- Output: binary coherent/incoherent judgment
- Rule-based and LLM agree → assign accordingly
- Rule-based and LLM disagree → flag as "ambiguous" → route to manual review

*Rationale for LLM over sentiment analysis:*
- Evaluative meta-language contains terms like "weaknesses," "deficiencies," "limitations" in affirmative contexts
- Generic sentiment classifiers (spaCy, TextBlob) produce systematic false negatives on evaluative discourse
- LLM check performs reading comprehension ("does rationale support label?"), not evaluative reasoning ("is this sound?")
- No circularity: coherence check is a simpler task than the benchmarked task

**RC2 — Automated Coherence Validation:**
- Stratified sample: 20 coherent + 20 incoherent + 10 ambiguous (or all ambiguous if <10)
- Researcher independently codes coherence without viewing automated result
- Cohen's kappa calculated; target: κ ≥ 0.80
- If target not met: recalibrate keyword lists and LLM prompt, repeat check

**Step 2 — Classification Accuracy (coherent responses only):**
- Compare model classification to gold standard
- Correct: classification matches
- Incorrect: classification diverges
- Incoherent responses bypass this step (already failed)

**Step 3 — Run-Level Outcome Assignment:**
- Pass = coherent AND correct
- Fail = incoherent OR incorrect (or both)
- Output: Pass/Fail for each of 5,400 runs

**Step 4 — Fragment-Level Adjudication:**
- Majority rule across 3 runs per fragment-model-prompt combination
- Pass: ≥2 of 3 runs pass
- Fail: ≥2 of 3 runs fail
- Output: Pass/Fail for each of 1,800 combinations

**Performance metrics:**
- Primary: fragment-level pass rate (% of 1,800 combinations passing)
- Secondary: unanimous agreement rate (% where all 3 runs pass)
- Tertiary: run-level pass rate (% of 5,400 runs passing)
- All metrics stratified by: model family, prompt condition, interaction

**Implementation:** Appendix G

## Phase 3: Analysis, Synthesis, and Meta-Evaluation

### 3.1 Failure Mode Analysis

**Scope:** GPT 5.2 fragment-level failures only; estimated 60-120 cases (150 fragments × 2 prompts × 20-40% failure rate)

**Failure type eligibility:**
- Type 1 (coherent but incorrect): eligible for coding — internally consistent reasoning enables diagnostic analysis
- Type 2 (incoherent but correct): documented quantitatively, excluded from coding
- Type 3 (incoherent and incorrect): documented quantitatively, excluded from coding
- Rationale: contradictory reasoning provides no stable pathway for diagnosis

**Coding framework (abductive):**
- Deductive: 22 checkpoints from Integrated Evaluative Reasoning Checklist mapped to failure modes across 6 domains (Appendix D.1)
- Inductive: emergent codes for patterns not captured by checkpoint framework (Appendix D.2)
- Tests whether theoretical framework predicts failures while remaining open to unanticipated breakdowns

**5-step coding procedure per fragment-level failure:**
1. Select one representative failed run (prioritize clearest failure pattern if multiple runs failed)
2. Review failed response alongside original fragment text and gold standard judgment
3. Identify primary reasoning breakdown explaining divergence
4. Assign 1-3 failure mode codes (primary + secondary when applicable)
5. Document brief justification in coding memo with interpretive decisions

**Optional exploratory analysis:** Single failures in 2/3 pass scenarios; contingent on failure count and resources

**Documentation:**
- Appendix D.1: checkpoint-to-failure-code mappings with operational definitions
- Appendix D.2: inductively developed emergent codes
- Appendix D.3: coding memo tracking analytical decisions

**RC3 — Failure Mode Coding Temporal Consistency:**
- Double-code 20% of randomly selected fragment-level failures
- Cohen's kappa between first and second coding passes
- Target: κ ≥ 0.80

**Scope limitation:**
- Only GPT 5.2 failures receive individual coding
- Comparison models: aggregate pass/fail rates and initial failure codes only
- Architecture-specific failure patterns across all models reserved for future research

### 3.2 Comparative Performance Analysis

**Metrics calculated for each of 12 model-prompt combinations (6 models × 2 prompts):**
- Fragment pass rate: fragments passed / 150 × 100
- Unanimous agreement rate: fragments where all 3 runs pass / 150 × 100
- Run-level pass rate: individual runs passing / 450 × 100

**Three analytical dimensions:**
- By model family: pass rates across 6 models, collapsed across prompts
- By prompt condition: zero-shot vs. few-shot, collapsed across models
- Interaction: whether prompt effects vary by architecture (open vs. closed)

**Outputs:**
- Performance matrix with all 12 combinations and 95% confidence intervals (Appendix H.1)
- Grouped bar charts for primary comparisons

### 3.3 Statistical Testing

**Test 1 — Prompt condition effect:**
- McNemar's test for paired proportions (within-fragment design)
- H₀: no difference in pass rates between zero-shot and few-shot
- α = 0.05
- Effect size: odds ratio with 95% CI

**Test 2 — Model family differences:**
- Chi-squared test of independence (6 models × 2 outcomes)
- H₀: pass rates do not differ across model families
- α = 0.05
- If significant: post-hoc pairwise comparisons with Bonferroni correction

**Test 3 — Architecture type comparison:**
- Independent proportions z-test (open vs. closed aggregated pass rates)
- H₀: no difference between open and closed models
- α = 0.05
- Effect size: Cohen's h

**Implementation:** Appendix H.2

### 3.4 Failure Pattern Synthesis

**Focus:** GPT 5.2 as primary diagnostic model; comparative metrics from all 6 models (Section 3.2) provide context

**Within-domain analysis:**
- Failure frequency per domain: (GPT 5.2 failures with domain code / total GPT 5.2 failures) × 100
- Most common checkpoint violations within each domain
- Domain-level failure rates compared across prompt conditions (few-shot mitigation assessment)

**Cross-domain analysis:**
- Co-occurrence matrix of failure-mode pairs
- Identifies whether certain breakdowns predict others
- Tests whether failures cluster around specific reasoning stages (e.g., evidence → synthesis)

**Emergent code analysis:**
- Frequency of inductive codes relative to deductive checkpoint-based codes
- Assessment of whether emergent patterns reveal unanticipated breakdowns

**GPT 5.2 diagnostic signature:**
- Primary failure domains and most common checkpoint violations
- Distinctive failure patterns and reasoning breakdown sequences
- Prompt condition effects on specific failure types

**Cross-model inference:** Comparative pass rates and challenge-case patterns across all 6 models suggest whether weaknesses are model-specific or reflect broader architectural limitations

**Documentation:** Narrative form with supporting tables in Appendix H.3

### 3.5 Challenge Case Analysis

**Criterion 1 — Systematic disagreement:**
- GPT 5.2 fails AND ≥2 comparison models also fail
- Indicates cross-architectural difficulty
- Analysis: evaluation criterion addressed, paragraph length, complexity indicators, common features, hypotheses about diagnostic difficulty

**Criterion 2 — Recurring failure mode:**
- Single failure mode accounts for ≥10% of all GPT 5.2 failures
- Indicates systematic reasoning weakness
- Analysis: example fragments, commonalities in content/structure, systematic gaps vs. ambiguous boundary cases

**Meta-analysis:**
- Calculate overlap between Criteria 1 and 2
- High overlap → genuinely difficult cases challenging models broadly
- Low overlap → orthogonal difficulty dimensions (systematic disagreement and recurring weaknesses identify distinct fragment characteristics)

**Documentation:** Appendix H.4 with annotated examples

### 3.6 Meta-Evaluation and Validity Assessment

**Construct validity triangulation:**
- Does GPT 5.2 struggle more with synthesis/integration (Domain 4) than evidence identification (Domain 2)?
- Does few-shot calibration improve performance on tacit judgment domains (Domain 6)?
- Do challenge cases cluster around theoretically complex criteria (e.g., sustainability vs. relevance)?

**Gold standard defensibility check:**
- Re-review expert classifications/rationales for challenge cases and high-disagreement fragments
- Assess whether expert judgments remain defensible given model outputs
- Document cases where model responses reveal ambiguity in expert classifications
- Gold standard remains locked; check is analytical reflection, not revision

**Methodological limitations documented:**
- Fragment selection bias (conclusion sections may not represent typical evaluative reasoning)
- Corpus scope (UN evaluation contexts only)
- Single-expert design (no consensus panel)
- Prompt engineering constraints (limited few-shot examples, single prompt formulation per condition)
- Single-model failure coding (GPT 5.2 only; other models unexplored)

**Reliability synthesis:**
- Summary table: RC1, RC2, RC3 with achieved kappa values
- Assessment of whether all targets (κ ≥ 0.80) achieved
- Documentation: Appendix H.5

### 3.7 Reporting and Documentation

**Primary outputs:**
- Executive summary (2-3 pages): pass rates, prompt effects, GPT 5.2 failure patterns, challenge case insights
- Full results chapter: Sections 3.1-3.6 with tables and figures
- Appendix package: H.1 (performance matrices/visualizations), H.2 (statistical tests/code), H.3 (failure pattern synthesis for GPT 5.2), H.4 (challenge case analysis), H.5 (meta-evaluation/validity)

**Reproducibility documentation:**
- Complete dataset: fragment IDs, gold standard classifications, all model responses (6 models), scoring outcomes, GPT 5.2 failure codes
- Analysis code repository: statistical tests, visualization scripts, failure coding tools
- Methodological audit trail: calibration memos, session logs, reliability check results, coding decisions
- Public deposition: Open Science Framework and/or GitHub upon completion

**Final deliverable:** Discussion chapter interpreting results and implications for evaluation practice
```
## Data Schemas

### 1. Fragment Corpus (data/raw/fragment_corpus.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| fragment_id | string | F_001–F_150 | Primary key |
| report_id | integer | 0–6864 | From randomizer |
| report_title | string | — | |
| report_url | string | Valid URL | |
| report_type | string | Decentralized CPE, Global Programme, Impact, Independent CPE, Outcome, Portfolio, Project, Regional Programme, Thematic, UNDAF | |
| country | string | — | |
| year | integer | 2000–2026 | |
| budget | string | — | If available |
| joint_evaluation | boolean | True/False | |
| gef_evaluation | boolean | True/False | |
| criterion_addressed | string | relevance, effectiveness, efficiency, sustainability, impact, coherence | |
| fragment_text | string | — | 1–5 paragraphs |
| paragraph_count | integer | 1–5 | |
| extraction_date | date | ISO format | |
| extraction_notes | string | — | Optional |

### 2. Gold Standard (data/gold_standard/gold_standard_locked_YYYY-MM-DD.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| fragment_id | string | F_001–F_150 | FK → fragment_corpus |
| expert_classification | string | sound, not_sound | IMMUTABLE after lock |
| expert_rationale | string | — | 2–4 sentences, identifies salient domains |
| salient_domains | string | 1,2,3,4,5,6 | Comma-separated |
| boundary_case_flag | boolean | True/False | |
| boundary_case_notes | string | — | If boundary_case_flag = True |
| session_id | integer | 1–10 | From session log |
| assessment_date | date | ISO format | |

**Integrity:** MD5 hash recorded in Appendix E.4 lock declaration. File is read-only after lock. expert_classification never modified after lock.

### 3. Calibration Examples (data/raw/calibration_examples.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| calibration_id | string | CAL_01–CAL_04 | |
| fragment_id | string | — | NOT in F_001–F_150 range |
| fragment_text | string | — | |
| classification | string | sound, not_sound | |
| rationale | string | — | |
| selection_rationale | string | — | Diversity, complexity, etc. |
| complexity_level | string | straightforward, moderate | |
| primary_characteristic | string | — | |

### 4. Response Database (data/responses/response_database.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| response_id | string | Auto-generated | Primary key |
| fragment_id | string | F_001–F_150 | FK → fragment_corpus |
| model_family | string | gpt_5, claude_opus_4_6, gemini_3_pro, deepseek_v3, kimi_k2, glm_4_7 | |
| prompt_condition | string | zero_shot, few_shot | |
| run_number | integer | 1, 2, 3 | |
| classification_output | string | sound, not_sound, None | None if parsing failed |
| rationale_text | string | — | Raw text |
| timestamp | datetime | ISO format | |
| api_latency_seconds | float | ≥ 0 | |
| token_count_input | integer | ≥ 0 | |
| token_count_output | integer | ≥ 0 | |
| api_version | string | — | Documented at execution |
| parse_method | string | json, text | |
| error_flag | boolean | True/False | |
| error_details | string | — | If error_flag = True |

**Expected rows:** 5,400 (150 × 6 × 2 × 3)

### 5. Scored Database (data/responses/scored_database.csv)

Extends response_database with scoring columns:

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| (all response_database columns) | | | |
| coherence_rule_based | string | coherent, incoherent | |
| strength_indicator_count | integer | ≥ 0 | |
| weakness_indicator_count | integer | ≥ 0 | |
| coherence_llm | string | coherent, incoherent, not_checked | Only for uncertain rule-based cases |
| coherence_final | string | coherent, incoherent, ambiguous | |
| manual_review_flag | boolean | True/False | |
| manual_coherence | string | coherent, incoherent, None | If reviewed |
| classification_accuracy | string | correct, incorrect, None | None if incoherent |
| run_outcome | string | pass, fail | |
| failure_type | string | type_1, type_2, type_3, None | None if pass |
| edge_case_type | string | substantive_refusal, safety_trigger, nonsensical_content, contradictory_claims, None | |
| exclude_from_analysis | boolean | True/False | Only for infrastructure failures |

### 6. Fragment-Level Outcomes (data/responses/fragment_outcomes.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| fragment_id | string | F_001–F_150 | |
| model_family | string | See response_database | |
| prompt_condition | string | zero_shot, few_shot | |
| pass_count | integer | 0, 1, 2, 3 | |
| fail_count | integer | 0, 1, 2, 3 | |
| fragment_outcome | string | pass, fail | Majority rule |
| unanimous_agreement | boolean | True/False | All 3 runs agree |

**Expected rows:** 1,800 (150 × 6 × 2)

### 7. Failure Mode Codes (data/analysis/failure_codes_gpt5.csv)

| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| fragment_id | string | F_001–F_150 | |
| prompt_condition | string | zero_shot, few_shot | |
| representative_run | integer | 1, 2, 3 | Clearest failure pattern |
| primary_code | string | F1.1–F6.6 or emergent | |
| secondary_code | string | F1.1–F6.6, emergent, or None | Optional |
| tertiary_code | string | F1.1–F6.6, emergent, or None | Optional |
| domain | integer | 1–6 or 0 for emergent | |
| code_type | string | deductive, inductive | |
| coding_justification | string | — | |
| coding_date | date | ISO format | |
| coding_session | integer | — | |

**Expected rows:** 60–120 (estimated GPT 5.2 failures across both prompts)

### Table Relationships

    fragment_corpus.fragment_id  ←→  gold_standard.fragment_id        (1:1)
    fragment_corpus.fragment_id  ←→  response_database.fragment_id    (1:many, 36 per fragment)
    response_database            →   scored_database                   (1:1 extension)
    scored_database              →   fragment_outcomes                 (3:1 aggregation via majority rule)
    fragment_outcomes            →   failure_codes_gpt5                (1:1 for GPT 5.2 failures only)
    calibration_examples         →   few_shot prompt template          (embedded in prompt)

## Edge Case Rules

Source: Instrument 6 (Model Output Capture and Scoring Instrument)

### Expected Output Format

- Binary classification: sound / not sound
- Rationale: 2-4 sentences explaining judgment
- Capture method: JSON for models supporting structured output; parsed text otherwise
- Normalization: classification values normalized across model phrasings
- Validation: rationale must contain ≥10 words of substantive reasoning (not mere restatement)

### Edge Case Categories

**Refusals (categorized by source):**
- Substantive refusal (e.g., "I cannot evaluate this conclusion") → Fail (reasoning failure)
- Technical failure (API timeout) → Exclude (infrastructure issue)
- Safety policy trigger → Fail, flagged separately for pattern analysis

**Malformed outputs:**
- Broken format + present reasoning content → 2 retry attempts, then technical exclusion
- Correct format + nonsensical content → Fail (reasoning failure)

**Multi-part judgments:**
- Clear final classification present → extract and score
- Unresolved ambivalence → Fail (cannot synthesize)
- Contradictory claims → Fail (coherence validation)

### Coherence Validation (tiered)

**Tier 1 — Rule-based keyword matching:**
- Match classification label against strength/weakness indicators in rationale
- Example: "sound" + "fatal flaws" or "missing evidence" → flagged incoherent
- Confident signal → assign automatically
- Mixed/balanced/insufficient → route to Tier 2

**Tier 2 — LLM screen (Claude Haiku, temperature 0):**
- Input: classification + rationale text ONLY
- Output: binary coherent/incoherent
- Agreement with Tier 1 → assign accordingly
- Disagreement → flag "ambiguous" → manual review

**Coherence outcome:** Classification-rationale contradiction → Fail regardless of classification accuracy

### Scoring Protocol (two stages)

1. Coherence validation: screen out incoherent outputs
2. Classification comparison: match remaining outputs against gold standard

**Pass requirement:** classification agreement AND reasoning coherence

### Performance Metrics

- Overall accuracy (pass rate) stratified by:
  - Failure type (coherence failure, classification error, refusal)
  - Fragment characteristics

### Implementation

Python scripts using pandas, JSON, and supporting libraries. See Appendix G.

```