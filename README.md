# Severance Reddit Discussion Analytics

Extract and analyze character mentions, scenes, and theories from Reddit TV show discussions using a three-pass LLM architecture.
The purpose is to take unstructured, raw reddit comments, and categorize the following:
- Entities (characters, settings, concepts)
- Theories (fan theories, predictions)
- Sentiments (scared, confused, happy, etc)

The frontend consists of visualizations to track the popularity of certain characters and scenes over time.

## How the Software is Implemented

The software is implemented as a 3-pass data pipeline.

**Pass 1:** Feed the LLM the first 64k tokens of all comments together, and prompt for a canonical list of characters, entities, etc. This avoids having to merge misspellings, aliases, etc. (e.g. "mark", "Mark", "Mark S").

**Pass 2:** With a list of canonical scenes, theories, and entities derived from pass 1, having the full context of the discussion as a whole, the LLM classifies each comment individually, attributing each comment to sentiment, theories, entities, scenes.

**Pass 3:** After Pass 1 and Pass 2 are performed for each episode, a final pass unifies disparate canonical references across episodes. Entity consolidation uses LLM matching. Scene consolidation builds a two-level hierarchy (major moments contain micro-moments). NOTE: Theory consolidation uses embedding similarity clustering to group semantically similar theories, then an LLM picks the best phrasing.

**Stack** OpenAI API (GPT-4o-mini for classification, GPT-4o for scene hierarchy, text-embedding-3-small for theory clustering) and Streamlit frontend with Plotly visualizations.

## Quick Start

NOTE TO GRADERS: running the full pipeline takes a few hours and costs around ~$5-10. Recommend to follow the "Quick Demo" section with the preprocessed data provided in this repo

### Prerequisites
- Python 3.12+
- OpenAI API key (if you want to run the full pipeline)
- Reddit comment data. (if you want to run the full pipeline) https://the-eye.eu/redarcs. Download the `submissions.zst` and `comments.zst` files, and place them under `data/raw_reddit` directory
- NOTE: As of Nov 2025, Reddit only issues new API keys on an approval basis, so we can only use archived comment data, instead of ingesting data directly from the API, unfortunately: https://www.reddit.com/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/

### Quick Demo

Installation
```bash
pip install -r requirements.txt
```

Run the dashboard using the pre-processed provided data
```bash
streamlit run app.py
```

Open http://localhost:8501

## Data Source

Reddit archive for /r/SeveranceAppleTVPlus (Season 1, Episodes 2-9)
- 14,012 comments analyzed
- https://the-eye.eu/redarcs/

## Architecture: Three-Pass Pipeline

### Pass 1: Extract Canonical References (per episode)
Extract entities, scenes, theories from full episode context
- **Script:** `scripts/extract_episode_context.py`
- **Model:** GPT-4o-mini
- **Output:** `data/context/s01e0X_context.json`

### Pass 2: Link Comments (per episode)
Link individual comments to canonical references
- **Script:** `scripts/classify_episode_v2.py`
- **Model:** GPT-4o-mini
- **Output:** `data/classifications_v2/s01e0X_classifications.json`

### Pass 3: Cross-Episode Merge
Consolidate canonicals across all episodes using hybrid approach:
1. **Entity consolidation** - LLM merge (GPT-4o-mini)
2. **Scene hierarchical clustering** - LLM hierarchy (GPT-4o)
3. **Theory embedding similarity** - text-embedding-3-small + LLM refinement

- **Script:** `scripts/merge_canonical_references.py`
- **Output:** `data/merged_canonicals.json`

### Export to CSV
Convert to flat CSV files for analysis
- **Script:** `scripts/export_to_csv.py`
- **Output:** `data/csv_exports/*.csv` (8 CSV files)

## Running the Full Pipeline

NOTE: To see a demo of the website using the pre-processed Severance sample data provided in this repo, skip to STEP 5.

```bash

# 0. Open AI key 
```
echo "your-openai-api-key" > ~/OPENAI_API_KEY
```
# 1. Parse Reddit archive (one-time setup)
python scripts/parse_archive_data.py

# 2. Extract context + classify (parallel for all episodes)
for ep in {2..9}; do
  python scripts/extract_episode_context.py $ep &
done
wait

for ep in {2..9}; do
  python scripts/classify_episode_v2.py $ep &
done
wait

# 3. Merge canonicals across episodes
python scripts/merge_canonical_references.py

# 4. Export to CSV
python scripts/export_to_csv.py

# 5. Run dashboard
streamlit run app.py
```
