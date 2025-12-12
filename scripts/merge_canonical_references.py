"""Merge canonical references across all episodes using three-pass strategy.

Pass 1: Entity consolidation via LLM
Pass 2: Scene hierarchical clustering via LLM
Pass 3: Theory embedding similarity + LLM clustering
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.openai_classifier import OpenAIClassifier


def setup_logging() -> logging.Logger:
    """Setup logging configuration.

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_all_episode_contexts(data_dir: Path) -> list[dict[str, Any]]:
    """Load all episode context files.

    Args:
        data_dir: Path to data directory

    Returns:
        List of episode context dictionaries
    """
    contexts = []
    for episode in range(2, 10):
        context_file = data_dir / f"context/s01e0{episode}_context.json"
        with open(context_file) as f:
            contexts.append(json.load(f))
    return contexts


def merge_entities(contexts: list[dict[str, Any]], classifier: OpenAIClassifier, logger: logging.Logger) -> dict[str, Any]:
    """Pass 1: Merge entities across all episodes via LLM.

    Args:
        contexts: List of episode contexts
        classifier: OpenAI classifier instance
        logger: Logger instance

    Returns:
        Dictionary with merged entities and mapping
    """
    logger.info("=== PASS 1: ENTITY CONSOLIDATION ===")

    # Collect all entities from all episodes
    all_entities = []
    for ctx in contexts:
        for entity in ctx['entities']:
            all_entities.append({
                'canonical_name': entity['canonical_name'],
                'aliases': entity['aliases'],
                'type': entity['type'],
                'episode': ctx['metadata']['episode']
            })

    logger.info(f"collected {len(all_entities)} entities from {len(contexts)} episodes")

    # Build prompt for LLM
    prompt = f"""You are merging character/entity lists from a TV show discussion analysis across multiple episodes.

INPUT: {len(all_entities)} entities from episodes 2-9

ENTITIES TO MERGE:
{json.dumps(all_entities, indent=2)}

TASK:
1. Identify which entities refer to the same character/organization/location
2. For duplicates, pick the MOST COMPLETE canonical name
3. Merge all aliases together
4. Output consolidated list

RULES:
- "Cobel", "Harmony Cobel", "Ms. Cobel" → single entity "Harmony Cobel"
- "Helly R.", "Helly R", "Helly" → single entity "Helly R."
- "Milchick", "Milchik", "Milchek" → single entity "Milchick" (correct spelling)
- Keep entity types (character/organization/location)
- Preserve ALL aliases from all episodes

OUTPUT FORMAT:
{{
  "merged_entities": [
    {{
      "canonical_name": "Harmony Cobel",
      "aliases": ["Cobel", "Ms. Cobel", "Mrs. Selvig", "Harmony", ...],
      "type": "character",
      "source_episodes": [2, 3, 4, 5, 6, 7, 8, 9]
    }},
    ...
  ],
  "entity_mapping": {{
    "Cobel": "Harmony Cobel",
    "Ms. Cobel": "Harmony Cobel",
    "Harmony Cobel": "Harmony Cobel",
    ...
  }}
}}

Return ONLY valid JSON, no other text.
"""

    logger.info("sending entity merge request to LLM...")
    response = classifier.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Calculate cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

    logger.info(f"entity merge complete: {len(all_entities)} → {len(result['merged_entities'])} entities")
    logger.info(f"cost: ${cost:.4f}")

    return result


def merge_scenes(contexts: list[dict[str, Any]], classifier: OpenAIClassifier, logger: logging.Logger) -> dict[str, Any]:
    """Pass 2: Create hierarchical scene clustering via LLM.

    Args:
        contexts: List of episode contexts
        classifier: OpenAI classifier instance
        logger: Logger instance

    Returns:
        Dictionary with hierarchical scenes
    """
    logger.info("=== PASS 2: SCENE HIERARCHICAL CLUSTERING ===")

    # Collect all scenes from all episodes
    all_scenes = []
    for ctx in contexts:
        for scene in ctx['scenes']:
            all_scenes.append({
                'canonical_description': scene['canonical_description'],
                'variations': scene['variations'],
                'episode': ctx['metadata']['episode'],
                'mention_count': scene['mention_count']
            })

    logger.info(f"collected {len(all_scenes)} scenes from {len(contexts)} episodes")

    # Build prompt for LLM
    prompt = f"""You are creating a hierarchical scene structure for a TV show discussion analysis.

INPUT: {len(all_scenes)} scenes from episodes 2-9

SCENES:
{json.dumps(all_scenes, indent=2)}

TASK:
Create a 2-level hierarchy:
1. MAJOR PLOT MOMENTS - Broad story beats (8-12 total)
2. MICRO-MOMENTS - Specific scenes within each major moment (40-50 total)

EXAMPLE HIERARCHY:
{{
  "major_moments": [
    {{
      "id": 1,
      "canonical_description": "Mark discovers Gemma is alive",
      "micro_moments": [
        {{
          "id": 1,
          "canonical_description": "Mark sees Gemma/Ms. Casey in wellness session",
          "episodes": [7],
          "mention_count": 58
        }},
        {{
          "id": 2,
          "canonical_description": "Mark reveals to Devon that Gemma is alive",
          "episodes": [9],
          "mention_count": 263
        }},
        {{
          "id": 3,
          "canonical_description": "Mark screams 'she's alive' at the party",
          "episodes": [9],
          "mention_count": 150
        }}
      ]
    }},
    ...
  ]
}}

RULES:
- Group related micro-moments under major moments
- Preserve episode numbers and mention counts
- Major moments should be season-spanning plot beats
- Micro-moments are specific memorable scenes

Return ONLY valid JSON, no other text.
"""

    logger.info("sending scene hierarchy request to GPT-4o...")
    response = classifier.client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Calculate cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)

    total_micro_moments = sum(len(major['micro_moments']) for major in result['major_moments'])

    logger.info(f"scene hierarchy complete: {len(result['major_moments'])} major moments, {total_micro_moments} micro-moments")
    logger.info(f"cost: ${cost:.4f}")

    return result


def merge_theories(contexts: list[dict[str, Any]], classifier: OpenAIClassifier, logger: logging.Logger) -> dict[str, Any]:
    """Pass 3: Merge theories using embedding similarity + LLM clustering.

    Args:
        contexts: List of episode contexts
        classifier: OpenAI classifier instance
        logger: Logger instance

    Returns:
        Dictionary with merged theories
    """
    logger.info("=== PASS 3: THEORY EMBEDDING SIMILARITY + LLM CLUSTERING ===")

    # Step 3a: Collect all theories and generate embeddings
    all_theories = []
    for ctx in contexts:
        for theory in ctx['theories']:
            all_theories.append({
                'canonical_claim': theory['canonical_claim'],
                'category': theory['category'],
                'consensus': theory['consensus'],
                'episode': ctx['metadata']['episode'],
                'mention_count': theory['mention_count'],
                'sample_phrasings': theory.get('sample_phrasings', [])
            })

    logger.info(f"collected {len(all_theories)} theories from {len(contexts)} episodes")
    logger.info("generating embeddings for all theories...")

    # Generate embeddings
    embedding_response = classifier.client.embeddings.create(
        model="text-embedding-3-small",
        input=[t['canonical_claim'] for t in all_theories]
    )

    embeddings = np.array([e.embedding for e in embedding_response.data])
    embedding_cost = embedding_response.usage.total_tokens * 0.02 / 1_000_000

    logger.info(f"embeddings generated: {embeddings.shape}")
    logger.info(f"embedding cost: ${embedding_cost:.4f}")

    # Step 3b: Cluster by cosine similarity
    logger.info("clustering theories by similarity...")
    similarity_matrix = cosine_similarity(embeddings)

    THRESHOLD = 0.85
    clusters = []
    visited = set()

    for i in range(len(all_theories)):
        if i in visited:
            continue

        cluster = [i]
        visited.add(i)

        for j in range(i + 1, len(all_theories)):
            if j not in visited and similarity_matrix[i][j] > THRESHOLD:
                cluster.append(j)
                visited.add(j)

        clusters.append(cluster)

    logger.info(f"found {len(clusters)} theory clusters (threshold: {THRESHOLD})")

    # Step 3c: Use LLM to pick best canonical claim for each cluster
    logger.info("using LLM to refine clusters...")

    merged_theories = []
    total_llm_cost = 0.0

    for cluster_idx, cluster in enumerate(clusters, 1):
        cluster_theories = [all_theories[i] for i in cluster]

        if len(cluster_theories) == 1:
            # No merge needed, use as-is
            theory = cluster_theories[0]
            merged_theories.append({
                'id': cluster_idx,
                'canonical_claim': theory['canonical_claim'],
                'category': theory['category'],
                'consensus': theory['consensus'],
                'total_mentions': theory['mention_count'],
                'episodes': [theory['episode']],
                'evolution': [{
                    'episode': theory['episode'],
                    'claim': theory['canonical_claim']
                }]
            })
            continue

        # Multiple theories in cluster, use LLM to merge
        prompt = f"""You are merging duplicate theories from a TV show discussion analysis.

THEORIES TO MERGE (all semantically similar):
{json.dumps(cluster_theories, indent=2)}

TASK:
1. Pick the CLEAREST, MOST SPECIFIC canonical_claim from the list
2. Prefer claims from later episodes (they have more context/confirmation)
3. Sum mention_counts across all theories
4. Track evolution across episodes

OUTPUT FORMAT:
{{
  "canonical_claim": "...",
  "category": "...",
  "consensus": "strong_support|moderate_support|weak_support",
  "total_mentions": 123,
  "episodes": [2, 5, 7],
  "evolution": [
    {{"episode": 2, "claim": "early speculation..."}},
    {{"episode": 5, "claim": "more specific..."}},
    {{"episode": 7, "claim": "confirmed..."}}
  ]
}}

Return ONLY valid JSON, no other text.
"""

        response = classifier.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result['id'] = cluster_idx
        merged_theories.append(result)

        # Track cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_llm_cost += (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        if cluster_idx % 50 == 0:
            logger.info(f"processed {cluster_idx}/{len(clusters)} clusters...")

    logger.info(f"theory merge complete: {len(all_theories)} → {len(merged_theories)} theories")
    logger.info(f"LLM refinement cost: ${total_llm_cost:.4f}")
    logger.info(f"total theory cost: ${embedding_cost + total_llm_cost:.4f}")

    return {
        'merged_theories': merged_theories,
        'original_count': len(all_theories),
        'merged_count': len(merged_theories)
    }


def main() -> None:
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting cross-episode canonical merge...")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_file = data_dir / "merged_canonicals.json"

    # Load API key
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = Path.home() / "OPENAI_API_KEY"
        if key_file.exists():
            api_key = key_file.read_text().strip()
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            logger.error("OPENAI_API_KEY not found")
            return

    # Load episode contexts
    logger.info("Loading episode contexts...")
    contexts = load_all_episode_contexts(data_dir)
    logger.info(f"Loaded {len(contexts)} episode contexts")

    # Initialize classifier
    classifier = OpenAIClassifier()

    # Pass 1: Merge entities
    entity_result = merge_entities(contexts, classifier, logger)

    # Pass 2: Hierarchical scenes
    scene_result = merge_scenes(contexts, classifier, logger)

    # Pass 3: Theory clustering
    theory_result = merge_theories(contexts, classifier, logger)

    # Combine results
    output = {
        'entities': entity_result,
        'scenes': scene_result,
        'theories': theory_result,
        'metadata': {
            'source_episodes': list(range(2, 10)),
            'merge_date': str(Path(__file__).stat().st_mtime)
        }
    }

    # Save output
    logger.info(f"Saving merged canonicals to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("=== MERGE COMPLETE ===")
    logger.info(f"Entities: {len(entity_result['merged_entities'])}")
    logger.info(f"Major moments: {len(scene_result['major_moments'])}")
    logger.info(f"Theories: {len(theory_result['merged_theories'])}")


if __name__ == "__main__":
    main()
