"""Extract canonical entity/scene/theory lists from full episode context.

This is Pass 1 of the two-pass classification architecture.
Processes all comments from an episode as a single document to extract
canonical references that individual comments will later link to.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.openai_classifier import OpenAIClassifier


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to file and stdout.

    Args:
        log_file: path to log file

    Returns:
        configured logger instance
    """
    logger = logging.getLogger("extract_context")
    logger.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_episode_comments(episode_file: Path) -> dict:
    """Load episode comments from parsed JSON file.

    Args:
        episode_file: path to parsed episode JSON file

    Returns:
        dict with thread metadata and comments list
    """
    with open(episode_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context_extraction_prompt(
    season: int,
    episode: int,
    episode_title: str,
    all_comments: str
) -> str:
    """Build prompt for extracting canonical references from episode.

    Args:
        season: season number
        episode: episode number
        episode_title: episode title
        all_comments: concatenated comment text

    Returns:
        formatted prompt string
    """
    return f"""You are analyzing ALL comments from a Reddit episode discussion thread.

**Episode:** Severance - Season {season}, Episode {episode}: "{episode_title}"

**Task:** Extract canonical reference lists by analyzing all comments together.

**Comments to analyze:**
{all_comments}

**Instructions:**

1. **Extract all CHARACTERS mentioned across all comments:**
   - Provide canonical full name (e.g., "Mark Scout" not "Mark S.")
   - List ALL variations/aliases used (e.g., ["Mark", "Mark S.", "MS", "mark scout"])
   - Classify type: character, location, organization, concept, object
   - Estimate total mentions across all comments
   - High confidence for clearly identified characters
   - Medium confidence for ambiguous references
   - Low confidence for unclear mentions

2. **Extract all SCENES discussed across comments:**
   - Identify distinct scenes/moments from the episode
   - Group different descriptions of the SAME scene together
   - Provide ONE canonical description per scene (≤15 words)
   - List variations: how different commenters described it
   - Estimate mentions per scene
   - Focus on specific moments, not vague references

3. **Extract all THEORIES/PREDICTIONS mentioned:**
   - Group similar theories together (don't create separate entries for minor wording differences)
   - Categorize: character_identity, plot_prediction, symbolism, workplace_nature, conspiracy, other
   - Provide canonical claim (≤25 words)
   - Assess consensus: strong_support, moderate_support, weak_support, rejected, mixed
   - Estimate mentions
   - List sample phrasings from comments

**Output as JSON with this EXACT schema:**

{{
  "entities": [
    {{
      "canonical_name": "Mark Scout",
      "aliases": ["Mark", "Mark S.", "MS", "mark scout", "our boy mark"],
      "type": "character",
      "confidence": "high",
      "mention_count": 59
    }}
  ],
  "scenes": [
    {{
      "id": 1,
      "canonical_description": "Helly's escape attempt in the elevator",
      "variations": [
        "when Helly tried to escape",
        "the elevator scene",
        "Helly's first day escape"
      ],
      "confidence": "high",
      "mention_count": 12
    }}
  ],
  "theories": [
    {{
      "category": "character_identity",
      "canonical_claim": "Helly is a member of the Eagan family",
      "consensus": "strong_support",
      "confidence": "high",
      "mention_count": 15,
      "sample_phrasings": [
        "Helly is an Eagan",
        "She's definitely related to the Eagans",
        "The Helly/Eagan connection is obvious"
      ]
    }}
  ]
}}

**Important guidelines:**
- Consolidate aggressively: "Irving" and "Irv" are the SAME character
- Use most common/full names as canonical
- Group similar scene descriptions (even if worded differently)
- Merge theories that express the same core idea
- Only create separate entries when genuinely distinct
- Be thorough but avoid over-fragmentation
- Return ONLY valid JSON, no other text

**Output:**"""


def extract_episode_context(
    season: int,
    episode: int,
    episode_title: str,
    comments: list[dict],
    classifier: OpenAIClassifier,
    logger: logging.Logger
) -> dict:
    """Extract canonical references from all episode comments.

    Args:
        season: season number
        episode: episode number
        episode_title: episode title
        comments: list of comment dicts
        classifier: OpenAIClassifier instance
        logger: logger instance

    Returns:
        dict with entities, scenes, theories
    """
    # concatenate comments with 64k token limit
    logger.info(f"concatenating comments (max 64k tokens)...")

    MAX_TOKENS = 64000
    token_budget = MAX_TOKENS

    selected_comments = []
    for comment in comments:
        comment_tokens = len(comment['body']) // 4
        if token_budget - comment_tokens < 0:
            break
        selected_comments.append(comment)
        token_budget -= comment_tokens

    logger.info(f"selected {len(selected_comments)} of {len(comments)} comments ({100*len(selected_comments)/len(comments):.1f}%)")
    logger.info(f"estimated tokens used: {MAX_TOKENS - token_budget:,} / {MAX_TOKENS:,}")

    comment_texts = []
    for i, comment in enumerate(selected_comments, 1):
        # include comment number for reference
        comment_texts.append(f"[Comment {i}] {comment['body']}")

    all_comments = "\n\n".join(comment_texts)

    total_chars = len(all_comments)
    estimated_tokens = total_chars // 4

    logger.info(f"final characters: {total_chars:,}")
    logger.info(f"final tokens: {estimated_tokens:,}")

    # build prompt
    prompt = build_context_extraction_prompt(
        season=season,
        episode=episode,
        episode_title=episode_title,
        all_comments=all_comments
    )

    prompt_tokens = len(prompt) // 4
    logger.info(f"total prompt tokens (estimated): {prompt_tokens:,}")

    # call LLM
    logger.info("calling openai api for context extraction...")

    response = classifier.client.chat.completions.create(
        model=classifier.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,  # generous for comprehensive output
        temperature=0,
        response_format={"type": "json_object"}
    )

    result_text = response.choices[0].message.content
    context = json.loads(result_text)

    # calculate cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    input_cost = input_tokens * 0.15 / 1_000_000
    output_cost = output_tokens * 0.60 / 1_000_000
    total_cost = input_cost + output_cost

    logger.info(f"extraction complete")
    logger.info(f"input tokens: {input_tokens:,}")
    logger.info(f"output tokens: {output_tokens:,}")
    logger.info(f"total tokens: {total_tokens:,}")
    logger.info(f"cost: ${total_cost:.4f}")

    # log extraction counts
    entity_count = len(context.get("entities", []))
    scene_count = len(context.get("scenes", []))
    theory_count = len(context.get("theories", []))

    logger.info(f"extracted {entity_count} canonical entities")
    logger.info(f"extracted {scene_count} canonical scenes")
    logger.info(f"extracted {theory_count} canonical theories")

    # add metadata
    context["metadata"] = {
        "season": season,
        "episode": episode,
        "episode_title": episode_title,
        "comment_count": len(comments),
        "extraction_date": datetime.now().isoformat(),
        "model": classifier.model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": total_cost
    }

    return context


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("usage: python extract_episode_context.py <episode_number>")
        print("example: python extract_episode_context.py 2")
        sys.exit(1)

    episode_num = int(sys.argv[1])
    episode_file = Path(f"data/parsed/s01e{episode_num:02d}.json")

    if not episode_file.exists():
        print(f"error: {episode_file} not found")
        sys.exit(1)

    # create output directories
    context_dir = Path("data/context")
    context_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # output files
    output_file = context_dir / f"s01e{episode_num:02d}_context.json"
    log_file = log_dir / f"extract_context_s01e{episode_num:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # setup logging
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info(f"context extraction started: episode {episode_num}")
    logger.info(f"log file: {log_file}")
    logger.info("=" * 60)

    # load episode data
    logger.info(f"loading episode {episode_num}...")
    episode_data = load_episode_comments(episode_file)
    thread = episode_data["thread"]
    comments = episode_data["comments"]

    logger.info(f"episode: {thread['title']}")
    logger.info(f"total comments: {len(comments)}")

    # initialize classifier
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # try loading from file
        key_file = Path.home() / "OPENAI_API_KEY"
        if key_file.exists():
            api_key = key_file.read_text().strip()

    if not api_key:
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    classifier = OpenAIClassifier(api_key=api_key)

    # extract context
    logger.info("starting context extraction...")

    context = extract_episode_context(
        season=thread["season"],
        episode=thread["episode"],
        episode_title=thread["title"],
        comments=comments,
        classifier=classifier,
        logger=logger
    )

    # save context
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    logger.info(f"context saved to {output_file}")

    # display summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"entities: {len(context['entities'])}")
    logger.info(f"scenes: {len(context['scenes'])}")
    logger.info(f"theories: {len(context['theories'])}")
    logger.info(f"cost: ${context['metadata']['cost']:.4f}")

    # show top entities
    logger.info("\ntop 10 entities:")
    entities_sorted = sorted(
        context['entities'],
        key=lambda x: x.get('mention_count', 0),
        reverse=True
    )
    for i, entity in enumerate(entities_sorted[:10], 1):
        aliases_str = ", ".join(entity.get('aliases', [])[:3])
        logger.info(f"  {i}. {entity['canonical_name']} ({entity['type']}) - {entity.get('mention_count', 0)} mentions")
        logger.info(f"     aliases: {aliases_str}...")

    # show top theories
    logger.info("\ntop 10 theories:")
    theories_sorted = sorted(
        context['theories'],
        key=lambda x: x.get('mention_count', 0),
        reverse=True
    )
    for i, theory in enumerate(theories_sorted[:10], 1):
        logger.info(f"  {i}. [{theory['category']}] {theory['canonical_claim']}")
        logger.info(f"     consensus: {theory['consensus']}, mentions: {theory.get('mention_count', 0)}")

    logger.info("=" * 60)
    logger.info("context extraction complete")


if __name__ == "__main__":
    main()
