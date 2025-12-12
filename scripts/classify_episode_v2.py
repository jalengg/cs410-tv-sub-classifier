"""Two-pass batch classification script (Pass 2: linking to canonical references).

Requires canonical context file from Pass 1 (extract_episode_context.py).
Links individual comments to canonical entities/scenes/theories.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.openai_classifier import OpenAIClassifier
from src.classifier.prompt_templates_v2 import build_linking_prompt
from src.utils.comment_filter import CommentFilter


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to file and stdout.

    Args:
        log_file: path to log file

    Returns:
        configured logger instance
    """
    logger = logging.getLogger("classify_v2")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

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


def load_canonical_context(context_file: Path) -> dict:
    """Load canonical context from Pass 1.

    Args:
        context_file: path to context JSON file

    Returns:
        dict with entities, scenes, theories
    """
    with open(context_file, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_complex_comments(comments: list[dict], comment_filter: CommentFilter) -> list[dict]:
    """Filter comments by complexity.

    Args:
        comments: list of comment dicts with 'body' field
        comment_filter: configured CommentFilter instance

    Returns:
        list of complex comments
    """
    complex_comments = []
    for comment in comments:
        if comment_filter.is_complex(comment["body"]):
            complex_comments.append(comment)
    return complex_comments


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint data if exists.

    Args:
        checkpoint_file: path to checkpoint JSON file

    Returns:
        dict with 'processed' list and 'classifications' dict
    """
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": [], "classifications": {}}


def save_checkpoint(checkpoint_file: Path, checkpoint_data: dict) -> None:
    """Save checkpoint data.

    Args:
        checkpoint_file: path to checkpoint JSON file
        checkpoint_data: dict with 'processed' and 'classifications'
    """
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)


def classify_comment_with_context(
    comment_body: str,
    season: int,
    episode: int,
    episode_title: str,
    canonical_context: dict,
    classifier: OpenAIClassifier
) -> dict:
    """Classify a comment by linking to canonical references.

    Args:
        comment_body: comment text
        season: season number
        episode: episode number
        episode_title: episode title
        canonical_context: canonical entities/scenes/theories from Pass 1
        classifier: OpenAIClassifier instance

    Returns:
        classification dict with entity_refs, scene_refs, theory_refs, sentiments
    """
    prompt = build_linking_prompt(
        comment_body=comment_body,
        season=season,
        episode=episode,
        episode_title=episode_title,
        canonical_entities=canonical_context.get("entities", []),
        canonical_scenes=canonical_context.get("scenes", []),
        canonical_theories=canonical_context.get("theories", [])
    )

    response = classifier.client.chat.completions.create(
        model=classifier.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0,
        response_format={"type": "json_object"}
    )

    result_text = response.choices[0].message.content
    classification = json.loads(result_text)

    # calculate cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    input_cost = input_tokens * 0.15 / 1_000_000
    output_cost = output_tokens * 0.60 / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "entity_refs": classification.get("entity_refs", []),
        "scene_refs": classification.get("scene_refs", []),
        "theory_refs": classification.get("theory_refs", []),
        "new_entities": classification.get("new_entities", []),
        "new_scenes": classification.get("new_scenes", []),
        "new_theories": classification.get("new_theories", []),
        "sentiments": classification.get("sentiments", []),
        "metadata": {
            "model": classifier.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost
        }
    }


def classify_episode(
    season: int,
    episode: int,
    episode_title: str,
    comments: list[dict],
    canonical_context: dict,
    classifier: OpenAIClassifier,
    checkpoint_file: Path,
    logger: logging.Logger,
    progress_interval: int = 10,
) -> dict:
    """Classify all comments in episode with checkpointing.

    Args:
        season: season number
        episode: episode number
        episode_title: episode title
        comments: list of comment dicts
        canonical_context: canonical references from Pass 1
        classifier: OpenAIClassifier instance
        checkpoint_file: path to checkpoint file
        logger: logger instance
        progress_interval: save checkpoint every N comments

    Returns:
        dict with classification results and metadata
    """
    checkpoint = load_checkpoint(checkpoint_file)
    processed_ids = set(checkpoint["processed"])
    classifications = checkpoint["classifications"]

    total = len(comments)
    start_idx = len(processed_ids)

    logger.info(f"processing {total} comments (resuming from {start_idx})...")

    # metrics
    start_time = time.time()
    total_cost = 0.0
    errors = []

    for i, comment in enumerate(comments):
        comment_id = comment["id"]

        # skip if already processed
        if comment_id in processed_ids:
            continue

        try:
            # classify comment
            result = classify_comment_with_context(
                comment_body=comment["body"],
                season=season,
                episode=episode,
                episode_title=episode_title,
                canonical_context=canonical_context,
                classifier=classifier
            )

            # store result
            classifications[comment_id] = {
                "comment": {
                    "id": comment_id,
                    "author": comment.get("author"),
                    "body": comment["body"],
                    "score": comment.get("score"),
                    "created_utc": comment.get("created_utc"),
                },
                "classification": result,
            }

            # mark as processed
            processed_ids.add(comment_id)
            checkpoint["processed"] = list(processed_ids)

            # update metrics
            cost = result['metadata']['cost']
            total_cost += cost

            # progress update
            current = len(processed_ids)
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining_time = (total - current) / rate if rate > 0 else 0

            logger.info(
                f"[{current}/{total}] {comment_id} | "
                f"cost: ${cost:.5f} | "
                f"total: ${total_cost:.3f} | "
                f"rate: {rate:.1f}/min | "
                f"eta: {remaining_time/60:.1f}m"
            )

            # save checkpoint periodically
            if current % progress_interval == 0:
                save_checkpoint(checkpoint_file, checkpoint)
                logger.info(f"checkpoint saved at {current}/{total}")

            # rate limiting (avoid hitting API limits)
            time.sleep(0.1)

        except Exception as e:
            error_msg = f"error processing {comment_id}: {e}"
            logger.error(error_msg)
            errors.append({"comment_id": comment_id, "error": str(e)})

            # save checkpoint on error
            save_checkpoint(checkpoint_file, checkpoint)

            # continue processing other comments
            continue

    # final checkpoint
    save_checkpoint(checkpoint_file, checkpoint)

    # final metrics
    elapsed_total = time.time() - start_time
    logger.info(f"classification complete: {len(processed_ids)}/{total} comments")
    logger.info(f"total time: {elapsed_total/60:.1f}m")
    logger.info(f"total cost: ${total_cost:.3f}")
    logger.info(f"errors: {len(errors)}")

    return {
        "classifications": classifications,
        "metrics": {
            "total_comments": total,
            "processed": len(processed_ids),
            "errors": len(errors),
            "total_cost": total_cost,
            "elapsed_seconds": elapsed_total,
            "rate_per_minute": len(processed_ids) / (elapsed_total / 60) if elapsed_total > 0 else 0
        },
        "errors": errors
    }


def calculate_statistics(classifications: dict, canonical_context: dict) -> dict:
    """Calculate statistics from classification results.

    Args:
        classifications: dict of comment_id -> classification result
        canonical_context: canonical references from Pass 1

    Returns:
        dict with statistics
    """
    total_comments = len(classifications)
    total_cost = sum(
        c["classification"]["metadata"]["cost"]
        for c in classifications.values()
    )

    # count canonical references
    entity_ref_counts = {}
    scene_ref_counts = {}
    theory_ref_counts = {}

    # count new entities/scenes/theories created
    new_entity_counts = {}
    new_scene_count = 0
    new_theory_count = 0

    sentiment_counts = {}

    for result in classifications.values():
        classification = result["classification"]

        # canonical entity refs
        for ref in classification.get("entity_refs", []):
            name = ref["canonical_name"]
            entity_ref_counts[name] = entity_ref_counts.get(name, 0) + 1

        # new entities
        for entity in classification.get("new_entities", []):
            name = entity["name"].lower()
            new_entity_counts[name] = new_entity_counts.get(name, 0) + 1

        # scene refs
        for ref in classification.get("scene_refs", []):
            scene_id = ref["scene_id"]
            scene_ref_counts[scene_id] = scene_ref_counts.get(scene_id, 0) + 1

        # new scenes
        new_scene_count += len(classification.get("new_scenes", []))

        # theory refs
        for ref in classification.get("theory_refs", []):
            theory_id = ref["theory_id"]
            theory_ref_counts[theory_id] = theory_ref_counts.get(theory_id, 0) + 1

        # new theories
        new_theory_count += len(classification.get("new_theories", []))

        # sentiments
        for sentiment in classification.get("sentiments", []):
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    return {
        "total_comments": total_comments,
        "total_cost": total_cost,
        "canonical_entity_refs": sorted(entity_ref_counts.items(), key=lambda x: x[1], reverse=True),
        "new_entities": sorted(new_entity_counts.items(), key=lambda x: x[1], reverse=True),
        "canonical_scene_refs": sorted(scene_ref_counts.items(), key=lambda x: x[1], reverse=True),
        "new_scenes_count": new_scene_count,
        "canonical_theory_refs": sorted(theory_ref_counts.items(), key=lambda x: x[1], reverse=True),
        "new_theories_count": new_theory_count,
        "top_sentiments": sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True)[:10],
    }


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("usage: python classify_episode_v2.py <episode_number>")
        print("example: python classify_episode_v2.py 2")
        sys.exit(1)

    episode_num = int(sys.argv[1])
    episode_file = Path(f"data/parsed/s01e{episode_num:02d}.json")
    context_file = Path(f"data/context/s01e{episode_num:02d}_context.json")

    if not episode_file.exists():
        print(f"error: {episode_file} not found")
        sys.exit(1)

    if not context_file.exists():
        print(f"error: {context_file} not found")
        print(f"run: python scripts/extract_episode_context.py {episode_num}")
        sys.exit(1)

    # create output directories
    output_dir = Path("data/classifications_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("data/checkpoints_v2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"s01e{episode_num:02d}_checkpoint.json"
    output_file = output_dir / f"s01e{episode_num:02d}_classifications.json"
    stats_file = output_dir / f"s01e{episode_num:02d}_stats.json"
    metrics_file = output_dir / f"s01e{episode_num:02d}_metrics.json"
    log_file = log_dir / f"classify_v2_s01e{episode_num:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # setup logging
    logger = setup_logging(log_file)

    logger.info(f"=" * 60)
    logger.info(f"two-pass classification (pass 2) started: episode {episode_num}")
    logger.info(f"log file: {log_file}")
    logger.info(f"=" * 60)

    # load episode data
    logger.info(f"loading episode {episode_num}...")
    episode_data = load_episode_comments(episode_file)
    thread = episode_data["thread"]
    comments = episode_data["comments"]

    logger.info(f"episode: {thread['title']}")
    logger.info(f"total comments: {len(comments)}")

    # load canonical context
    logger.info(f"loading canonical context from pass 1...")
    canonical_context = load_canonical_context(context_file)

    logger.info(f"canonical entities: {len(canonical_context.get('entities', []))}")
    logger.info(f"canonical scenes: {len(canonical_context.get('scenes', []))}")
    logger.info(f"canonical theories: {len(canonical_context.get('theories', []))}")

    # filter complex comments
    logger.info("\nfiltering complex comments...")
    comment_filter = CommentFilter()
    complex_comments = filter_complex_comments(comments, comment_filter)

    logger.info(f"complex comments: {len(complex_comments)} ({len(complex_comments)/len(comments)*100:.1f}%)")

    # estimate cost
    estimated_cost = len(complex_comments) * 0.00042
    logger.info(f"estimated cost: ${estimated_cost:.2f}")

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

    # classify comments
    logger.info("\nstarting classification...")
    result = classify_episode(
        season=thread["season"],
        episode=thread["episode"],
        episode_title=thread["title"],
        comments=complex_comments,
        canonical_context=canonical_context,
        classifier=classifier,
        checkpoint_file=checkpoint_file,
        logger=logger,
    )

    # save final results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["classifications"], f, indent=2)

    logger.info(f"\nclassifications saved to {output_file}")

    # save metrics
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    logger.info(f"metrics saved to {metrics_file}")

    # calculate and save statistics
    stats = calculate_statistics(result["classifications"], canonical_context)

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"statistics saved to {stats_file}")

    # display statistics
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION STATISTICS (TWO-PASS)")
    logger.info("="*60)
    logger.info(f"total comments classified: {stats['total_comments']}")
    logger.info(f"total cost: ${stats['total_cost']:.2f}")

    logger.info(f"\ntop 10 canonical entity references:")
    for entity, count in stats["canonical_entity_refs"][:10]:
        logger.info(f"  {entity}: {count}")

    logger.info(f"\nnew entities created: {len(stats['new_entities'])}")
    if stats['new_entities']:
        for entity, count in stats["new_entities"][:5]:
            logger.info(f"  {entity}: {count}")

    logger.info(f"\ncanonical scene references:")
    for scene_id, count in stats["canonical_scene_refs"]:
        logger.info(f"  scene {scene_id}: {count}")

    logger.info(f"\nnew scenes created: {stats['new_scenes_count']}")

    logger.info(f"\ncanonical theory references:")
    for theory_id, count in stats["canonical_theory_refs"]:
        logger.info(f"  theory {theory_id}: {count}")

    logger.info(f"\nnew theories created: {stats['new_theories_count']}")

    logger.info(f"\ntop sentiments:")
    for sentiment, count in stats["top_sentiments"]:
        logger.info(f"  {sentiment}: {count}")

    logger.info("="*60)
    logger.info("two-pass classification complete")


if __name__ == "__main__":
    main()
