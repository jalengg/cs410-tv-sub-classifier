"""Export classifications to CSV files for analysis.

Creates flat CSV files:
- comments.csv: One row per comment with metadata
- comment_entities.csv: One row per entity mention
- comment_scenes.csv: One row per scene mention
- comment_theories.csv: One row per theory mention
- comment_sentiments.csv: One row per sentiment
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def load_merged_canonicals(data_dir: Path) -> dict[str, Any]:
    """Load merged canonical references.

    Args:
        data_dir: Path to data directory

    Returns:
        Merged canonicals dictionary
    """
    merged_file = data_dir / "merged_canonicals.json"
    with open(merged_file) as f:
        return json.load(f)


def export_comments_csv(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export comments to CSV.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of comments exported
    """
    output_file = output_dir / "comments.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'comment_id', 'episode', 'author', 'body',
            'score', 'created_utc', 'word_count'
        ])

        total_comments = 0

        for episode in range(2, 10):
            classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

            if not classification_file.exists():
                continue

            with open(classification_file) as cf:
                classifications = json.load(cf)

            for comment_id, comment_data in classifications.items():
                comment = comment_data['comment']

                writer.writerow([
                    comment_id,
                    episode,
                    comment['author'],
                    comment['body'],
                    comment['score'],
                    comment['created_utc'],
                    len(comment['body'].split())
                ])

                total_comments += 1

        logger.info(f"Exported {total_comments} comments to {output_file}")
        return total_comments


def export_comment_entities_csv(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export comment-entity links to CSV.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of entity mentions exported
    """
    output_file = output_dir / "comment_entities.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'comment_id', 'episode', 'entity_canonical_name',
            'extracted_as', 'confidence'
        ])

        total_mentions = 0

        for episode in range(2, 10):
            classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

            if not classification_file.exists():
                continue

            with open(classification_file) as cf:
                classifications = json.load(cf)

            for comment_id, comment_data in classifications.items():
                classification = comment_data.get('classification', {})

                for entity in classification.get('entity_refs', []):
                    writer.writerow([
                        comment_id,
                        episode,
                        entity.get('canonical_name', ''),
                        entity.get('extracted_as', ''),
                        entity.get('confidence', '')
                    ])

                    total_mentions += 1

        logger.info(f"Exported {total_mentions} entity mentions to {output_file}")
        return total_mentions


def export_comment_scenes_csv(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export comment-scene links to CSV.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of scene mentions exported
    """
    output_file = output_dir / "comment_scenes.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'comment_id', 'episode', 'scene_id', 'confidence'
        ])

        total_mentions = 0

        for episode in range(2, 10):
            classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

            if not classification_file.exists():
                continue

            with open(classification_file) as cf:
                classifications = json.load(cf)

            for comment_id, comment_data in classifications.items():
                classification = comment_data.get('classification', {})

                for scene in classification.get('scene_refs', []):
                    writer.writerow([
                        comment_id,
                        episode,
                        scene.get('scene_id', ''),
                        scene.get('confidence', '')
                    ])

                    total_mentions += 1

        logger.info(f"Exported {total_mentions} scene mentions to {output_file}")
        return total_mentions


def export_comment_theories_csv(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export comment-theory links to CSV.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of theory mentions exported
    """
    output_file = output_dir / "comment_theories.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'comment_id', 'episode', 'theory_id',
            'endorsement', 'confidence'
        ])

        total_mentions = 0

        for episode in range(2, 10):
            classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

            if not classification_file.exists():
                continue

            with open(classification_file) as cf:
                classifications = json.load(cf)

            for comment_id, comment_data in classifications.items():
                classification = comment_data.get('classification', {})

                for theory in classification.get('theory_refs', []):
                    writer.writerow([
                        comment_id,
                        episode,
                        theory.get('theory_id', ''),
                        theory.get('endorsement', ''),
                        theory.get('confidence', '')
                    ])

                    total_mentions += 1

        logger.info(f"Exported {total_mentions} theory mentions to {output_file}")
        return total_mentions


def export_comment_sentiments_csv(
    data_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export comment sentiments to CSV.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of sentiments exported
    """
    output_file = output_dir / "comment_sentiments.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'comment_id', 'episode', 'sentiment', 'intensity'
        ])

        total_sentiments = 0

        for episode in range(2, 10):
            classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

            if not classification_file.exists():
                continue

            with open(classification_file) as cf:
                classifications = json.load(cf)

            for comment_id, comment_data in classifications.items():
                classification = comment_data.get('classification', {})

                for sentiment in classification.get('sentiments', []):
                    if isinstance(sentiment, dict):
                        writer.writerow([
                            comment_id,
                            episode,
                            sentiment.get('sentiment', ''),
                            sentiment.get('intensity', '')
                        ])
                    else:
                        # Sentiment is just a string
                        writer.writerow([
                            comment_id,
                            episode,
                            sentiment,
                            ''
                        ])

                    total_sentiments += 1

        logger.info(f"Exported {total_sentiments} sentiments to {output_file}")
        return total_sentiments


def export_entities_csv(
    merged: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export merged entities to CSV.

    Args:
        merged: Merged canonicals dictionary
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of entities exported
    """
    output_file = output_dir / "entities.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'canonical_name', 'type', 'aliases', 'source_episodes'
        ])

        for entity in merged['entities']['merged_entities']:
            writer.writerow([
                entity['canonical_name'],
                entity['type'],
                ', '.join(entity['aliases']),
                ', '.join(map(str, entity['source_episodes']))
            ])

        total = len(merged['entities']['merged_entities'])
        logger.info(f"Exported {total} entities to {output_file}")
        return total


def export_theories_csv(
    merged: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export merged theories to CSV.

    Args:
        merged: Merged canonicals dictionary
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of theories exported
    """
    output_file = output_dir / "theories.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'theory_id', 'canonical_claim', 'category',
            'consensus', 'total_mentions', 'episodes'
        ])

        for theory in merged['theories']['merged_theories']:
            writer.writerow([
                theory['id'],
                theory['canonical_claim'],
                theory['category'],
                theory['consensus'],
                theory['total_mentions'],
                ', '.join(map(str, theory['episodes']))
            ])

        total = len(merged['theories']['merged_theories'])
        logger.info(f"Exported {total} theories to {output_file}")
        return total


def export_scenes_csv(
    merged: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> int:
    """Export merged scenes to CSV.

    Args:
        merged: Merged canonicals dictionary
        output_dir: Path to output directory
        logger: Logger instance

    Returns:
        Total number of scenes exported
    """
    output_file = output_dir / "scenes.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scene_id', 'major_moment_id', 'major_moment_name',
            'canonical_description', 'episodes', 'mention_count'
        ])

        total = 0

        for major_moment in merged['scenes']['major_moments']:
            major_id = major_moment['id']
            major_desc = major_moment['canonical_description']

            for micro_moment in major_moment['micro_moments']:
                writer.writerow([
                    micro_moment['id'],
                    major_id,
                    major_desc,
                    micro_moment['canonical_description'],
                    ', '.join(map(str, micro_moment['episodes'])),
                    micro_moment['mention_count']
                ])
                total += 1

        logger.info(f"Exported {total} scenes to {output_file}")
        return total


def main() -> None:
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting CSV export...")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = data_dir / "csv_exports"
    output_dir.mkdir(exist_ok=True)

    # Load merged canonicals
    logger.info("Loading merged canonical references...")
    merged = load_merged_canonicals(data_dir)

    # Export canonical reference tables
    logger.info("Exporting canonical reference tables...")
    export_entities_csv(merged, output_dir, logger)
    export_theories_csv(merged, output_dir, logger)
    export_scenes_csv(merged, output_dir, logger)

    # Export comment tables
    logger.info("Exporting comment tables...")
    total_comments = export_comments_csv(data_dir, output_dir, logger)
    total_entity_mentions = export_comment_entities_csv(data_dir, output_dir, logger)
    total_scene_mentions = export_comment_scenes_csv(data_dir, output_dir, logger)
    total_theory_mentions = export_comment_theories_csv(data_dir, output_dir, logger)
    total_sentiments = export_comment_sentiments_csv(data_dir, output_dir, logger)

    logger.info("=== CSV EXPORT COMPLETE ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Comments: {total_comments}")
    logger.info(f"Entity mentions: {total_entity_mentions}")
    logger.info(f"Scene mentions: {total_scene_mentions}")
    logger.info(f"Theory mentions: {total_theory_mentions}")
    logger.info(f"Sentiments: {total_sentiments}")


if __name__ == "__main__":
    main()
