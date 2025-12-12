"""Backfill comment classifications with merged canonical references.

Remaps entity_refs, scene_refs, theory_refs in all classification files
to use the consolidated canonical IDs from merged_canonicals.json.
"""

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


def build_entity_mapping(merged: dict[str, Any]) -> dict[str, str]:
    """Build mapping from old entity names to new canonical names.

    Args:
        merged: Merged canonicals dictionary

    Returns:
        Mapping dictionary {old_name: new_canonical_name}
    """
    mapping = {}

    # Add explicit mappings
    mapping.update(merged['entities']['entity_mapping'])

    # Add self-mappings for canonical names
    for entity in merged['entities']['merged_entities']:
        canonical = entity['canonical_name']
        mapping[canonical] = canonical
        for alias in entity['aliases']:
            mapping[alias] = canonical

    return mapping


def build_scene_mapping(merged: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Build mapping from scene descriptions to (major_id, micro_id).

    Args:
        merged: Merged canonicals dictionary

    Returns:
        Mapping dictionary {scene_description: (major_id, micro_id)}
    """
    mapping = {}

    for major_moment in merged['scenes']['major_moments']:
        major_id = major_moment['id']
        for micro_moment in major_moment['micro_moments']:
            micro_id = micro_moment['id']
            desc = micro_moment['canonical_description']
            mapping[desc] = (major_id, micro_id)

    return mapping


def build_theory_mapping(merged: dict[str, Any]) -> dict[str, int]:
    """Build mapping from theory claims to new theory IDs.

    Args:
        merged: Merged canonicals dictionary

    Returns:
        Mapping dictionary {canonical_claim: theory_id}
    """
    mapping = {}

    for theory in merged['theories']['merged_theories']:
        theory_id = theory['id']
        canonical_claim = theory['canonical_claim']
        mapping[canonical_claim] = theory_id

        # Also map evolution claims to same ID
        for evolution in theory.get('evolution', []):
            mapping[evolution['claim']] = theory_id

    return mapping


def remap_classification(
    classification: dict[str, Any],
    entity_map: dict[str, str],
    theory_map: dict[str, int],
    logger: logging.Logger
) -> dict[str, Any]:
    """Remap a single classification to use merged canonical references.

    Args:
        classification: Original classification
        entity_map: Entity name mapping
        theory_map: Theory claim mapping
        logger: Logger instance

    Returns:
        Updated classification
    """
    # Remap entity_refs (list of {canonical_name, extracted_as, confidence})
    if 'entity_refs' in classification:
        remapped_entities = []
        seen_canonical = set()

        for entity_obj in classification['entity_refs']:
            if isinstance(entity_obj, dict):
                old_canonical = entity_obj.get('canonical_name', '')
                new_canonical = entity_map.get(old_canonical, old_canonical)

                if new_canonical not in seen_canonical:
                    entity_obj['canonical_name'] = new_canonical
                    remapped_entities.append(entity_obj)
                    seen_canonical.add(new_canonical)
            else:
                # Handle case where entity_refs might be list of strings
                canonical = entity_map.get(entity_obj, entity_obj)
                if canonical not in seen_canonical:
                    remapped_entities.append({'canonical_name': canonical})
                    seen_canonical.add(canonical)

        classification['entity_refs'] = remapped_entities

    # Theory refs already use IDs from per-episode context - no remapping needed for now
    # Could remap to merged theory IDs in future if needed
    pass

    return classification


def backfill_episode(
    episode: int,
    data_dir: Path,
    entity_map: dict[str, str],
    theory_map: dict[str, int],
    logger: logging.Logger
) -> tuple[int, int]:
    """Backfill classifications for a single episode.

    Args:
        episode: Episode number
        data_dir: Path to data directory
        entity_map: Entity name mapping
        theory_map: Theory claim mapping
        logger: Logger instance

    Returns:
        Tuple of (total_comments, updated_comments)
    """
    classification_file = data_dir / f"classifications_v2/s01e{episode:02d}_classifications.json"

    if not classification_file.exists():
        logger.warning(f"Classification file not found: {classification_file}")
        return (0, 0)

    # Load classifications (dict of comment_id -> {comment, classification})
    with open(classification_file) as f:
        classifications = json.load(f)

    total_comments = len(classifications)
    updated_comments = 0

    # Remap each classification
    for comment_id, comment_data in classifications.items():
        if 'classification' not in comment_data:
            continue

        classification = comment_data['classification']

        original_entity_count = len(classification.get('entity_refs', []))
        original_theory_count = len(classification.get('theory_refs', []))

        remap_classification(classification, entity_map, theory_map, logger)

        new_entity_count = len(classification.get('entity_refs', []))
        new_theory_count = len(classification.get('theory_refs', []))

        if original_entity_count != new_entity_count or original_theory_count != new_theory_count:
            updated_comments += 1

    # Save updated classifications
    with open(classification_file, 'w') as f:
        json.dump(classifications, f, indent=2)

    return (total_comments, updated_comments)


def main() -> None:
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting classification backfill...")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    # Load merged canonicals
    logger.info("Loading merged canonical references...")
    merged = load_merged_canonicals(data_dir)

    # Build mappings
    logger.info("Building entity mapping...")
    entity_map = build_entity_mapping(merged)
    logger.info(f"Entity mapping: {len(entity_map)} entries")

    logger.info("Building theory mapping...")
    theory_map = build_theory_mapping(merged)
    logger.info(f"Theory mapping: {len(theory_map)} entries")

    # Backfill each episode
    logger.info("Backfilling episode classifications...")

    total_comments_all = 0
    total_updated_all = 0

    for episode in range(2, 10):
        logger.info(f"Processing episode {episode}...")
        total_comments, updated_comments = backfill_episode(
            episode, data_dir, entity_map, theory_map, logger
        )

        logger.info(f"Episode {episode}: {updated_comments}/{total_comments} comments updated")

        total_comments_all += total_comments
        total_updated_all += updated_comments

    logger.info("=== BACKFILL COMPLETE ===")
    logger.info(f"Total comments processed: {total_comments_all}")
    logger.info(f"Total comments updated: {total_updated_all}")
    logger.info(f"Update rate: {100*total_updated_all/total_comments_all:.1f}%")


if __name__ == "__main__":
    main()
