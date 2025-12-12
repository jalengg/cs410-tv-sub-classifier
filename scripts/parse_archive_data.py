"""script to parse reddit archive data and extract episodes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.archive_parser import ArchiveParser


def main():
    """parse reddit archive and save episode data."""
    submissions_path = "data/raw_reddit/submissions"
    comments_path = "data/raw_reddit/comments"
    output_dir = "data/parsed"

    print("initializing parser...")
    parser = ArchiveParser(submissions_path, comments_path)

    print("\nfinding episode discussion threads...")
    stats = parser.get_episode_stats()

    print(f"\nfound {len(stats)} episode threads:")
    for stat in stats:
        print(f"  {stat['episode']}: {stat['num_comments']:5} comments - {stat['title'][:60]}")

    print(f"\nparsing all episodes and saving to {output_dir}/...")
    output_files = parser.parse_all_episodes(output_dir)

    print(f"\n✓ parsed {len(output_files)} episodes")
    print(f"✓ output saved to {output_dir}/")


if __name__ == "__main__":
    main()
