"""parser for reddit archive data (ndjson format)."""

import json
import re
from pathlib import Path


class ArchiveParser:
    """parses reddit archive submissions and comments.

    Attributes:
        submissions_path: path to submissions ndjson file
        comments_path: path to comments ndjson file
    """

    def __init__(self, submissions_path: str | Path, comments_path: str | Path):
        """initialize parser.

        Args:
            submissions_path: path to submissions file
            comments_path: path to comments file
        """
        self.submissions_path = Path(submissions_path)
        self.comments_path = Path(comments_path)

    def parse_episode_threads(
        self,
        pattern: str = r"Severance - (\d+)x(\d+).*Episode Discussion"
    ) -> list[dict]:
        """extract episode discussion threads from submissions.

        Args:
            pattern: regex pattern to match episode discussion titles

        Returns:
            list of episode thread metadata dicts
        """
        threads = []

        with open(self.submissions_path, 'r') as f:
            for line in f:
                submission = json.loads(line)
                title = submission.get('title', '')

                match = re.search(pattern, title)
                if match:
                    season = int(match.group(1))
                    episode = int(match.group(2))

                    threads.append({
                        'id': submission['id'],
                        'title': title,
                        'season': season,
                        'episode': episode,
                        'created_utc': submission['created_utc'],
                        'num_comments': submission.get('num_comments', 0),
                        'permalink': submission['permalink'],
                        'author': submission.get('author', '[deleted]')
                    })

        return sorted(threads, key=lambda x: (x['season'], x['episode']))

    def parse_comments_for_thread(
        self,
        thread_id: str,
        max_comments: int | None = None
    ) -> list[dict]:
        """extract comments for specific thread.

        Args:
            thread_id: reddit thread id
            max_comments: optional limit on comments returned

        Returns:
            list of comment dicts
        """
        comments = []
        link_id = f"t3_{thread_id}"

        with open(self.comments_path, 'r') as f:
            for line in f:
                comment = json.loads(line)

                if comment.get('link_id') == link_id:
                    comments.append({
                        'id': comment['id'],
                        'body': comment['body'],
                        'author': comment.get('author', '[deleted]'),
                        'score': comment.get('score', 0),
                        'created_utc': comment['created_utc'],
                        'parent_id': comment['parent_id'],
                        'is_top_level': comment['parent_id'] == link_id,
                        'permalink': comment['permalink']
                    })

                    if max_comments and len(comments) >= max_comments:
                        break

        return sorted(comments, key=lambda x: x['created_utc'])

    def parse_all_episodes(
        self,
        output_dir: str | Path,
        pattern: str = r"Severance - (\d+)x(\d+).*Episode Discussion"
    ) -> dict[str, str]:
        """parse all episode threads and save to separate files.

        Args:
            output_dir: directory to save parsed episode files
            pattern: regex pattern for episode threads

        Returns:
            dict mapping episode keys (s01e01) to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        threads = self.parse_episode_threads(pattern)
        output_files = {}

        for thread in threads:
            episode_key = f"s{thread['season']:02d}e{thread['episode']:02d}"

            print(f"parsing {episode_key}: {thread['title'][:50]}...")
            comments = self.parse_comments_for_thread(thread['id'])

            episode_data = {
                'thread': thread,
                'comments': comments,
                'stats': {
                    'total_comments': len(comments),
                    'top_level_comments': sum(1 for c in comments if c['is_top_level']),
                    'total_score': sum(c['score'] for c in comments)
                }
            }

            output_file = output_dir / f"{episode_key}.json"
            with open(output_file, 'w') as f:
                json.dump(episode_data, f, indent=2)

            output_files[episode_key] = str(output_file)
            print(f"  saved {len(comments)} comments to {output_file}")

        return output_files

    def get_episode_stats(
        self,
        pattern: str = r"Severance - (\d+)x(\d+).*Episode Discussion"
    ) -> list[dict]:
        """get statistics for all episode threads without parsing comments.

        Args:
            pattern: regex pattern for episode threads

        Returns:
            list of episode stats dicts
        """
        threads = self.parse_episode_threads(pattern)

        stats = []
        for thread in threads:
            stats.append({
                'episode': f"s{thread['season']:02d}e{thread['episode']:02d}",
                'title': thread['title'],
                'num_comments': thread['num_comments'],
                'thread_id': thread['id']
            })

        return stats
