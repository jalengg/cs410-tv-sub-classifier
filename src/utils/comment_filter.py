"""filter comments by complexity before classification."""

import re


class CommentFilter:
    """filters comments to identify which need llm classification.

    Attributes:
        min_length: minimum comment length in characters
        min_words: minimum word count
        entity_keywords: keywords indicating entity mentions
        theory_keywords: keywords indicating theories/predictions
    """

    def __init__(
        self,
        min_length: int = 50,
        min_words: int = 10
    ):
        """initialize filter.

        Args:
            min_length: minimum comment length
            min_words: minimum word count
        """
        self.min_length = min_length
        self.min_words = min_words

        self.entity_keywords = {
            'mark', 'helly', 'irving', 'dylan', 'petey', 'cobel',
            'milchick', 'ricken', 'gemma', 'devon', 'burt', 'harmony',
            'lumon', 'eagan', 'severance', 'innie', 'outie',
            'macrodata', 'refinement', 'kier'
        }

        self.theory_keywords = {
            'think', 'theory', 'believe', 'predict', 'probably',
            'maybe', 'might', 'could be', 'bet', 'guess',
            'wondering', 'suspect', 'speculation', 'feel like'
        }

        self.scene_keywords = {
            'scene', 'when', 'moment', 'part where', 'episode',
            'remember when', 'that time', 'sequence'
        }

    def is_complex(self, comment_body: str) -> bool:
        """check if comment is complex enough for classification.

        Args:
            comment_body: comment text

        Returns:
            true if comment should be classified
        """
        body_lower = comment_body.lower()

        if len(comment_body) < self.min_length:
            return False

        words = comment_body.split()
        if len(words) < self.min_words:
            return False

        has_entity = any(keyword in body_lower for keyword in self.entity_keywords)
        has_theory = any(keyword in body_lower for keyword in self.theory_keywords)
        has_scene = any(keyword in body_lower for keyword in self.scene_keywords)

        return has_entity or has_theory or has_scene

    def is_simple_reaction(self, comment_body: str) -> bool:
        """check if comment is just a simple reaction.

        Args:
            comment_body: comment text

        Returns:
            true if comment is simple reaction (no classification needed)
        """
        body_lower = comment_body.lower()
        words = comment_body.split()

        if len(words) <= 5:
            reaction_patterns = [
                'wow', 'omg', 'lol', 'lmao', 'holy', 'shit', 'damn',
                '!!!', 'amazing', 'incredible', 'love', 'hate'
            ]
            return any(pattern in body_lower for pattern in reaction_patterns)

        return False

    def filter_comments(
        self,
        comments: list[dict],
        max_simple_reactions: int = 100
    ) -> tuple[list[dict], list[dict]]:
        """split comments into complex (needs classification) and simple.

        Args:
            comments: list of comment dicts
            max_simple_reactions: max simple reactions to keep

        Returns:
            tuple of (complex_comments, simple_reactions)
        """
        complex_comments = []
        simple_reactions = []

        for comment in comments:
            if self.is_simple_reaction(comment['body']):
                if len(simple_reactions) < max_simple_reactions:
                    simple_reactions.append(comment)
            elif self.is_complex(comment['body']):
                complex_comments.append(comment)

        return complex_comments, simple_reactions

    def get_complexity_stats(self, comments: list[dict]) -> dict:
        """get statistics on comment complexity.

        Args:
            comments: list of comment dicts

        Returns:
            dict with complexity breakdown
        """
        total = len(comments)
        complex_count = sum(1 for c in comments if self.is_complex(c['body']))
        simple_count = sum(1 for c in comments if self.is_simple_reaction(c['body']))
        other_count = total - complex_count - simple_count

        return {
            'total': total,
            'complex': complex_count,
            'simple_reactions': simple_count,
            'other': other_count,
            'complex_percentage': (complex_count / total * 100) if total > 0 else 0
        }
