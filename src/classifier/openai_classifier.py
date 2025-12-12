"""openai api classifier for reddit comments using gpt-4o-mini."""

import json
import logging
import os
import time

from openai import OpenAI

from src.classifier.prompt_templates import build_classification_prompt

logger = logging.getLogger(__name__)


class OpenAIClassifier:
    """classifies reddit comments using openai gpt-4o-mini.

    Attributes:
        client: openai client instance
        model: gpt model to use
        max_retries: max retry attempts on failure
        retry_delay: delay between retries in seconds
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """initialize classifier.

        Args:
            api_key: openai api key (or from env)
            model: gpt model name
            max_retries: max retry attempts
            retry_delay: delay between retries
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def classify_comment(
        self,
        comment_body: str,
        season: int,
        episode: int,
        episode_title: str
    ) -> dict:
        """classify a single comment.

        Args:
            comment_body: comment text
            season: season number
            episode: episode number
            episode_title: episode title

        Returns:
            classification dict with entities, scenes, theories, sentiments

        Raises:
            Exception: if classification fails after retries
        """
        prompt = build_classification_prompt(
            comment_body=comment_body,
            season=season,
            episode=episode,
            episode_title=episode_title
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                classification = json.loads(result_text)

                # calculate cost (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
                input_cost = response.usage.prompt_tokens * 0.15 / 1_000_000
                output_cost = response.usage.completion_tokens * 0.60 / 1_000_000
                total_cost = input_cost + output_cost

                return {
                    "entities": classification.get("entities", []),
                    "scenes": classification.get("scenes", []),
                    "theories": classification.get("theories", []),
                    "sentiments": classification.get("sentiments", []),
                    "metadata": {
                        "model": self.model,
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cost": total_cost
                    }
                }

            except Exception as e:
                logger.warning(f"classification attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"classification failed after {self.max_retries} attempts")
                    raise

    def classify_comments_batch(
        self,
        comments: list[dict],
        season: int,
        episode: int,
        episode_title: str,
        max_comments: int | None = None
    ) -> list[dict]:
        """classify multiple comments.

        Args:
            comments: list of comment dicts
            season: season number
            episode: episode number
            episode_title: episode title
            max_comments: optional limit on comments to classify

        Returns:
            list of classification results with comment metadata
        """
        results = []
        comments_to_process = comments[:max_comments] if max_comments else comments

        for i, comment in enumerate(comments_to_process):
            logger.info(f"classifying comment {i + 1}/{len(comments_to_process)}")

            try:
                classification = self.classify_comment(
                    comment_body=comment['body'],
                    season=season,
                    episode=episode,
                    episode_title=episode_title
                )

                results.append({
                    "comment_id": comment['id'],
                    "comment_author": comment['author'],
                    "comment_score": comment['score'],
                    "comment_body": comment['body'],
                    "classification": classification
                })

            except Exception as e:
                logger.error(f"failed to classify comment {comment['id']}: {e}")
                results.append({
                    "comment_id": comment['id'],
                    "comment_author": comment['author'],
                    "comment_score": comment['score'],
                    "comment_body": comment['body'],
                    "classification": None,
                    "error": str(e)
                })

        return results
