"""prompt templates for comment classification."""


CLASSIFICATION_PROMPT = """You are analyzing a Reddit comment from an episode discussion thread for the TV show "Severance".

Your task is to extract structured information from the comment.

**Episode Context:**
Season {season}, Episode {episode}: "{episode_title}"

**Comment to analyze:**
{comment_body}

**Instructions:**
1. Identify all characters, entities, locations, or concepts mentioned
2. Identify any specific scenes referenced (be as specific as possible)
3. Identify any fan theories or predictions the author endorses, mentions, or rejects
4. Identify the emotional tone/sentiment of the comment (can be multiple)
5. Output as JSON with this exact schema:

{{
  "entities": [
    {{
      "name": "character or entity name",
      "type": "character|location|organization|concept|object",
      "confidence": "high|medium|low"
    }}
  ],
  "scenes": [
    {{
      "description": "brief scene description",
      "confidence": "high|medium|low"
    }}
  ],
  "theories": [
    {{
      "description": "theory or prediction description",
      "endorsement": "strong|implied|mentioned|rejected",
      "confidence": "high|medium|low"
    }}
  ],
  "sentiments": ["shocked", "excited", "confused", "sad", "angry", "disappointed", "love", "hate", "scared", "amused", "impressed", "annoyed"]
}}

**Important:**
- Only extract information explicitly present in the comment
- Don't infer beyond what's stated
- Use "high" confidence when entity/scene is clearly mentioned, "medium" when implied, "low" when uncertain
- For theories, capture the core prediction and whether the author endorses it
- Sentiments should reflect the author's emotional tone
- If nothing is found for a category, return an empty array
- Return ONLY valid JSON, no other text

**Output:**"""


def build_classification_prompt(
    comment_body: str,
    season: int,
    episode: int,
    episode_title: str
) -> str:
    """build classification prompt for a comment.

    Args:
        comment_body: comment text to classify
        season: season number
        episode: episode number
        episode_title: episode title

    Returns:
        formatted prompt string
    """
    return CLASSIFICATION_PROMPT.format(
        season=season,
        episode=episode,
        episode_title=episode_title,
        comment_body=comment_body
    )
