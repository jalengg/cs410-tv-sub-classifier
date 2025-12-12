"""Prompt templates for two-pass classification (Pass 2: linking to canonical references)."""


def build_linking_prompt(
    comment_body: str,
    season: int,
    episode: int,
    episode_title: str,
    canonical_entities: list[dict],
    canonical_scenes: list[dict],
    canonical_theories: list[dict]
) -> str:
    """Build prompt for linking comment to canonical references.

    Args:
        comment_body: comment text to classify
        season: season number
        episode: episode number
        episode_title: episode title
        canonical_entities: list of canonical entity dicts from Pass 1
        canonical_scenes: list of canonical scene dicts from Pass 1
        canonical_theories: list of canonical theory dicts from Pass 1

    Returns:
        formatted prompt string
    """
    # format entity list
    entity_lines = []
    for entity in canonical_entities:
        aliases_str = ", ".join(entity.get("aliases", [])[:5])
        entity_lines.append(
            f"  - {entity['canonical_name']} ({entity['type']}) "
            f"[also: {aliases_str}]"
        )
    entities_text = "\n".join(entity_lines) if entity_lines else "  (none)"

    # format scene list
    scene_lines = []
    for scene in canonical_scenes:
        variations_str = " / ".join(scene.get("variations", [])[:3])
        scene_lines.append(
            f"  [id:{scene['id']}] {scene['canonical_description']}"
        )
        if variations_str:
            scene_lines.append(f"    (also described as: {variations_str})")
    scenes_text = "\n".join(scene_lines) if scene_lines else "  (none)"

    # format theory list
    theory_lines = []
    for i, theory in enumerate(canonical_theories, 1):
        theory_lines.append(
            f"  [id:{i}] [{theory['category']}] {theory['canonical_claim']}"
        )
        theory_lines.append(f"    (consensus: {theory['consensus']})")
    theories_text = "\n".join(theory_lines) if theory_lines else "  (none)"

    return f"""You are analyzing a Reddit comment from an episode discussion thread for the TV show "Severance".

**Episode Context:**
Season {season}, Episode {episode}: "{episode_title}"

**Known Entities from this episode:**
{entities_text}

**Known Scenes from this episode:**
{scenes_text}

**Known Theories discussed in this episode:**
{theories_text}

**Comment to analyze:**
{comment_body}

**Task:** Link this comment to canonical references from the lists above.

**Instructions:**
1. For each entity mentioned in the comment:
   - Match to canonical entity from the list above (if applicable)
   - Use the canonical_name, NOT the variation used in comment
   - Include what the comment called it (extracted_as)
   - If entity not in list AND is genuinely new character/concept, you may create new entity

2. For each scene referenced:
   - Match to canonical scene by id (if applicable)
   - OR create new scene if genuinely different moment

3. For each theory mentioned:
   - Match to canonical theory by id (if applicable)
   - Determine if author endorses/rejects/mentions
   - OR create new theory if genuinely different speculation

4. Identify sentiment (can be multiple)

**Output as JSON with this EXACT schema:**

{{
  "entity_refs": [
    {{
      "canonical_name": "Mark Scout",
      "extracted_as": "MS",
      "confidence": "high"
    }}
  ],
  "scene_refs": [
    {{
      "scene_id": 2,
      "confidence": "high"
    }}
  ],
  "theory_refs": [
    {{
      "theory_id": 1,
      "endorsement": "strong|implied|mentioned|rejected",
      "confidence": "high"
    }}
  ],
  "new_entities": [
    {{
      "name": "Ricken",
      "type": "character",
      "confidence": "medium"
    }}
  ],
  "new_scenes": [
    {{
      "description": "brief scene description (≤15 words)",
      "confidence": "medium"
    }}
  ],
  "new_theories": [
    {{
      "category": "character_identity|plot_prediction|symbolism|workplace_nature|conspiracy|other",
      "description": "theory description (≤25 words)",
      "endorsement": "strong|implied|mentioned|rejected",
      "confidence": "medium"
    }}
  ],
  "sentiments": ["shocked", "excited", "confused", "sad", "angry", "disappointed", "love", "hate", "scared", "amused", "impressed", "annoyed"]
}}

**Important:**
- Prefer matching to canonical lists over creating new entries
- Only create "new_*" entries when genuinely not in canonical lists
- For entities, check all aliases before creating new
- For scenes, check if comment describes existing scene differently
- For theories, check if comment expresses existing theory in different words
- Sentiment should reflect author's emotional tone
- Return ONLY valid JSON, no other text

**Output:**"""


def build_classification_prompt_fallback(
    comment_body: str,
    season: int,
    episode: int,
    episode_title: str
) -> str:
    """Build fallback prompt for episodes without canonical context.

    This is the original single-pass extraction prompt.

    Args:
        comment_body: comment text to classify
        season: season number
        episode: episode number
        episode_title: episode title

    Returns:
        formatted prompt string
    """
    return f"""You are analyzing a Reddit comment from an episode discussion thread for the TV show "Severance".

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
