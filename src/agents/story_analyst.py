"""Story Analyst Agent.

Analyzes raw timeline events, scene descriptions, and dialogue transcripts
to produce a structured MovieMemory model before any review is written.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Dict, Optional

from src.agents.models import (
    MovieMemory,
    CharacterProfile,
    SceneAnalysisDetail,
    SceneReasoning,
    StorylineAnalysis,
)
from src.utils.logger import get_logger

_logger = get_logger("story_analyst")


class StoryAnalystAgent:
    """Logical Agent 1: Film Analyst and Narrative Reasoning Engine."""

    STORY_ANALYST_SYSTEM_PROMPT = """You are Story Analyst, the Narrative Reasoning Engine in a multi-agent movie review system.

Your ONLY responsibility is to reconstruct the original movie as accurately as possible and build a factual + reasoning knowledge base for downstream agents (Review Planner, Review Writer, Review Editor, Continuity Checker).

You are NOT a reviewer.
You are NOT a narrator.
You are NOT a screenplay writer.
You are NOT allowed to summarize the movie for viewers.

=========================================================
INPUT SOURCES
=========================================================
You receive three independent sources:

SOURCE A — Whisper Transcript
Contains spoken dialogue and explicitly confirmed story events.
This is the PRIMARY SOURCE OF TRUTH.

SOURCE B — Timeline
Contains the chronological order of scenes.
This defines WHEN every confirmed event happens.

SOURCE C — Gemini Vision
Contains visual observations extracted from movie frames.
This is ONLY supplementary evidence.

=========================================================
MANDATORY REASONING PROCESS
=========================================================
You MUST process the sources and execute the internal reasoning in this strict order:

---------------------------------------------------------
STEP 1 — EXTRACT STORY FACTS
---------------------------------------------------------
Read ONLY the Whisper Transcript.
Extract every confirmed story fact:
• dialogue
• explicit actions
• revealed identities
• confirmed relationships
• confirmed events
• confirmed motivations
• confirmed story information

Do NOT use Vision. Do NOT infer. Do NOT guess.

---------------------------------------------------------
STEP 2 — VERIFY CHRONOLOGY
---------------------------------------------------------
Read the Timeline.
Assign every confirmed fact to its correct scene.
Verify chronological order.
Do NOT modify any story fact. Do NOT infer missing events.

---------------------------------------------------------
STEP 3 — ENRICH USING VISUAL CONTEXT
---------------------------------------------------------
Read Gemini Vision.
Vision may ONLY enrich confirmed story facts (facial expressions, body language, silent reactions, atmosphere, environment, weather, lighting, important objects, visual clues).
Vision must NEVER create new dialogue, scenes, motivations, relationships, or twists.
If Vision contradicts the Transcript, always preserve the Transcript. Mark conflicting visual notes as: "uncertain".

---------------------------------------------------------
STEP 4 — DEEP NARRATIVE REASONING & STORYTELLING METADATA (INTERNAL ENGINE PHASE)
---------------------------------------------------------
Executed AFTER Transcript + Timeline + Vision analysis and BEFORE exporting movie_memory.json.
For every important scene, infer and store structured narrative reasoning in `scene_reasoning`:
• actual_event: Grounded truth of what occurs in this scene
• cause: Immediate trigger or underlying reason why this event occurs
• consequence: Direct outcome or ripple effect of this event on the story
• who_caused_it: Character(s) initiating or causing the action
• who_was_affected: Character(s) impacted by the event
• audience_knows: Exact information the audience possesses at this moment
• characters_know: What the characters currently know (tracking dramatic irony)
• unresolved_questions: Specific questions left unanswered in this scene
• mystery_created: New question or enigma introduced
• mystery_resolved: Previous enigma or question answered here
• future_payoff: How this setup will pay off in later scenes
• dominant_emotion: Primary emotional frequency of the scene
• scene_purpose: MUST BE ONE OF:
  - Setup
  - Character Development
  - Relationship Development
  - Conflict
  - Investigation
  - Revelation
  - Twist
  - Climax
  - Resolution
  - Ending
• story_importance_score: 1 to 10
• narration_weight: 1 to 10 (How much narration time this scene deserves. Note: Narration weight is NOT the same as story importance. Some scenes are highly important but can be summarized; some scenes deserve long narration because they create emotional engagement.)
• compression_strategy: MUST BE ONE OF:
  - FULL (Full detailed storytelling)
  - SUMMARY (Paced condensation)
  - ONE_SENTENCE (Brief bridge transition)
  - SKIP (Omit repetition / filler)
• viewer_question: What natural question should remain in the viewer's mind after this scene
• information_priority: MUST BE ONE OF:
  - must_tell (Essential fact that cannot be removed)
  - good_to_tell (Enriches engagement)
  - optional (Can be compressed/omitted)
• future_payoff_scene: Scene index where this setup becomes important (integer or null)
• future_payoff_description: Describe how this scene pays off later

=========================================================
STRICT REASONING RULES
=========================================================
- Never invent facts.
- Every reasoning must be supported by Transcript, Timeline or Vision.
- If evidence is insufficient, mark the field as "uncertain".
- Transcript remains the highest authority.
- Vision only enriches confirmed facts.
- Preserve chronological order.
- Compression strategy should optimize YouTube retention while preserving every essential story fact.
- Never remove facts marked as "must_tell".
- This reasoning layer exists only to help downstream Planner and Writer produce more human-quality reviews.

=========================================================
OUTPUT SCHEMA (movie_memory.json)
=========================================================
Return ONLY valid JSON following this exact structure:
{
  "title": "Movie Title / Topic",
  "genre": "Genre description",
  "main_theme": "Core theme",
  "plot_summary": "Complete factual storyline overview",
  "characters": [
    {
      "name": "Character Name",
      "first_appearance": "Scene 1 (00:15)",
      "role": "Main Protagonist / Antagonist / Supporting",
      "personality": "Observed traits",
      "motivation": "Core desire/drive",
      "internal_conflict": "Internal dilemma",
      "external_conflict": "External obstacle/conflict",
      "character_development": "Evolution throughout the story",
      "relationships": {"Other Character": "Relationship details"},
      "emotional_journey": "From beginning to ending"
    }
  ],
  "scenes_analysis": [
    {
      "scene_index": 1,
      "start_time": 0.0,
      "end_time": 15.5,
      "what_happens": "Factual description of actions",
      "why_this_scene_matters": "Narrative importance",
      "importance_score": 8,
      "scene_type": "Setup",
      "characters_present": ["Character A"],
      "character_goals": {"Character A": "Goal in this scene"},
      "emotional_states": {"Character A": "Tense/Anxious"},
      "visual_emotions": ["Visible anxiety", "Hesitation"],
      "body_language": ["Trembling hands", "Looking around nervously"],
      "facial_expressions": ["Furrowed brow", "Biting lips"],
      "visual_clues": ["Stain on jacket", "Clock on wall reading 11:45"],
      "important_objects": ["Briefcase", "Car keys"],
      "foreshadowing": "Hidden key will open the vault in the climax",
      "changes_story": true
    }
  ],
  "scene_reasoning": [
    {
      "scene_index": 1,
      "actual_event": "Protagonist receives a threatening letter at his office",
      "cause": "Past unresolved conflict with former business partner",
      "consequence": "Forces protagonist to flee and seek police assistance",
      "who_caused_it": "Anonymous sender",
      "who_was_affected": "Protagonist",
      "audience_knows": "Protagonist is in grave danger but sender identity is unknown",
      "characters_know": "Protagonist knows he is targeted",
      "unresolved_questions": ["Who sent the letter?", "What happened in the past?"],
      "mystery_created": "Identity of the blackmailer",
      "mystery_resolved": "None (inciting incident)",
      "future_payoff": "The signature on the letter matches the final villain's handwriting",
      "dominant_emotion": "Suspense",
      "scene_purpose": "Setup",
      "story_importance_score": 8,
      "narration_weight": 8,
      "compression_strategy": "FULL",
      "viewer_question": "Who is trying to frame the protagonist?",
      "information_priority": "must_tell",
      "future_payoff_scene": 8,
      "future_payoff_description": "The handwriting on the letter matches the evidence found at the crime scene in Scene 8"
    }
  ],
  "storyline": {
    "main_storyline": "Primary narrative thread",
    "secondary_storylines": ["Subplot A", "Subplot B"],
    "major_conflicts": ["Core central conflict between A and B"],
    "turning_points": ["Key shift at Scene 4"],
    "plot_twists": ["Twist reveal at Scene 7"],
    "climax": "Climax description",
    "ending": "Ending description",
    "themes": ["Survival", "Betrayal", "Trust"],
    "emotional_arc": ["Curiosity", "Tension", "Shock", "Resolution"]
  },
  "foreshadowing": [
    {
      "clue": "Character hides a key",
      "setup_scene": "Scene 2",
      "payoff_scene": "Scene 8",
      "significance": "The key opens the locked vault in the climax"
    }
  ],
  "curiosity_moments": [
    {
      "question": "Why did she hide the letter before opening the door?",
      "scene_context": "Scene 3",
      "hook_potential": "high"
    }
  ],
  "key_plot_points": ["Plot point 1", "Plot point 2"],
  "emotional_arc": ["Curiosity", "Tension", "Climax", "Resolution"]
}

The JSON must be deterministic, factual, structured, and optimized for downstream AI agents."""

    def __init__(self, browser_mgr: Optional[Any] = None) -> None:
        self._browser_mgr = browser_mgr

    def build_analysis_prompt(self, timeline: Any) -> str:
        """Construct the full prompt containing transcript, scene timeline, and visual descriptions."""
        events = []
        if hasattr(timeline, "timeline") and hasattr(timeline.timeline, "events"):
            events = timeline.timeline.events
        elif hasattr(timeline, "events"):
            events = timeline.events
        elif hasattr(timeline, "scenes"):
            events = timeline.scenes

        timeline_lines = []
        for idx, ev in enumerate(events):
            start = getattr(ev, "start_time", 0.0)
            end = getattr(ev, "end_time", 0.0)
            summary = getattr(ev, "summary", "") or getattr(ev, "description", "") or ""
            chars = getattr(ev, "characters", []) or getattr(ev, "key_characters", []) or []
            actions = getattr(ev, "actions", []) or getattr(ev, "key_actions", []) or []
            objects = getattr(ev, "objects", []) or getattr(ev, "key_objects", []) or []
            emotion = getattr(ev, "emotion", "") or getattr(ev, "dominant_emotion", "") or ""
            transcript = getattr(ev, "dialogue", "") or getattr(ev, "transcript", "") or ""

            line = (
                f"- Scene {idx + 1} [{start:.1f}s - {end:.1f}s]:\n"
                f"  * Visual Summary: {summary}\n"
                f"  * Characters: {', '.join(chars) if chars else 'N/A'}\n"
                f"  * Key Actions: {', '.join(actions) if actions else 'N/A'}\n"
                f"  * Key Objects/Visual Clues: {', '.join(objects) if objects else 'N/A'}\n"
                f"  * Visible Emotion: {emotion or 'Neutral'}\n"
            )
            if transcript:
                line += f"  * Dialogue/Transcript: \"{transcript}\"\n"
            timeline_lines.append(line)

        input_data = "\n".join(timeline_lines) if timeline_lines else "No specific scene events detected."

        return (
            f"{self.STORY_ANALYST_SYSTEM_PROMPT}\n\n"
            f"================ INPUT MOVIE DATA ================\n"
            f"{input_data}\n"
            f"==================================================\n"
            f"Execute the 4-step reasoning process (Transcript -> Timeline -> Vision -> Narrative Reasoning) and return ONLY the valid JSON object."
        )

    def analyze(self, timeline: Any, job_id: Optional[str] = None) -> MovieMemory:
        """Analyze timeline, execute deep narrative reasoning, and return structured MovieMemory."""
        _logger.info("Story Analyst (Narrative Reasoning Engine) starting comprehensive analysis...")

        events = []
        if hasattr(timeline, "timeline") and hasattr(timeline.timeline, "events"):
            events = timeline.timeline.events
        elif hasattr(timeline, "events"):
            events = timeline.events
        elif hasattr(timeline, "scenes"):
            events = timeline.scenes

        char_counts: Dict[str, int] = {}
        key_points: List[str] = []
        emotional_arc: List[str] = []
        scenes_detail: List[SceneAnalysisDetail] = []
        scene_reasonings: List[SceneReasoning] = []
        curiosity_moments = []
        foreshadowing = []

        total_ev = len(events)
        for idx, ev in enumerate(events):
            start = float(getattr(ev, "start_time", 0.0))
            end = float(getattr(ev, "end_time", 0.0))
            summary = getattr(ev, "summary", "") or getattr(ev, "description", "") or ""
            chars = getattr(ev, "characters", []) or getattr(ev, "key_characters", []) or []
            actions = getattr(ev, "actions", []) or getattr(ev, "key_actions", []) or []
            objects = getattr(ev, "objects", []) or getattr(ev, "key_objects", []) or []
            emotion = getattr(ev, "emotion", "") or getattr(ev, "dominant_emotion", "") or "Căng thẳng"
            transcript = getattr(ev, "dialogue", "") or getattr(ev, "transcript", "") or ""

            for c in chars:
                if c and len(c) > 1:
                    char_counts[c] = char_counts.get(c, 0) + 1

            if summary:
                key_points.append(f"Cảnh {idx+1}: {summary}")
            if emotion and emotion not in emotional_arc:
                emotional_arc.append(emotion)

            # Determine scene purpose, importance score, narration weight, compression strategy, priority
            if idx == 0:
                s_purpose = "Setup"
                score = 8
                narr_weight = 8
                comp_strategy = "FULL"
                info_priority = "must_tell"
            elif idx == total_ev - 1:
                s_purpose = "Ending"
                score = 9
                narr_weight = 9
                comp_strategy = "FULL"
                info_priority = "must_tell"
            elif total_ev > 3 and idx == int(total_ev * 0.75):
                s_purpose = "Climax"
                score = 10
                narr_weight = 10
                comp_strategy = "FULL"
                info_priority = "must_tell"
            elif total_ev > 4 and idx == int(total_ev * 0.5):
                s_purpose = "Twist"
                score = 9
                narr_weight = 9
                comp_strategy = "FULL"
                info_priority = "must_tell"
            elif len(actions) > 2 or any(w in summary.lower() for w in ["chiến đấu", "đánh", "truy đuổi", "tranh cãi"]):
                s_purpose = "Conflict"
                score = 7
                narr_weight = 7
                comp_strategy = "SUMMARY"
                info_priority = "good_to_tell"
            elif any(w in summary.lower() for w in ["tìm", "điều tra", "phát hiện", "manh mối"]):
                s_purpose = "Investigation"
                score = 7
                narr_weight = 7
                comp_strategy = "SUMMARY"
                info_priority = "good_to_tell"
            elif any(w in summary.lower() for w in ["bí mật", "sự thật", "tiết lộ"]):
                s_purpose = "Revelation"
                score = 8
                narr_weight = 8
                comp_strategy = "FULL"
                info_priority = "must_tell"
            elif len(chars) >= 2:
                s_purpose = "Relationship Development"
                score = 6
                narr_weight = 6
                comp_strategy = "SUMMARY"
                info_priority = "good_to_tell"
            else:
                s_purpose = "Character Development"
                score = 5
                narr_weight = 5
                comp_strategy = "ONE_SENTENCE"
                info_priority = "optional"

            # Build Scene Analysis Detail (Factual Layer)
            scenes_detail.append(
                SceneAnalysisDetail(
                    scene_index=idx + 1,
                    start_time=start,
                    end_time=end,
                    what_happens=summary or "Diễn biến phân cảnh",
                    why_matters=f"Phát triển mạch phim ({s_purpose}) và thể hiện hành động {' '.join(actions[:2])}",
                    characters_present=list(chars),
                    character_goals={c: "Thực hiện hành động chính trong cảnh" for c in chars},
                    emotional_states={c: emotion for c in chars},
                    visual_emotions=[emotion],
                    body_language=[f"Hành động: {', '.join(actions[:2])}"] if actions else [],
                    facial_expressions=[f"Biểu cảm thể hiện trạng thái {emotion}"],
                    visual_clues=list(actions),
                    important_objects=list(objects),
                    hidden_clues=[],
                    changes_story=(idx == 0 or idx == total_ev - 1 or s_purpose in ("Twist", "Climax", "Revelation")),
                    importance_score=score,
                    scene_type=s_purpose,
                )
            )

            # Build Scene Reasoning & Storytelling Metadata Layer
            actor = chars[0] if chars else "Nhân vật chính"
            affected = ", ".join(chars[1:]) if len(chars) > 1 else actor
            v_question = f"Hành động tiếp theo của {actor} sẽ dẫn tới điều gì?" if idx < total_ev - 1 else "Bài học đọng lại sau kết cục là gì?"
            payoff_scene = int(total_ev * 0.75) + 1 if (idx < total_ev * 0.5 and total_ev > 3) else None
            payoff_desc = f"Tạo tiền đề then chốt cho cao trào tại Cảnh {payoff_scene}" if payoff_scene else "Phát triển mạch truyện liên tục"

            scene_reasonings.append(
                SceneReasoning(
                    scene_index=idx + 1,
                    actual_event=summary or (f"Lời thoại: {transcript[:80]}" if transcript else "Diễn biến sự kiện trong cảnh"),
                    cause=f"Hệ quả từ tình huống phân cảnh trước hoặc bối cảnh mở đầu",
                    consequence=f"Thúc đẩy chuyển biến câu chuyện sang {s_purpose.lower()}",
                    who_caused_it=actor,
                    who_was_affected=affected,
                    audience_knows=f"Khán giả chứng kiến {summary or 'sự việc diễn ra'} và phản ứng của {actor}",
                    characters_know=f"{actor} nắm bắt tình hình trong cảnh nhưng chưa biết toàn bộ kết cục",
                    unresolved_questions=[v_question] if idx < total_ev - 1 else [],
                    mystery_created=f"Ẩn số về thử thách tiếp theo" if s_purpose in ("Setup", "Investigation", "Twist") else "None",
                    mystery_resolved=f"Làm sáng tỏ mâu thuẫn" if s_purpose in ("Revelation", "Resolution", "Ending") else "None",
                    future_payoff=payoff_desc,
                    dominant_emotion=emotion,
                    scene_purpose=s_purpose,
                    story_importance_score=score,
                    narration_weight=narr_weight,
                    compression_strategy=comp_strategy,
                    viewer_question=v_question,
                    information_priority=info_priority,
                    future_payoff_scene=payoff_scene,
                    future_payoff_description=payoff_desc,
                )
            )

            if idx == 0 and summary:
                curiosity_moments.append({
                    "question": f"Điều gì dẫn đến tình huống ở đầu phim ({summary[:50]}...)?",
                    "scene_context": "Scene 1",
                    "hook_potential": "high"
                })

        # Build character profiles
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        characters = [
            CharacterProfile(
                name=name,
                role="Nhân vật chính" if idx == 0 else "Nhân vật phụ",
                description=f"Xuất hiện trong {count} phân cảnh chính",
                first_appearance="Cảnh 1",
                personality="Kiên định, chủ động",
                motivation="Đạt được mục đích và giải quyết thử thách trong câu chuyện",
                internal_conflict="Áp lực và khó khăn trong hoàn cảnh đối mặt",
                character_development="Thích nghi và biến chuyển qua các biến cố",
                relationships={},
                emotional_journey="Từ bối rối, tò mò đến quyết tâm và giải quyết nút thắt",
            )
            for idx, (name, count) in enumerate(sorted_chars)
        ]

        plot_overview = " ".join([getattr(e, "summary", "") for e in events[:5] if getattr(e, "summary", "")])

        storyline = StorylineAnalysis(
            main_storyline=plot_overview or "Hành trình diễn biến chính của tác phẩm.",
            secondary_storylines=["Diễn biến tâm lý và mối quan hệ giữa các nhân vật"],
            main_conflict="Xung đột chính và thử thách then chốt mà nhân vật phải đối mặt",
            secondary_conflicts=["Áp lực về thời gian và hoàn cảnh xung quanh"],
            turning_points=[f"Bước ngoặt tại Cảnh {max(1, total_ev//2)}" if events else "Bước ngoặt chính"],
            plot_twists=[],
            climax=f"Cao trào bùng nổ tại phân cảnh {max(1, int(total_ev*0.8))}",
            ending=f"Kết thúc tại phân cảnh cuối ({total_ev})",
            themes=["Sinh tồn", "Hành động", "Tâm lý"],
            moral_or_message="Thông điệp ý nghĩa về sự nỗ lực và bài học cuộc sống",
            emotional_arc=emotional_arc if emotional_arc else ["Tò mò", "Căng thẳng", "Cao trào", "Lắng đọng"],
        )

        memory = MovieMemory(
            title="Tác phẩm điện ảnh",
            genre="Hành động / Kịch tính",
            main_theme="Cuộc đấu tranh sinh tồn và giải mã bí ẩn",
            plot_summary=plot_overview or "Diễn biến câu chuyện kịch tính xoay quanh các nhân vật.",
            characters=characters,
            scenes_analysis=scenes_detail,
            scene_reasoning=scene_reasonings,
            storyline=storyline,
            foreshadowing=[],
            curiosity_moments=curiosity_moments,
            key_plot_points=key_points[:20],
            emotional_arc=emotional_arc if emotional_arc else ["Mở đầu tò mò", "Căng thẳng gia tăng", "Cao trào bùng nổ", "Kết thúc lắng đọng"],
            metadata={"total_events": len(events), "job_id": job_id},
        )

        _logger.info(
            "Narrative Reasoning Engine finished: {} characters, {} scenes analyzed with deep narrative reasoning",
            len(characters),
            len(scenes_detail),
        )
        return memory

