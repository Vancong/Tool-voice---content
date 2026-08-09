"""Domain models for the Multi-Agent Review Architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class CharacterProfile:
    name: str
    role: str = ""
    description: str = ""
    first_appearance: str = ""
    personality: str = ""
    motivation: str = ""
    internal_conflict: str = ""
    external_conflict: str = ""
    character_development: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)
    emotional_journey: str = ""


@dataclass
class SceneAnalysisDetail:
    scene_index: int
    start_time: float = 0.0
    end_time: float = 0.0
    what_happens: str = ""
    why_this_scene_matters: str = ""
    why_matters: str = ""
    characters_present: List[str] = field(default_factory=list)
    character_goals: Dict[str, str] = field(default_factory=dict)
    emotional_states: Dict[str, str] = field(default_factory=dict)
    visual_emotions: List[str] = field(default_factory=list)
    body_language: List[str] = field(default_factory=list)
    facial_expressions: List[str] = field(default_factory=list)
    visual_clues: List[str] = field(default_factory=list)
    important_objects: List[str] = field(default_factory=list)
    hidden_clues: List[str] = field(default_factory=list)
    foreshadowing: str = ""
    changes_story: bool = False
    importance_score: int = 5
    scene_type: str = "Development"  # Setup, Development, Conflict, Twist, Climax, Resolution, Ending


@dataclass
class SceneReasoning:
    """Deep Narrative Reasoning and Storytelling Metadata for a single scene."""
    scene_index: int
    actual_event: str = ""
    cause: str = ""
    consequence: str = ""
    who_caused_it: str = ""
    who_was_affected: str = ""
    audience_knows: str = ""
    characters_know: str = ""
    unresolved_questions: List[str] = field(default_factory=list)
    mystery_created: str = ""
    mystery_resolved: str = ""
    future_payoff: str = ""
    dominant_emotion: str = ""
    scene_purpose: str = "Setup"  # Setup, Character Development, Relationship Development, Conflict, Investigation, Revelation, Twist, Climax, Resolution, Ending
    story_importance_score: int = 5  # 1-10
    narration_weight: int = 5  # 1-10: How much narration time this scene deserves
    compression_strategy: str = "SUMMARY"  # FULL, SUMMARY, ONE_SENTENCE, SKIP
    viewer_question: str = ""  # Natural question remaining in viewer's mind
    information_priority: str = "must_tell"  # must_tell, good_to_tell, optional
    future_payoff_scene: Optional[int] = None  # Scene index where setup pays off
    future_payoff_description: str = ""  # Describe how this scene pays off later


@dataclass
class ForeshadowingClue:
    clue: str
    setup_scene: str
    payoff_scene: str = ""
    significance: str = ""


@dataclass
class ViewerCuriosityMoment:
    question: str
    scene_context: str
    hook_potential: str = "high"


@dataclass
class StorylineAnalysis:
    main_storyline: str = ""
    secondary_storylines: List[str] = field(default_factory=list)
    major_conflicts: List[str] = field(default_factory=list)
    main_conflict: str = ""
    secondary_conflicts: List[str] = field(default_factory=list)
    turning_points: List[str] = field(default_factory=list)
    plot_twists: List[str] = field(default_factory=list)
    climax: str = ""
    ending: str = ""
    themes: List[str] = field(default_factory=list)
    moral_or_message: str = ""
    emotional_arc: List[str] = field(default_factory=list)


@dataclass
class MovieMemory:
    """Output of Story Analyst & Narrative Intelligence System (Factual + Reasoning Knowledge Base)."""
    title: str = "Tác phẩm điện ảnh"
    genre: str = "Drama / Action"
    main_theme: str = ""
    plot_summary: str = ""
    characters: List[CharacterProfile] = field(default_factory=list)
    scenes_analysis: List[SceneAnalysisDetail] = field(default_factory=list)
    scene_reasoning: List[SceneReasoning] = field(default_factory=list)
    storyline: StorylineAnalysis = field(default_factory=StorylineAnalysis)
    foreshadowing: List[ForeshadowingClue] = field(default_factory=list)
    curiosity_moments: List[ViewerCuriosityMoment] = field(default_factory=list)
    key_plot_points: List[str] = field(default_factory=list)
    emotional_arc: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MovieMemory:
        chars = []
        for c in data.get("characters", []):
            if isinstance(c, dict):
                c_copy = dict(c)
                if "emotional_development" in c_copy and "emotional_journey" not in c_copy:
                    c_copy["emotional_journey"] = c_copy["emotional_development"]
                chars.append(CharacterProfile(**c_copy))
            else:
                chars.append(CharacterProfile(name=str(c)))

        scenes = []
        for idx, s in enumerate(data.get("scenes_analysis", [])):
            if isinstance(s, dict):
                s_copy = dict(s)
                if "why_this_scene_matters" in s_copy and not s_copy.get("why_matters"):
                    s_copy["why_matters"] = s_copy["why_this_scene_matters"]
                elif "why_matters" in s_copy and not s_copy.get("why_this_scene_matters"):
                    s_copy["why_this_scene_matters"] = s_copy["why_matters"]
                scenes.append(SceneAnalysisDetail(**s_copy))
            else:
                scenes.append(SceneAnalysisDetail(scene_index=idx, what_happens=str(s)))

        reasonings = []
        for idx, r in enumerate(data.get("scene_reasoning", [])):
            if isinstance(r, dict):
                reasonings.append(SceneReasoning(**r))
            else:
                reasonings.append(SceneReasoning(scene_index=idx + 1, actual_event=str(r)))

        storyline_data = data.get("storyline", {})
        storyline = StorylineAnalysis(**storyline_data) if isinstance(storyline_data, dict) else StorylineAnalysis()
        foreshadowing = [
            ForeshadowingClue(**f) if isinstance(f, dict) else ForeshadowingClue(clue=str(f), setup_scene="")
            for f in data.get("foreshadowing", [])
        ]
        curiosity_moments = [
            ViewerCuriosityMoment(**q) if isinstance(q, dict) else ViewerCuriosityMoment(question=str(q), scene_context="")
            for q in data.get("curiosity_moments", [])
        ]
        return cls(
            title=data.get("title", "Tác phẩm điện ảnh"),
            genre=data.get("genre", "Drama / Action"),
            main_theme=data.get("main_theme", ""),
            plot_summary=data.get("plot_summary", ""),
            characters=chars,
            scenes_analysis=scenes,
            scene_reasoning=reasonings,
            storyline=storyline,
            foreshadowing=foreshadowing,
            curiosity_moments=curiosity_moments,
            key_plot_points=data.get("key_plot_points", []),
            emotional_arc=data.get("emotional_arc", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChapterPlan:
    chapter_index: int
    title: str
    objective: str = ""
    narrative_focus: str = ""
    included_scenes: List[int] = field(default_factory=list)
    skipped_scenes: List[int] = field(default_factory=list)
    review_priority: str = "medium"  # high, medium, low
    emotional_tone: str = "Curiosity"  # Curiosity, Mystery, Hope, Fear, Shock, Sadness, Relief
    suspense_level: int = 5  # 1-10
    transition_goal: str = ""
    key_events: List[str] = field(default_factory=list)


@dataclass
class ClimaxStrategy:
    emotional_peak_event: str = ""
    hidden_until_climax: List[str] = field(default_factory=list)
    earlier_scenes_to_reference: List[str] = field(default_factory=list)
    climax_focus: str = ""


@dataclass
class EndingStrategy:
    conclusion: str = ""
    final_emotional_tone: str = "Relief"
    discussion_question: str = ""
    ending_text: str = ""


@dataclass
class ReviewPlan:
    """Output of Review Planner Agent (Storytelling Blueprint)."""
    hook: str
    intro: str = ""
    chapters: List[ChapterPlan] = field(default_factory=list)
    climax_strategy: ClimaxStrategy = field(default_factory=ClimaxStrategy)
    ending_strategy: EndingStrategy = field(default_factory=EndingStrategy)
    climax_focus: str = ""
    ending: str = ""
    target_duration_sec: int = 180
    style: str = "documentary"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewPlan:
        chapters = []
        for i, ch in enumerate(data.get("chapters", [])):
            if isinstance(ch, dict):
                ch_copy = dict(ch)
                if "narrative_focus" not in ch_copy and "objective" in ch_copy:
                    ch_copy["narrative_focus"] = ch_copy["objective"]
                elif "objective" not in ch_copy and "narrative_focus" in ch_copy:
                    ch_copy["objective"] = ch_copy["narrative_focus"]
                chapters.append(ChapterPlan(**ch_copy))
            else:
                chapters.append(ChapterPlan(chapter_index=i + 1, title=str(ch), objective=str(ch)))

        climax_data = data.get("climax_strategy", {})
        if isinstance(climax_data, dict):
            climax_strategy = ClimaxStrategy(**climax_data)
        else:
            climax_strategy = ClimaxStrategy(climax_focus=str(climax_data))

        ending_data = data.get("ending_strategy", {})
        if isinstance(ending_data, dict):
            ending_strategy = EndingStrategy(**ending_data)
        else:
            ending_strategy = EndingStrategy(conclusion=str(ending_data))

        return cls(
            hook=data.get("hook", ""),
            intro=data.get("intro", ""),
            chapters=chapters,
            climax_strategy=climax_strategy,
            ending_strategy=ending_strategy,
            climax_focus=data.get("climax_focus", climax_strategy.climax_focus),
            ending=data.get("ending", ending_strategy.ending_text or ending_strategy.conclusion),
            target_duration_sec=data.get("target_duration_sec", 180),
            style=data.get("style", "documentary"),
        )


@dataclass
class QualityReport:
    """Output of Review Editor Agent assessing the generated review."""
    coherence_score: float = 0.85        # 0.0 to 1.0: Logical continuity and flow
    repetition_score: float = 0.90       # 0.0 to 1.0: 1.0 means zero unwanted repetition
    timeline_accuracy: float = 0.88      # 0.0 to 1.0: Chronological consistency with video
    engagement_estimate: float = 0.90    # 0.0 to 1.0: Hook strength and storytelling tension
    grammar_score: float = 0.92          # 0.0 to 1.0: Vietnamese grammar & readability
    overall_score: float = 0.89
    passed: bool = True
    feedback: List[str] = field(default_factory=list)
    edits_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QualityReport:
        return cls(
            coherence_score=float(data.get("coherence_score", 0.85)),
            repetition_score=float(data.get("repetition_score", 0.90)),
            timeline_accuracy=float(data.get("timeline_accuracy", 0.88)),
            engagement_estimate=float(data.get("engagement_estimate", 0.90)),
            grammar_score=float(data.get("grammar_score", 0.92)),
            overall_score=float(data.get("overall_score", 0.89)),
            passed=bool(data.get("passed", True)),
            feedback=data.get("feedback", []),
            edits_applied=data.get("edits_applied", []),
        )


@dataclass
class ContinuityIssue:
    """A specific issue detected by the Continuity Checker Agent."""
    issue_type: str                  # timeline_inconsistency, repeated_scene, character_naming, impossible_event, missing_plot_point, contradiction
    description: str
    affected_section_index: int = 0
    section_snippet: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContinuityReport:
    """Output of Continuity Checker Agent."""
    passed: bool = True
    issues: List[ContinuityIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() if isinstance(i, ContinuityIssue) else i for i in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContinuityReport:
        issues = [
            ContinuityIssue(**i) if isinstance(i, dict) else i
            for i in data.get("issues", [])
        ]
        return cls(
            passed=bool(data.get("passed", True)),
            issues=issues,
        )
