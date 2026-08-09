"""Multi-Agent Review Pipeline Orchestrator.

Coordinates five logical agents:
1. Story Analyst -> movie_memory.json
2. Review Planner -> planner.json
3. Review Writer -> draft_review.json
4. Review Editor -> quality_report.json & final_review.json
5. Continuity Checker -> continuity_report.json (with automatic targeted section regeneration)

Features:
- Caches all intermediate artifacts.
- Skips completed AI stages on resume.
- Performs targeted section regeneration if continuity issues are detected without regenerating the whole script.
- Ensures only high quality reviews pass to the TTS stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Callable

from src.agents.models import MovieMemory, ReviewPlan, QualityReport, ContinuityReport
from src.agents.story_analyst import StoryAnalystAgent
from src.agents.review_planner import ReviewPlannerAgent
from src.agents.review_writer import ReviewWriterAgent
from src.agents.review_editor import ReviewEditorAgent
from src.agents.continuity_checker import ContinuityCheckerAgent
from src.review.models import ReviewResult, ReviewMetadata
from src.review.exceptions import ReviewError
from src.utils.logger import get_logger

_logger = get_logger("multi_agent_pipeline")


class MultiAgentReviewPipeline:
    """Orchestrates multi-agent review generation with artifact persistence, continuity checking, and quality gating."""

    def __init__(
        self,
        browser_mgr: Optional[Any] = None,
        openai_provider: Optional[Any] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        quality_threshold: float = 0.70,
    ) -> None:
        self._browser_mgr = browser_mgr
        self._openai_provider = openai_provider
        self._sample_style = sample_style
        self._custom_sample_text = custom_sample_text
        self._quality_threshold = quality_threshold
        self._story_analyst = StoryAnalystAgent()
        self._review_planner = ReviewPlannerAgent()
        self._review_writer = ReviewWriterAgent(
            browser_mgr=browser_mgr,
            openai_provider=openai_provider,
            sample_style=sample_style,
            custom_sample_text=custom_sample_text,
        )
        self._review_editor = ReviewEditorAgent(quality_threshold=quality_threshold)
        self._continuity_checker = ContinuityCheckerAgent()

    @staticmethod
    def _get_artifacts_dir(job_id: str) -> Path:
        p = Path("data") / "jobs" / job_id / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def execute(
        self,
        timeline: Any,
        style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        custom_instructions: Optional[str] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        job_id: Optional[str] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ReviewResult:
        """Run all 5 agents in sequence with caching, targeted regeneration, and quality gating."""
        active_job = job_id or "default_job"
        art_dir = self._get_artifacts_dir(active_job)

        def _notify(stage_name: str, pct: float) -> None:
            _logger.info("[Multi-Agent] {} ({:.0%})", stage_name, pct)
            if progress_callback:
                try:
                    progress_callback(stage_name, pct)
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Stage 1: Story Analyst Agent -> movie_memory.json
        # -------------------------------------------------------------
        _notify("Story Analyst: Building Movie Memory", 0.10)
        memory_file = art_dir / "movie_memory.json"
        if resume and memory_file.exists():
            try:
                data = json.loads(memory_file.read_text(encoding="utf-8"))
                memory = MovieMemory.from_dict(data)
                _logger.info("[Resume] Loaded cached movie_memory.json")
            except Exception:
                memory = self._story_analyst.analyze(timeline, job_id=active_job)
                memory_file.write_text(json.dumps(memory.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            memory = self._story_analyst.analyze(timeline, job_id=active_job)
            memory_file.write_text(json.dumps(memory.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        # -------------------------------------------------------------
        # Stage 2: Review Planner Agent -> planner.json
        # -------------------------------------------------------------
        _notify("Review Planner: Constructing Blueprint", 0.25)
        planner_file = art_dir / "planner.json"
        if resume and planner_file.exists():
            try:
                data = json.loads(planner_file.read_text(encoding="utf-8"))
                plan = ReviewPlan.from_dict(data)
                _logger.info("[Resume] Loaded cached planner.json")
            except Exception:
                plan = self._review_planner.plan(
                    memory=memory,
                    timeline=timeline,
                    style=style,
                    target_duration=target_duration or 180,
                )
                planner_file.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            plan = self._review_planner.plan(
                memory=memory,
                timeline=timeline,
                style=style,
                target_duration=target_duration or 180,
            )
            planner_file.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        # -------------------------------------------------------------
        # Stage 3: Review Writer Agent -> draft_review.json
        # -------------------------------------------------------------
        _notify("Review Writer: Generating Continuous Review", 0.50)
        draft_file = art_dir / "draft_review.json"
        if resume and draft_file.exists():
            try:
                data = json.loads(draft_file.read_text(encoding="utf-8"))
                draft_review = ReviewResult(
                    title=data.get("title", ""),
                    hook=data.get("hook", ""),
                    script=data.get("script", ""),
                    metadata=ReviewMetadata(**data.get("metadata", {})),
                )
                _logger.info("[Resume] Loaded cached draft_review.json")
            except Exception:
                draft_review = self._review_writer.write(
                    memory=memory,
                    plan=plan,
                    timeline=timeline,
                    language=language,
                    custom_instructions=custom_instructions,
                    sample_style=sample_style,
                    custom_sample_text=custom_sample_text,
                    job_id=active_job,
                    progress_callback=progress_callback,
                )
                self._save_review_json(draft_file, draft_review)
        else:
            draft_review = self._review_writer.write(
                memory=memory,
                plan=plan,
                timeline=timeline,
                language=language,
                custom_instructions=custom_instructions,
                job_id=active_job,
                progress_callback=progress_callback,
            )
            self._save_review_json(draft_file, draft_review)

        # -------------------------------------------------------------
        # Stage 4: Review Editor Agent -> quality_report.json & final_review.json
        # -------------------------------------------------------------
        _notify("Review Editor: Validating & Polishing Review", 0.70)
        final_review_file = art_dir / "final_review.json"
        quality_file = art_dir / "quality_report.json"

        final_review, quality_report = self._review_editor.edit_and_evaluate(
            draft=draft_review,
            memory=memory,
            plan=plan,
        )

        quality_file.write_text(json.dumps(quality_report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._save_review_json(final_review_file, final_review)

        # -------------------------------------------------------------
        # Stage 5: Continuity Checker Agent -> continuity_report.json
        # -------------------------------------------------------------
        _notify("Continuity Checker: Verifying Narrative & Timeline Integrity", 0.85)
        continuity_file = art_dir / "continuity_report.json"
        continuity_report = self._continuity_checker.check(
            review=final_review,
            memory=memory,
            plan=plan,
        )

        if not continuity_report.passed and len(continuity_report.issues) > 0:
            _logger.warning(
                "[Continuity Checker] Found {} issues. Asking Review Writer to regenerate affected sections...",
                len(continuity_report.issues),
            )
            _notify("Continuity Checker: Auto-Regenerating Affected Sections", 0.90)
            final_review = self._review_writer.regenerate_sections(
                current_review=final_review,
                issues=continuity_report.issues,
                memory=memory,
                plan=plan,
                job_id=active_job,
            )
            # Re-evaluate with editor to ensure absolute clean formatting
            final_review, quality_report = self._review_editor.edit_and_evaluate(
                draft=final_review,
                memory=memory,
                plan=plan,
            )
            # Re-check continuity
            continuity_report = self._continuity_checker.check(
                review=final_review,
                memory=memory,
                plan=plan,
            )
            self._save_review_json(final_review_file, final_review)
            quality_file.write_text(json.dumps(quality_report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        continuity_file.write_text(json.dumps(continuity_report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        # -------------------------------------------------------------
        # Quality Gate Check
        # -------------------------------------------------------------
        _logger.info(
            "[Quality Gate] Score: {:.2f} (Threshold: {:.2f}) | Passed: {}",
            quality_report.overall_score,
            self._quality_threshold,
            quality_report.passed,
        )

        if not quality_report.passed:
            err_msg = f"Review quality score ({quality_report.overall_score:.2f}) is below threshold ({self._quality_threshold:.2f}). Feedback: {'; '.join(quality_report.feedback)}"
            _logger.error(err_msg)
            raise ReviewError(err_msg)

        _notify("Multi-Agent Review Complete", 1.0)
        return final_review

    @staticmethod
    def _save_review_json(path: Path, result: ReviewResult) -> None:
        data = {
            "title": result.title,
            "hook": result.hook,
            "script": result.script,
            "metadata": {
                "total_words": result.metadata.total_words,
                "estimated_duration": result.metadata.estimated_duration,
                "model_name": result.metadata.model_name,
                "processing_time": result.metadata.processing_time,
            },
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
