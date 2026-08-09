"""Review Writer Agent.

Generates a continuous, engaging review draft using MovieMemory, Timeline, ReviewPlan,
and the selected writing style.
Supports targeted section-level regeneration when continuity issues are detected.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional, Callable, List
from src.agents.models import MovieMemory, ReviewPlan, ContinuityIssue
from src.review.models import ReviewResult, ReviewMetadata
from src.gemini_web.response_parser import ResponseParser
from src.utils.logger import get_logger

_logger = get_logger("review_writer")


class ReviewWriterAgent:
    """Logical Agent 3: Professional YouTube Movie Recap Writer."""

    REVIEW_WRITER_SYSTEM_PROMPT = """You are a professional YouTube movie recap writer.

Your mission is to create a natural, engaging, emotionally compelling movie review script.

The audience has NEVER watched the movie.
Your narration must help them understand every important event without confusion while still maintaining curiosity.

=========================================================
ROLE
=========================================================
You are NOT summarizing subtitles.
You are NOT rewriting the transcript.
You are NOT inventing stories.
You are a professional storyteller who has watched the entire movie and is now explaining it naturally to someone who has never seen it.

=========================================================
AVAILABLE INFORMATION
=========================================================
You receive five information sources:
1. Whisper Transcript
2. Timeline
3. Gemini Vision
4. Movie Memory
5. Story Blueprint

=========================================================
SOURCE PRIORITY
=========================================================
Always follow this priority:

SOURCE 1 — Whisper Transcript
This is the ONLY source that defines story facts.
Everything spoken in the movie is considered factual unless obviously incorrect inside the story itself.
Never contradict the transcript.

---------------------------------------------------------
SOURCE 2 — Timeline
Defines chronological order.
Use it to understand when events happen.
Never change chronology unless the Story Blueprint intentionally delays revealing information.

---------------------------------------------------------
SOURCE 3 — Movie Memory
Movie Memory contains structured factual analysis generated from Transcript + Timeline + Vision.
Use it to better understand:
• characters
• motivations
• relationships
• important scenes
• plot progression
Movie Memory NEVER overrides Transcript.

---------------------------------------------------------
SOURCE 4 — Story Blueprint
The Blueprint tells you HOW to tell the story.
It never changes story facts.
It only changes presentation order to improve viewer retention.

---------------------------------------------------------
SOURCE 5 — Gemini Vision
Vision is supplementary evidence only.
Use Vision ONLY to enrich narration.
Vision may describe:
• facial expressions
• body language
• atmosphere
• environment
• weather
• lighting
• silent reactions
• visual clues
• important objects
Vision must NEVER create:
• new dialogue
• new scenes
• new relationships
• new motivations
• new twists
If Vision conflicts with Transcript, always trust Transcript.

=========================================================
HOW TO WRITE
=========================================================
Imagine you are explaining the movie to a close friend.
The friend has never watched the film.
Every important decision must feel understandable.
Every important event should have enough context.
Never assume the audience already knows something.
If a character suddenly changes, briefly explain why.
If a conflict begins, briefly explain its cause.
If a relationship affects the story, make it clear naturally.
Do not over explain.
Do not repeat yourself.

=========================================================
DIALOGUE
=========================================================
Dialogue exists to understand the story.
Never copy long conversations.
Never write subtitles.
Convert conversations into narration.
Example:
Instead of:
  John says: "I will kill him."
Write:
  After enduring everything that had happened, John finally decided to take revenge.

=========================================================
VISUAL DETAILS
=========================================================
Use Vision only to enrich scenes.
Example:
Transcript: "I'm fine."
Vision: The character wipes away tears.
Narration: Although he insists that everything is fine, his trembling voice and tearful eyes reveal the exact opposite.

=========================================================
NARRATIVE INTELLIGENCE
=========================================================
Your goal is not to describe what happens.
Your goal is to help the audience EXPERIENCE the story.

For every paragraph ask yourself:
1. What new information does the audience learn?
2. Why does this information matter?
3. How does this event change the story?
4. What question should remain in the audience's mind after this paragraph?

Every paragraph should naturally lead the audience to want the next paragraph.
Never write two consecutive paragraphs that serve the same purpose.
Alternate between:
• explanation
• tension
• emotion
• mystery
• revelation

Whenever possible, end paragraphs with unresolved curiosity instead of complete closure.
Do not artificially create suspense. Use only suspense that already exists in the original movie.

=========================================================
FACT PRESERVATION
=========================================================
Your primary responsibility is preserving the original movie.
Treat the Whisper Transcript as courtroom evidence.
Every important statement in the narration must be traceable back to one or more confirmed transcript facts.

Never summarize multiple independent events into a single invented event.
Never simplify a character's motivation unless the transcript clearly supports it.
If several conversations gradually reveal information, preserve that gradual reveal.
Do not compress emotional development into one sentence.

Maintain the same causal chain as the original movie:
If Event B only happens because of Event A, the narration must explain Event A first.
Never reverse cause and effect.

When compressing scenes, remove repetition only.
Never remove information required to understand later scenes.
The audience should receive exactly the same understanding as someone who watched the entire movie.
The narration may be shorter than the movie, but it must never contain less truth.

=========================================================
PACING
=========================================================
Do not spend equal time on every scene.
Spend more narration on scenes that:
• change the story
• reveal important information
• change a character
• create emotional impact
Compress scenes that only repeat previous information.
Always preserve story continuity.

=========================================================
LANGUAGE
=========================================================
Write like spoken Vietnamese.
Avoid book-like writing.
Avoid overly poetic sentences.
Avoid repetitive transitions such as:
"Meanwhile..."
"After that..."
"Then..."
Instead, connect ideas naturally as people speak in high-quality YouTube movie recap videos.
The narration should sound conversational, emotional and effortless.

=========================================================
STRICT RULES
=========================================================
Never invent dialogue.
Never invent events.
Never invent scenes.
Never invent motivations.
Never invent relationships.
Never invent plot twists.
Never contradict the Whisper Transcript.
Never contradict confirmed Movie Memory.
Never reveal the ending too early.
If information cannot be confirmed, do not mention it.

=========================================================
FINAL SELF CHECK
=========================================================
Before returning the script silently verify:
✓ Every important event comes from the Whisper Transcript.
✓ Vision only enriches confirmed events.
✓ Character motivations are supported by evidence.
✓ Plot twists are preserved.
✓ No scene has been invented.
✓ Someone who has never watched the movie can understand the entire story.

If any sentence fails these checks, rewrite it before returning the final narration.

=========================================================
OUTPUT
=========================================================
Return ONLY the narration.
No markdown.
No titles.
No timestamps.
No scene numbers.
No explanations.
No notes.
The output must be immediately usable for Text-to-Speech.
The narration should be indistinguishable from a professional human YouTube movie recap creator."""

    def __init__(
        self,
        browser_mgr: Optional[Any] = None,
        openai_provider: Optional[Any] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
    ) -> None:
        self._browser_mgr = browser_mgr
        self._openai_provider = openai_provider
        self._sample_style = sample_style
        self._custom_sample_text = custom_sample_text

    def build_prompt_with_plan(
        self,
        memory: MovieMemory,
        plan: ReviewPlan,
        timeline: Any = None,
        language: str = "vi",
        custom_instructions: Optional[str] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
    ) -> str:
        """Construct multi-agent context prompt combining all 5 sources with strict priorities and sample styles."""
        from src.agents.sample_styles import get_sample_style_context

        chapters_outline = "\n".join([
            f"- Chapter {ch.chapter_index} [{ch.title}] (Tone: {ch.emotional_tone}, Suspense: {ch.suspense_level}/10): {ch.objective}\n"
            f"  Included Scenes: {ch.included_scenes} | Priority: {ch.review_priority}\n"
            f"  Key Events: {', '.join(ch.key_events)}"
            for ch in plan.chapters
        ])

        chars_str = ", ".join([f"{c.name} ({c.role}): {c.personality}, motivation: {c.motivation}" for c in memory.characters]) if memory.characters else "Characters in the movie"
        plot_str = "\n".join(memory.key_plot_points[:20])
        style_block = get_sample_style_context(
            sample_style or self._sample_style,
            custom_sample_text or self._custom_sample_text,
        )

        prompt = f"""{self.REVIEW_WRITER_SYSTEM_PROMPT}

{style_block}

=========================================================
SOURCE 4: STORY BLUEPRINT (HOW TO TELL THE STORY)
=========================================================
- Opening Hook: {plan.hook}
- Chapter Outline & Emotional Pacing:
{chapters_outline}
- Climax Strategy: {plan.climax_strategy.climax_focus or plan.climax_focus}
- Ending Strategy: {plan.ending_strategy.conclusion or plan.ending}
- Style: {plan.style}
- Language: {language}

=========================================================
SOURCE 1, 2 & 3: TRANSCRIPT FACTS, TIMELINE & MOVIE MEMORY
=========================================================
- Title/Topic: {memory.title} ({memory.genre} - {memory.main_theme})
- Confirmed Characters & Dynamics:
  {chars_str}
- Confirmed Chronological Events:
{plot_str}
{f"- Custom Directives: {custom_instructions}" if custom_instructions else ""}

Write a comprehensive, engaging, and detailed YouTube movie recap narration now in {language}.
- Tell the story thoroughly across all chapters from the opening hook, rising tension, climax to the ending.
- Maintain a smooth, gripping narrative pacing throughout the entire story.
Remember: Return ONLY the continuous narration text with NO titles, NO markdown formatting, NO scene tags, and NO timestamps."""
        return prompt.strip()

    @staticmethod
    def _clean_tts_script(raw_text: str) -> str:
        """Strip markdown, titles, section headers, and timestamps for clean TTS playback."""
        text = raw_text.strip()
        # Remove markdown headers and bolding markers
        text = re.sub(r"^#+\s*.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        # Remove common prefixes like Title:, Script:, Hook:, Chapter 1:, Scene 1:
        text = re.sub(r"^(?:Title|Tiêu đề|Hook|Mở đầu|Script|Kịch bản|Lời bình|Chapter \d+|Phần \d+|Scene \d+|Cảnh \d+):\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove timestamps like [00:15 - 01:30]
        text = re.sub(r"\[\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?\]", "", text)
        # Clean multiple blank lines
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return "\n\n".join(paragraphs)

    def write(
        self,
        memory: MovieMemory,
        plan: ReviewPlan,
        timeline: Any = None,
        language: str = "vi",
        custom_instructions: Optional[str] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ReviewResult:
        """Generate draft review script using OpenAI ChatGPT, Gemini Web, or deterministic fallback."""
        _logger.info("Review Writer Agent generating continuous review narration script...")
        start_ts = time.time()

        prompt = self.build_prompt_with_plan(
            memory=memory,
            plan=plan,
            timeline=timeline,
            language=language,
            custom_instructions=custom_instructions,
            sample_style=sample_style,
            custom_sample_text=custom_sample_text,
        )

        # 1. Prefer OpenAI ChatGPT if configured
        if self._openai_provider:
            _logger.info("Using OpenAI ChatGPT Review Writer...")
            try:
                raw_text = self._openai_provider.generate_chat_completion(
                    system_prompt=self.REVIEW_WRITER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                clean_script = self._clean_tts_script(raw_text)
                total_words = len(clean_script.split())
                return ReviewResult(
                    title=f"Review Phim: {memory.title}",
                    hook=plan.hook,
                    script=clean_script,
                    metadata=ReviewMetadata(
                        total_words=total_words,
                        estimated_duration=round(total_words / 2.5, 1),
                        model_name=getattr(self._openai_provider, "_model_name", "gpt-4o"),
                        processing_time=time.time() - start_ts,
                    ),
                )
            except Exception as exc:
                _logger.warning("OpenAI Writer failed, trying fallback: {}", exc)

        # 2. Prefer Gemini Web if available
        if self._browser_mgr:
            web_resp = self._browser_mgr.send_prompt(
                prompt=prompt,
                job_id=job_id,
                progress_callback=progress_callback,
            )
            raw_text = web_resp.text
            clean_script = self._clean_tts_script(raw_text)
            total_words = len(clean_script.split())
            result = ReviewResult(
                title=f"Review Phim: {memory.title}",
                hook=plan.hook,
                script=clean_script,
                metadata=ReviewMetadata(
                    total_words=total_words,
                    estimated_duration=round(total_words / 2.5, 1),
                    model_name=web_resp.model_name,
                    processing_time=time.time() - start_ts,
                ),
            )
        else:
            # 3. Deterministic synthesis fallback
            chapter_blocks = []
            for ch in plan.chapters:
                events_narrative = " ".join(ch.key_events) if ch.key_events else ch.objective
                chapter_blocks.append(events_narrative)

            script_parts = [
                plan.hook,
                plan.intro,
                "\n\n".join(chapter_blocks),
                plan.climax_strategy.climax_focus or plan.climax_focus,
                plan.ending_strategy.ending_text or plan.ending,
            ]
            full_script = self._clean_tts_script("\n\n".join([p for p in script_parts if p.strip()]))
            total_words = len(full_script.split())
            result = ReviewResult(
                title=f"Review Phim: {memory.title}",
                hook=plan.hook,
                script=full_script,
                metadata=ReviewMetadata(
                    total_words=total_words,
                    estimated_duration=round(total_words / 2.5, 1),
                    model_name="deterministic-writer",
                    processing_time=time.time() - start_ts,
                ),
            )

        _logger.info("Review Writer finished: created {} words narration script", result.metadata.total_words)
        return result

    def regenerate_sections(
        self,
        current_review: ReviewResult,
        issues: List[ContinuityIssue],
        memory: MovieMemory,
        plan: ReviewPlan,
        job_id: Optional[str] = None,
    ) -> ReviewResult:
        """Regenerate ONLY the affected sections flagged by Continuity Checker without rewriting the whole review."""
        _logger.info(
            "Review Writer Agent performing targeted regeneration for {} affected section(s)...",
            len(issues),
        )
        paragraphs = [p.strip() for p in current_review.script.split("\n\n") if p.strip()]
        if not paragraphs:
            return current_review

        # Group issues by affected section index
        issues_by_sec = {}
        for iss in issues:
            idx = max(0, min(iss.affected_section_index, len(paragraphs) - 1))
            issues_by_sec.setdefault(idx, []).append(iss)

        for sec_idx, sec_issues in issues_by_sec.items():
            original_text = paragraphs[sec_idx]
            issues_desc = "; ".join([f"[{i.issue_type}] {i.description} (Gợi ý: {i.suggestion})" for i in sec_issues])

            _logger.info(
                "Regenerating section {}/{} to fix: {}",
                sec_idx + 1,
                len(paragraphs),
                issues_desc,
            )

            if self._browser_mgr:
                prompt = f"""Bạn là Review Writer. Hãy viết lại DUY NHẤT đoạn văn sau để sửa lỗi tính liên tục/logic theo hướng dẫn:

- Đoạn văn gốc cần viết lại:
"{original_text}"

- Các lỗi cần khắc phục:
{issues_desc}

- Thông tin bối cảnh phim: {memory.genre} - {memory.main_theme}
- Các nhân vật: {', '.join([c.name for c in memory.characters])}

YÊU CẦU:
- Chỉ trả về DUY NHẤT đoạn văn đã được sửa chữa và hoàn thiện, không kèm lời chào hay giải thích gì thêm.
- Giữ nguyên độ dài và văn phong tự nhiên, liền mạch với các đoạn trước và sau.
"""
                try:
                    resp = self._browser_mgr.send_prompt(prompt=prompt, job_id=job_id)
                    fixed_text = resp.text.strip()
                    # Clean any accidental labels
                    fixed_text = re.sub(r"^(?:Script|Đoạn văn|Đoạn sửa lại):\s*", "", fixed_text, flags=re.IGNORECASE)
                    if len(fixed_text) > 20:
                        paragraphs[sec_idx] = fixed_text
                        _logger.info("Successfully updated section {} with fixed version.", sec_idx + 1)
                except Exception as exc:
                    _logger.warning("Browser regeneration failed for section {}: {}. Applying rule fix.", sec_idx + 1, exc)
                    # Apply local fallback fixes
                    for iss in sec_issues:
                        if iss.issue_type == "character_naming":
                            for c in memory.characters:
                                paragraphs[sec_idx] = re.sub(rf"\b{re.escape(c.name.lower())}\b", c.name, paragraphs[sec_idx])
            else:
                # Deterministic targeted fix
                fixed = original_text
                for iss in sec_issues:
                    if iss.issue_type == "character_naming":
                        for c in memory.characters:
                            fixed = re.sub(rf"\b{re.escape(c.name.lower())}\b", c.name, fixed)
                    elif iss.issue_type == "missing_plot_point":
                        fixed += f" Đồng thời, {memory.key_plot_points[0] if memory.key_plot_points else ''}"
                paragraphs[sec_idx] = fixed

        new_script = "\n\n".join(paragraphs)
        new_words = len(new_script.split())

        return ReviewResult(
            title=current_review.title,
            hook=current_review.hook,
            script=new_script,
            metadata=ReviewMetadata(
                total_words=new_words,
                estimated_duration=round(new_words / 2.5, 1),
                model_name=current_review.metadata.model_name,
                processing_time=current_review.metadata.processing_time,
            ),
        )
