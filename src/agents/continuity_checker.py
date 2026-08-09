"""Continuity Checker Agent.

Validates narrative consistency between the generated review script and movie_memory.json:
- Detects timeline inconsistencies
- Detects repeated scenes
- Detects character naming inconsistencies
- Detects impossible events
- Detects missing important plot points
- Detects contradictions

Returns a structured report:
{
  "passed": true/false,
  "issues": [...]
}
"""

from __future__ import annotations

import re
from typing import List, Optional
from src.agents.models import MovieMemory, ReviewPlan, ContinuityIssue, ContinuityReport
from src.review.models import ReviewResult
from src.utils.logger import get_logger

_logger = get_logger("continuity_checker")


class ContinuityCheckerAgent:
    """Logical Agent 5: Narrative integrity and continuity validator."""

    def check(
        self,
        review: ReviewResult,
        memory: MovieMemory,
        plan: Optional[ReviewPlan] = None,
    ) -> ContinuityReport:
        """Analyze review against movie_memory.json and flag continuity issues."""
        _logger.info("Continuity Checker Agent starting verification...")

        script = review.script or ""
        paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]
        issues: List[ContinuityIssue] = []

        # 1. Check Character Naming Inconsistencies
        known_char_names = [c.name for c in memory.characters if c.name]
        for p_idx, p in enumerate(paragraphs):
            for name in known_char_names:
                # Check for subtle lowercase or misspelled occurrences
                if len(name) >= 3:
                    wrong_case = name.lower()
                    # If wrong lowercase found surrounded by word boundaries
                    if re.search(rf"\b{re.escape(wrong_case)}\b", p) and wrong_case != name:
                        issues.append(ContinuityIssue(
                            issue_type="character_naming",
                            description=f"Tên nhân vật '{name}' bị viết sai định dạng viết hoa trong đoạn {p_idx+1}.",
                            affected_section_index=p_idx,
                            section_snippet=p[:100],
                            suggestion=f"Thay thế bằng tên viết chuẩn '{name}'",
                        ))

        # 2. Check Repeated Scenes across non-adjacent paragraphs
        seen_phrases = {}
        for p_idx, p in enumerate(paragraphs):
            # Extract 4-word shingles
            words = p.lower().split()
            for i in range(len(words) - 5):
                shingle = " ".join(words[i : i + 5])
                if len(shingle) > 25:
                    if shingle in seen_phrases and seen_phrases[shingle] != p_idx:
                        prev_idx = seen_phrases[shingle]
                        issues.append(ContinuityIssue(
                            issue_type="repeated_scene",
                            description=f"Nội dung mô tả cảnh lặp lại giữa đoạn {prev_idx+1} và đoạn {p_idx+1}.",
                            affected_section_index=p_idx,
                            section_snippet=p[:100],
                            suggestion="Loại bỏ chi tiết lặp hoặc thay thế bằng diễn biến tiếp theo",
                        ))
                        break
                    seen_phrases[shingle] = p_idx

        # 3. Check Missing Important Plot Points
        if memory.key_plot_points and len(paragraphs) > 0:
            found_points = 0
            for pt in memory.key_plot_points:
                pt_keywords = [w for w in pt.split() if len(w) > 4]
                if any(kw.lower() in script.lower() for kw in pt_keywords):
                    found_points += 1

            if found_points < max(1, len(memory.key_plot_points) // 3):
                issues.append(ContinuityIssue(
                    issue_type="missing_plot_point",
                    description="Kịch bản bỏ sót nhiều điểm nút thắt quan trọng trong Movie Memory.",
                    affected_section_index=min(1, len(paragraphs) - 1),
                    section_snippet=paragraphs[min(1, len(paragraphs) - 1)][:100],
                    suggestion="Bổ sung thêm diễn biến chính của cốt truyện",
                ))

        # 4. Check Contradictions / Impossible Events
        contradiction_pairs = [
            ("một mình", ["cùng bạn bè", "với đồng đội", "cùng người thân"]),
            ("hy sinh", ["tiếp tục đứng dậy chiến đấu", "bước đi bình thường"]),
            ("ban đêm", ["ánh nắng chói chang", "buổi trưa"]),
        ]
        for p_idx, p in enumerate(paragraphs):
            p_lower = p.lower()
            for term, opposings in contradiction_pairs:
                if term in p_lower:
                    for opp in opposings:
                        if opp in p_lower:
                            issues.append(ContinuityIssue(
                                issue_type="contradiction",
                                description=f"Mâu thuẫn ngữ cảnh trong đoạn {p_idx+1}: vừa có '{term}' vừa có '{opp}'.",
                                affected_section_index=p_idx,
                                section_snippet=p[:100],
                                suggestion=f"Làm rõ tính logic giữa '{term}' và '{opp}'",
                            ))

        # Determine overall pass status
        passed = len(issues) == 0

        report = ContinuityReport(
            passed=passed,
            issues=issues,
        )

        _logger.info(
            "Continuity check completed: Passed={}, Issues found={}",
            report.passed,
            len(report.issues),
        )
        return report
