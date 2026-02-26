"""
feedback/generator.py
---------------------
Converts ScoreResult into human-readable feedback messages
and color-coded joint overlay data for the frontend.

Pipeline position:
    scorer.py → [generator.py] → api/pipeline.py
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from core.scorer import ScoreResult


@dataclass
class FeedbackResult:
    """
    Final output sent to the frontend via the API.
    """
    overall_score: float = 0.0
    is_correct: bool = False
    target_sign: str = ""

    # Human readable messages (max 3, prioritized by severity)
    messages: list[str] = field(default_factory=list)

    # Positive reinforcement message
    praise: str = ""

    # Per-joint color for skeleton overlay
    # "green" = correct, "orange" = close, "red" = wrong
    joint_colors: dict[str, str] = field(default_factory=dict)

    # Emoji summary for Duolingo-style UI
    emoji: str = "🤔"

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "is_correct": self.is_correct,
            "target_sign": self.target_sign,
            "messages": self.messages,
            "praise": self.praise,
            "joint_colors": self.joint_colors,
            "emoji": self.emoji,
        }


class FeedbackGenerator:
    """
    Generates user-facing feedback from a ScoreResult.

    Loads sign-specific message templates from feedback/templates/signs.json
    and generates contextual correction messages.
    """

    JOINT_TO_FINGER = {
        "thumb_mcp": "thumb", "thumb_ip": "thumb",
        "index_mcp": "index finger", "index_pip": "index finger",
        "middle_mcp": "middle finger", "middle_pip": "middle finger",
        "ring_mcp": "ring finger", "ring_pip": "ring finger",
        "pinky_mcp": "pinky", "pinky_pip": "pinky",
    }

    JOINT_TO_ACTION = {
        "mcp": "at the base",
        "pip": "in the middle",
        "ip": "at the tip",
    }

    def __init__(self, templates_path: str = "feedback/templates/signs.json"):
        self.templates_path = Path(templates_path)
        self._templates = self._load_templates()

    def generate(self, score_result: ScoreResult) -> FeedbackResult:
        """
        Generate feedback from a ScoreResult.

        Args:
            score_result: Output from ASLScorer.score()

        Returns:
            FeedbackResult ready to send to frontend
        """
        fb = FeedbackResult(
            overall_score=score_result.overall_score,
            is_correct=score_result.is_correct,
            target_sign=score_result.target_sign,
        )

        # ── Emoji ──
        fb.emoji = self._get_emoji(score_result.overall_score)

        # ── Praise ──
        fb.praise = self._get_praise(score_result.overall_score)

        # ── Joint Colors ──
        fb.joint_colors = self._compute_joint_colors(score_result)

        # ── Correction Messages ──
        fb.messages = self._generate_messages(score_result)

        return fb

    def _get_emoji(self, score: float) -> str:
        if score >= 90:   return "🌟"
        if score >= 75:   return "✅"
        if score >= 50:   return "👍"
        if score >= 25:   return "🤔"
        return "❌"

    def _get_praise(self, score: float) -> str:
        if score >= 90:   return "Perfect! Great sign!"
        if score >= 75:   return "Nice work! Almost perfect."
        if score >= 50:   return "Good try! A few adjustments needed."
        if score >= 25:   return "Keep practicing, you're getting there!"
        return "Let's try again — watch the example carefully."

    def _compute_joint_colors(self, result: ScoreResult) -> dict[str, str]:
        colors = {}
        for joint, score in result.joint_scores.items():
            if score >= 0.8:
                colors[joint] = "green"
            elif score >= 0.5:
                colors[joint] = "orange"
            else:
                colors[joint] = "red"

        # Position and orientation
        colors["position"] = "green" if result.position_score >= 0.8 else \
                             "orange" if result.position_score >= 0.5 else "red"
        colors["orientation"] = "green" if result.orientation_score >= 0.8 else \
                                "orange" if result.orientation_score >= 0.5 else "red"
        return colors

    def _generate_messages(self, result: ScoreResult) -> list[str]:
        messages = []

        # Check for sign-specific template messages first
        template = self._templates.get(result.target_sign.upper(), {})
        template_hints = template.get("hints", {})

        # Find worst-scoring joints
        sorted_joints = sorted(
            result.joint_scores.items(),
            key=lambda x: x[1]
        )

        for joint, score in sorted_joints:
            if score >= 0.8:
                break  # all remaining are fine
            if len(messages) >= 3:
                break  # max 3 messages

            # Check template for specific hint
            if joint in template_hints:
                messages.append(template_hints[joint])
                continue

            # Generate generic message
            msg = self._joint_message(joint, score, result.angle_deviations.get(joint, 0))
            if msg:
                messages.append(msg)

        # Position feedback
        if result.position_score < 0.6 and len(messages) < 3:
            pos_hint = template.get("position_hint", "")
            messages.append(pos_hint if pos_hint else "Adjust your hand position relative to your face/body.")

        # Orientation feedback
        if result.orientation_score < 0.6 and len(messages) < 3:
            ori_hint = template.get("orientation_hint", "")
            messages.append(ori_hint if ori_hint else "Rotate your palm to face the correct direction.")

        return messages

    def _joint_message(self, joint: str, score: float, deviation: float) -> str:
        finger = self.JOINT_TO_FINGER.get(joint, "finger")
        severity = "slightly" if score >= 0.5 else "more"

        # Determine if curling or extending
        if "pip" in joint or "ip" in joint:
            if deviation > 0:
                return f"Curl your {finger} {severity} {self.JOINT_TO_ACTION.get(joint.split('_')[1], '')}."
            else:
                return f"Straighten your {finger} {severity} {self.JOINT_TO_ACTION.get(joint.split('_')[1], '')}."
        elif "mcp" in joint:
            return f"Adjust your {finger} at the knuckle."
        elif "splay" in joint:
            return "Spread your fingers differently."
        return ""

    def _load_templates(self) -> dict:
        if not self.templates_path.exists():
            return {}
        with open(self.templates_path) as f:
            return json.load(f)
