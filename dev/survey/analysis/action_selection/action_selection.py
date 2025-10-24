import dspy
import argparse
import csv
import json
import os
from typing import Literal, Optional, List

from dotenv import load_dotenv
from PIL import Image as PILImage
import pydantic

class ActionFeasibility(pydantic.BaseModel):
    action: str
    missing_context: Optional[str] = None
    feasibility: Literal[1, 2, 3, 4, 5]

class FeasibilityEstimationSignature(dspy.Signature):
    """
    Given a project scratchpad and a set of potential next steps/suggestions, estimate the feasibility of you being able to help complete the next steps/suggestions without requiring the user to do any work or share any context.

    You should consider the following scoring rubric:
    - 1: This action is extremely ambiguous and attempting to complete it is likely to require the user to do significant work or share significant context.
    - 2: This action is somewhat ambiguous and attempting it without more context is unlikely to be successful.
    - 3: This action has some relevant context.  With appropriate tools such as file agents, code agents, or google drive agents you should be able to gather enough context to complete the action.
    - 4: This action is mostly clear and given the right tools you should be able to complete it without requiring the user to do any work or share any context.
    - 5: This action is completely clear given the project scratchpad as context.  You are highly confident in the feasibility of you being able to complete the action without requiring the user to do any work or share any context.
    """
    project_scratchpad: str = dspy.InputField(description="The current project scratchpad")
    next_steps: list[str] = dspy.InputField(description="A list of potential next steps/suggestions")
    feasibility: list[ActionFeasibility] = dspy.OutputField(description="The feasibility of you being able to help complete the next steps/suggestions without requiring the user to do any work")

class FeasibilityEstimation(dspy.Module):
    def __init__(self, batch_size: int = 10):
        super().__init__()
        self.estimator = dspy.ChainOfThought(FeasibilityEstimationSignature)
        self.batch_size = batch_size

    def forward(self, project_scratchpad: str):
        def _extract_section(text: str, header: str) -> str:
            parts = text.split(header, 1)
            if len(parts) < 2:
                return ""
            tail = parts[1]
            # stop at next header start "## " if present
            if "\n## " in tail:
                tail = tail.split("\n## ", 1)[0]
            return tail.strip()

        # extract sections safely
        suggestions_block = _extract_section(project_scratchpad or "", "## Suggestions")
        next_steps_block = _extract_section(project_scratchpad or "", "## Next Steps")

        def _lines_to_actions(block: str) -> List[str]:
            if not block:
                return []
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            actions: List[str] = []
            for ln in lines:
                # Expected format: "[idx] text (confidence: N)"
                body = ln.split("] ", 1)[1] if "] " in ln else ln
                body = body.split(" (confidence:", 1)[0].strip()
                if body:
                    actions.append(body)
            return actions

        suggestions = _lines_to_actions(suggestions_block)
        next_steps = _lines_to_actions(next_steps_block)

        # Estimate the feasibility of each suggestion/next step
        dataset = []
        full_suggestions = suggestions + next_steps
        if not full_suggestions:
            return []
        for i in range(0, len(full_suggestions), self.batch_size):
            batch = full_suggestions[i:min(i + self.batch_size, len(full_suggestions))]
            dataset.append(
                dspy.Example(project_scratchpad=project_scratchpad, next_steps=batch)
                .with_inputs("project_scratchpad", "next_steps")
            )
        outputs = self.estimator.batch(dataset)
        return [feasibility for output in outputs for feasibility in output.feasibility]