import math
from typing import Dict, Any
from storyteller.algorithm.mcts_node import MCTSNode, ReportGenerationState
from storyteller.algorithm.evaluator import evaluate_report
from storyteller.algorithm.utils.html2image import convert_html_to_image
import base64
import os

class StorytellingRewardModel:
    def __init__(self, llm_kwargs: Dict[str, Any] = None):
        """
        Data story MCTS reward function

        Args:
            llm_kwargs: LLM call parameters
        """
        self.llm_kwargs = llm_kwargs or {}
        self.last_quality_reward = 0.0

    def compute_reward(self, node: MCTSNode, html_path: str, image_path: str) -> float:
        """Calculate reward value for the node"""
        quality_reward = self._compute_quality_reward(node, html_path, image_path)
        self.last_quality_reward = quality_reward

        print(f"[REWARD] Quality score: {quality_reward:.2f}")

        return quality_reward

    def _compute_quality_reward(self, node: MCTSNode, html_path: str, image_path: str) -> float:
        """Calculate quality reward (0-10 points)"""
        try:
            if node.node_type != ReportGenerationState.FINALIZED:
                return 5.0

            dataset_context = node.report.data_context or ""
            query = node.report.original_query

            md_path = os.path.join(os.path.dirname(html_path), "report.md")

            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                print(f"[SUCCESS] Successfully read Markdown report: {md_path}")
            except Exception as e:
                print(f"[WARNING] Cannot read Markdown report: {str(e)}, trying to convert from HTML")
                with open(html_path, 'r', encoding='utf-8') as f:
                    md_content = f"# HTML Report Content (Markdown unavailable)\n\n```html\n{f.read()}\n```"

            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()

            quality_score = evaluate_report(
                dataset_context=dataset_context,
                query=query,
                md_report=md_content,
                report_image=image_base64,
                llm_kwargs=self.llm_kwargs
            )

            return quality_score

        except Exception as e:
            print(f"[ERROR] Quality evaluation error: {str(e)}")
            return 5.0

    def _generate_html_report(self, node: MCTSNode) -> str:
        """Generate HTML format report"""
        return str(node.report)