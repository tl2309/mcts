import random
import math
import copy
import os
import json
import sys
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote
import shutil

from storyteller.algorithm.mcts_node import MCTSNode, Report, ReportGenerationState
from storyteller.algorithm.mcts_action import (
    DataStorytellingAction,
    Query2Chapters,
    Chapters2Tasks,
    Tasks2Charts,
    ReviseVis,
    Charts2Captions,
    Captions2Summaries,
    ReviseNarrativeStrategy,
    TransitionAction
)
NODE_TYPE_TO_VALID_ACTIONS = {
    ReportGenerationState.EMPTY: [
        Query2Chapters
    ],
    ReportGenerationState.a1: [
        Chapters2Tasks,
    ],
    ReportGenerationState.a2: [
        Tasks2Charts
    ],
    ReportGenerationState.a3: [
        ReviseVis,
        Charts2Captions
    ],
    ReportGenerationState.a4: [
        Charts2Captions
    ],
    ReportGenerationState.a5: [
        Captions2Summaries,
    ],
    ReportGenerationState.a6: [
        ReviseNarrativeStrategy,
        TransitionAction
    ],
    ReportGenerationState.REVISECHAPTERSORDERS: [
        TransitionAction
    ], 
    ReportGenerationState.FINALIZED: []
}
from storyteller.algorithm.reward import StorytellingRewardModel
from .utils.html2image import convert_html_file_to_image

class DataStorytellingMCTSSolver:
    def __init__(self, 
                 original_query: str,
                 dataset_path: str,
                 output_dir: str,
                 max_iterations: int = 100,
                 max_depth: int = 10,
                 exploration_constant: float = 1.414,
                 data_context: str = "",
                 llm_kwargs: dict = None):
        """
        MCTS solver for automatic data story generation.

        Args:
            original_query: User input question
            dataset_path: Dataset path
            output_dir: Output directory
            max_iterations: Maximum search iterations
            max_depth: Maximum search depth
            exploration_constant: Exploration constant in UCB1 formula
            data_context: Dataset context information
            llm_kwargs: Parameters passed to LLM (large model)
        """
        self.original_query = original_query
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.data_context = json.load(open(data_context, 'r', encoding='utf-8'))
        self.llm_kwargs = llm_kwargs or {}

        # Create reward model
        self.reward_model = StorytellingRewardModel(llm_kwargs=self.llm_kwargs)

        # Define action space
        self.action_space = [
            Query2Chapters(),
            Chapters2Tasks(),
            Tasks2Charts(),
            ReviseVis(),
            Charts2Captions(),
            Captions2Summaries()
        ]

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize root node - use original MCTSNode initialization method
        self.root = MCTSNode(
            node_type=ReportGenerationState.EMPTY,
            report=Report(
                original_query=self.original_query, 
                dataset_path=self.dataset_path, 
                data_context=self.data_context
            ),
            original_query=self.original_query,
            llm_kwargs=self.llm_kwargs
        )

        # Add best node tracking
        self.best_node = self.root
        self.best_score = float('-inf')


    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Use UCB1 formula to select the most promising Node for expansion.

        Args:
            node: Current MCTS node

        Returns:
            Selected `Node`
        """
        while node.children:
            if any(child.N == 0 for child in node.children):
                return next(child for child in node.children if child.N == 0)

            # Select child node with highest UCB1 score
            node = max(node.children, key=lambda c: (c.Q / c.N) + self.exploration_constant * math.sqrt(math.log(node.N) / c.N))
        return node
    def expand(self, node: MCTSNode) -> None:
        """Expand leaf node, add all possible child nodes"""
        print("[PROCESS] Expanding node...")
        print(f"\n[DEBUG]:")
        
        if node.children:
            print(f"[WARNING] Node {node.node_type} already has {len(node.children)} child nodes before expansion, will clear these child nodes")
            node.children = []
        
        current_state = node.node_type
        print(f"Current state: {current_state}")
        
        valid_action_types = NODE_TYPE_TO_VALID_ACTIONS.get(current_state, [])
        
        if not valid_action_types:
            print(f"[WARNING] State {current_state.name} has no valid action types")
            return
        
        print(f"Found {len(valid_action_types)} available action types")
        
        for action_class in valid_action_types:
            try:
                action_instance = action_class()
                print(f"Trying to execute action: {action_class.__name__}")
                
                children = action_instance.create_children_nodes(node, self.llm_kwargs)
                
                if not children:
                    print(f"[WARNING] Action {action_class.__name__} did not generate any child nodes, trying to create a default child node")
                    default_child = copy.deepcopy(node)
                    default_child.parent_node = node
                    default_child.parent_action = action_instance
                    default_child.depth = node.depth + 1
                    
                    if action_class == Query2Chapters:
                        default_child.node_type = ReportGenerationState.a1
                    elif action_class == Chapters2Tasks:
                        default_child.node_type = ReportGenerationState.a2
                    elif action_class == Tasks2Charts:
                        default_child.node_type = ReportGenerationState.a3
                    elif action_class == ReviseVis:
                        default_child.node_type = ReportGenerationState.a4
                    elif action_class == Charts2Captions:
                        default_child.node_type = ReportGenerationState.a5
                    elif action_class == Captions2Summaries:
                        default_child.node_type = ReportGenerationState.FINALIZED
                    
                    children = [default_child]
                    print(f"[SUCCESS] Created a default child node for action {action_class.__name__}")
                else:
                    print(f"[SUCCESS] Action {action_class.__name__} generated {len(children)} child nodes")
                
                node.children.extend(children)
                
                current_iteration = self.root.report.current_iteration
                for child in children:
                    child.report.current_iteration = current_iteration
                
            except Exception as e:
                print(f"[ERROR] Error executing action {action_class.__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not node.children:
            print("[WARNING] No child nodes generated after expansion")
        else:
            print(f"[SUCCESS] Generated {len(node.children)} child nodes in total")
        
        random.shuffle(node.children)

    def simulate(self, node: MCTSNode) -> tuple[MCTSNode, float]:
        """Simulation phase: Randomly execute actions from current node until reaching terminal state"""
        print("[PROCESS] Simulation phase...")
        
        current = copy.deepcopy(node)
        current.report.current_iteration = self.root.report.current_iteration
        
        reward = 5.0
        
        while not current.is_terminal() and current.depth < self.max_depth:
            self.expand(current)
            
            if not current.children:
                print("[WARNING] No child nodes after expansion, stopping simulation")
                break
            
            current = random.choice(current.children)
            current.report.current_iteration = self.root.report.current_iteration
            print(f"[SIMULATE] Entering state: {current.node_type.name} (depth {current.depth})")
        
        if current.is_terminal():
            print("[SUCCESS] Simulation generated complete report!")
            
            iteration_dir = os.path.join(self.output_dir, "iterations", f"iteration_{current.report.current_iteration}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            html_path, original_contents = self._save_html_report(current, os.path.join(iteration_dir, "report.html"))
            
            html_dir = os.path.dirname(html_path)
            html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
            
            png_path = None
            if html_files:
                selected_html = random.choice(html_files)
                selected_html_path = os.path.join(html_dir, selected_html)
                print(f"[RANDOM] Randomly selected {selected_html} for PNG conversion...")
                
                if selected_html in original_contents:
                    with open(selected_html_path, 'w', encoding='utf-8') as f:
                        f.write(original_contents[selected_html])
                
                png_path = os.path.splitext(selected_html_path)[0] + ".png"
                convert_html_file_to_image(selected_html_path, png_path)
                print(f"[SUCCESS] PNG file generated: {png_path}")
                
                html_content = original_contents[selected_html]
                user_home = os.path.expanduser("~")
                project_base = os.path.join(user_home, "mcts")
                if os.path.exists(project_base):
                    html_content = html_content.replace(project_base, "")
                else:
                    import re
                    pattern = r'(src=")(\/.*?\/mcts)(\/storyteller\/.*?\.png")'
                    html_content = re.sub(pattern, r'\1\3', html_content)
                
                with open(selected_html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            try:
                if png_path and os.path.exists(png_path):
                    quality_reward = self.reward_model._compute_quality_reward(current, html_path, png_path)
                    print(f"[INFO] Quality reward calculated: {quality_reward:.2f}")
                    
                    reward = quality_reward
                else:
                    print(f"[WARNING] PNG chart not found, cannot calculate quality reward, using default reward: {reward:.2f}")
                    
                print(f"[INFO] Final reward: {reward:.2f}")
                
            except Exception as e:
                print(f"[ERROR] Error calculating reward: {str(e)}")
                print(f"[WARNING] Using default reward: {reward:.2f}")
        else:
            print(f"[WARNING] Terminal state not reached, using default reward: {reward:.2f}")
        
        return current, reward

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backtrack phase: Update statistics of all nodes on the path
        """
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent_node

    def update_selected_path_iteration(self, node: MCTSNode) -> None:
        """
        Update iteration numbers of all nodes on selected path, and copy charts from previous iterations to current iteration

        Args:
            node: Currently selected node
        """
        import os
        import re
        import shutil

        current_iteration = self.root.report.current_iteration

        while node is not None:
            old_iteration = node.report.current_iteration

            node.report.current_iteration = current_iteration

            if old_iteration != current_iteration:
                current_charts_dir = os.path.join(self.output_dir, "iterations", f"iteration_{current_iteration}", "charts")
                os.makedirs(current_charts_dir, exist_ok=True)

                for chapter in node.report.chapters:
                    for chart in getattr(chapter, 'charts', []):
                        if hasattr(chart, 'url') and chart.url:
                            old_iteration_pattern = rf"iteration_{old_iteration}"
                            if old_iteration_pattern in chart.url:
                                try:
                                    chart_filename = os.path.basename(chart.url)

                                    old_chart_path = chart.url
                                    new_chart_path = os.path.join(current_charts_dir, chart_filename)

                                    new_chart_url = os.path.join(
                                        self.output_dir, "iterations",
                                        f"iteration_{current_iteration}", "charts", chart_filename
                                    )

                                    if os.path.exists(old_chart_path):
                                        shutil.copy2(old_chart_path, new_chart_path)
                                        print(f"[SUCCESS] Copied chart from {old_chart_path} to {new_chart_path}")
                                    else:
                                        print(f"[WARNING] Source chart file not found: {old_chart_path}")

                                    chart.url = new_chart_url
                                    print(f"[SUCCESS] Updated chart URL to: {new_chart_url}")

                                except Exception as e:
                                    print(f"[ERROR] Error copying chart: {str(e)}")

            node = node.parent_node

    def solve(self) -> MCTSNode:
        """Execute MCTS search"""
        log_file = os.path.join("storyteller", "output", "log.txt")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        original_stdout = sys.stdout

        log_f = open(log_file, 'w', encoding='utf-8')

        class TeeOutput:
            def __init__(self, file):
                self.file = file
                self.stdout = original_stdout

            def write(self, message):
                self.stdout.write(message)
                self.file.write(message)
                self.file.flush()

            def flush(self):
                self.stdout.flush()
                self.file.flush()

        sys.stdout = TeeOutput(log_f)

        try:
            print("\n[SEARCH] MCTS search started")
            print("=" * 50)
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            history_dir = os.path.join(self.output_dir, "iterations")
            os.makedirs(history_dir, exist_ok=True)

            start_time = datetime.now()
            best_node = None
            best_score = float('-inf')

            for iteration in range(self.max_iterations):
                self.root.report.current_iteration = iteration + 1
                print(f"[DEBUG] Set root node iteration to {self.root.report.current_iteration}")

                iteration_dir = os.path.join(history_dir, f"iteration_{iteration + 1}")
                os.makedirs(iteration_dir, exist_ok=True)
                os.makedirs(os.path.join(iteration_dir, "charts"), exist_ok=True)

                print(f"\n[ITERATION] **MCTS iteration {iteration + 1}/{self.max_iterations}**")

                leaf = self.select(self.root)
                print(f"[SELECT] Selected node (depth {leaf.depth}) | State: {leaf.node_type}")

                self.update_selected_path_iteration(leaf)
                print(f"[DEBUG] Updated selected node iteration to {leaf.report.current_iteration}")

                self.expand(leaf)

                if leaf.children:
                    child_for_simulation = random.choice(leaf.children)
                    final_node, simulated_reward = self.simulate(child_for_simulation)
                else:
                    final_node, simulated_reward = self.simulate(leaf)

                iteration_dir = os.path.join(history_dir, f"iteration_{iteration + 1}")
                os.makedirs(iteration_dir, exist_ok=True)

                html_path, original_contents = self._save_html_report(final_node,
                    output_path=os.path.join(iteration_dir, "report.html"))

                html_files = [f for f in os.listdir(iteration_dir) if f.endswith('.html')]
                if html_files:
                    selected_html = html_files[0]
                    selected_html_path = os.path.join(iteration_dir, selected_html)

                    if selected_html in original_contents:
                        with open(selected_html_path, 'w', encoding='utf-8') as f:
                            f.write(original_contents[selected_html])

                    image_path = os.path.splitext(selected_html_path)[0] + ".png"
                    convert_html_file_to_image(selected_html_path, image_path)

                    html_content = original_contents[selected_html]
                    user_home = os.path.expanduser("~")
                    project_base = os.path.join(user_home, "mcts")
                    if os.path.exists(project_base):
                        html_content = html_content.replace(project_base, "")
                    else:
                        import re
                        pattern = r'(src=")(\/.*?\/mcts)(\/storyteller\/.*?\.png")'
                        html_content = re.sub(pattern, r'\1\3', html_content)

                    with open(selected_html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                else:
                    image_path = os.path.join(iteration_dir, "report.png")

                score_info = {
                    "iteration": iteration + 1,
                    "score": float(simulated_reward),
                    "score_breakdown": {
                        "quality_reward": simulated_reward
                    },
                    "is_best": simulated_reward > best_score,
                    "node_type": final_node.node_type.name,
                    "depth": final_node.depth,
                    "chapter_count": len(final_node.report.chapters),
                    "chart_count": sum(len(chapter.charts) for chapter in final_node.report.chapters),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_time": str(datetime.now() - start_time)
                }

                with open(os.path.join(iteration_dir, "score.json"), 'w', encoding='utf-8') as f:
                    json.dump(score_info, f, indent=2, ensure_ascii=False)

                print(f"[SUCCESS] Iteration {iteration + 1} report saved to: {iteration_dir}")
                print(f"   Score: {simulated_reward:.2f}")

                if simulated_reward > best_score:
                    best_score = simulated_reward
                    best_node = copy.deepcopy(final_node)
                    print(f"[IMPROVE] Found better complete report! Score: {best_score:.2f}")

                self.backpropagate(leaf, simulated_reward)
                print(f"   [STATS] Q value updated: {leaf.Q}, Visit count: {leaf.N}")
                print("-" * 50)

            history_summary = {
                "total_iterations": self.max_iterations,
                "completed_iterations": iteration + 1,
                "best_score": float(best_score),
                "best_iteration": max(range(1, iteration + 2),
                    key=lambda i: os.path.exists(os.path.join(history_dir, f"iteration_{i}", "score.json")) and
                        json.load(open(os.path.join(history_dir, f"iteration_{i}", "score.json")))["score"]),
                "final_depth": best_node.depth if best_node else 0,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration": str(datetime.now() - start_time)
            }

            with open(os.path.join(history_dir, "search_summary.json"), 'w', encoding='utf-8') as f:
                json.dump(history_summary, f, indent=2, ensure_ascii=False)

            print("\n[SUCCESS] MCTS search completed!")
            print("=" * 50)

            if best_node.node_type.name == "FINALIZED":
                print(f"[RESULT] Return best complete report | Score: {best_score:.2f}")
                return best_node
            else:
                print("[WARNING] No complete report found, returning root node")
                return self.root

        finally:
            sys.stdout = original_stdout
            log_f.close()

    def _save_html_report(self, node: MCTSNode, output_path: str = None) -> str:
        """
        Generate and save HTML report

        Args:
            node: Current node
            output_path: Specified output path (optional)

        Returns:
            str: HTML file path
        """
        try:
            if output_path is None:
                default_dir = os.path.join(self.output_dir, "iterations", "default")
                os.makedirs(default_dir, exist_ok=True)
                output_path = os.path.join(default_dir, "report.html")

            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            markdown_content = self._generate_markdown_report(node)

            md_path = os.path.join(output_dir, "report.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            process_script = os.path.join(script_dir, "utils", "process_all_reports.py")

            print(f"[INFO] Generating all style reports for {output_dir}...")
            process_result = subprocess.run([
                'python',
                process_script,
                '--all',
                '--dir', output_dir
            ], check=True, capture_output=True, text=True)

            if process_result.stdout:
                print("--- HTML generation process output ---")
                print(process_result.stdout)
            if process_result.stderr:
                print("--- HTML generation process error ---")
                print(process_result.stderr)

            if not os.path.exists(output_path):
                html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
                if html_files:
                    output_path = os.path.join(output_dir, html_files[0])
                    print(f"[INFO] Using generated HTML file: {output_path}")

            html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
            original_contents = {}

            for html_file in html_files:
                html_file_path = os.path.join(output_dir, html_file)
                try:
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        original_contents[html_file] = f.read()
                except Exception as e:
                    print(f"[ERROR] Error reading HTML file {html_file}: {str(e)}")

            for html_file in html_files:
                html_file_path = os.path.join(output_dir, html_file)
                try:
                    html_content = original_contents[html_file]

                    user_home = os.path.expanduser("~")
                    project_base = os.path.join(user_home, "mcts")

                    if os.path.exists(project_base):
                        html_content = html_content.replace(project_base, "")
                    else:
                        import re
                        pattern = r'(src=")(\/.*?\/mcts)(\/storyteller\/.*?\.png")'
                        html_content = re.sub(pattern, r'\1\3', html_content)

                    with open(html_file_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    print(f"[SUCCESS] Fixed image paths in HTML file: {html_file}")

                except Exception as e:
                    print(f"[ERROR] Error processing HTML file {html_file}: {str(e)}")

            return output_path, original_contents

        except Exception as e:
            print(f"[ERROR] Error saving HTML report: {str(e)}")
            raise e

    def _generate_markdown_report(self, node: MCTSNode) -> str:
        """Generate Markdown report"""
        markdown = []

        markdown.append("# Data Analysis Report\n")

        if hasattr(node.report, 'key_abstract') and node.report.key_abstract:
            markdown.append("## Abstract\n")
            markdown.append(node.report.key_abstract + "\n")

        for i, chapter in enumerate(node.report.chapters):
            chapter_title = chapter.title
            if isinstance(chapter_title, str) and chapter_title.startswith("{'title': '") and chapter_title.endswith("'}"):
                chapter_title = chapter_title[len("{'title': '"):-2]

            markdown.append(f"\n## {chapter_title}\n")

            if hasattr(chapter, 'transition') and chapter.transition:
                markdown.append(f"{chapter.transition}\n\n")

            charts_by_group = {}
            has_groups = False

            if hasattr(chapter, 'chart_groups') and chapter.chart_groups:
                has_groups = True

                for group in chapter.chart_groups:
                    group_id = group.get('group_id', 0)
                    chart_indices = group.get('chart_indices', [])
                    group_theme = group.get('theme', 'Chart Group')
                    group_captions = []
                    group_charts = []

                    for idx in chart_indices:
                        if 0 <= idx < len(chapter.charts):
                            chart = chapter.charts[idx]
                            if hasattr(chart, 'url') and chart.url:
                                group_charts.append(chart)
                                if hasattr(chart, 'caption') and chart.caption:
                                    group_captions.append(chart.caption)

                    if group_charts:
                        group_caption = group_captions[0] if group_captions else f"Chart Group: {group_theme}"
                        charts_by_group[group_id] = {
                            'charts': group_charts,
                            'caption': group_caption,
                            'theme': group_theme
                        }

            if has_groups and charts_by_group:
                for group_id, group_info in charts_by_group.items():
                    group_charts = group_info['charts']
                    group_caption = group_info['caption']
                    group_theme = group_info['theme']

                    markdown.append(f"\n### {group_theme}\n")

                    markdown.append(f"\n> {group_caption}\n")

                    markdown.append("\n<!-- chart-group-start -->\n")

                    for chart in group_charts:
                        try:
                            img_filename = os.path.basename(chart.url)

                            markdown.append(f"![{group_theme}](charts/{img_filename})\n")
                        except Exception as e:
                            print(f"[ERROR] Error processing group image path: {str(e)}")
                            continue

                    markdown.append("\n<!-- chart-group-end -->\n")
            else:
                for chart in getattr(chapter, 'charts', []):
                    if hasattr(chart, 'caption') and chart.caption:
                        markdown.append(f"\n> {chart.caption}\n")

                    if hasattr(chart, 'url') and chart.url:
                        try:
                            img_filename = os.path.basename(chart.url)

                            markdown.append(f"\n![{chapter_title}](charts/{img_filename})\n")
                        except Exception as e:
                            print(f"[ERROR] Error processing image path: {str(e)}")
                            continue

            if hasattr(chapter, 'summary') and chapter.summary:
                markdown.append("\n### Chapter Summary\n")
                markdown.append(chapter.summary + "\n")

        if hasattr(node.report, 'brief_conclusion') and node.report.brief_conclusion:
            markdown.append("\n## Summary and Recommendations\n")
            markdown.append(node.report.brief_conclusion + "\n")

        return "\n".join(markdown)

    def _generate_html_report(self, markdown_content: str, output_dir: str) -> str:
        """
        This method is no longer used, kept for historical reference
        """
        return ""
