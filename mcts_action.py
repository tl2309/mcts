import copy
import json
import re,os,traceback
from typing import Dict, List, Any, Optional
from storyteller.algorithm.utils.DatasetContextGenerator import DatasetContextGenerator  # Import dataset parser
from storyteller.algorithm.mcts_node import *
from storyteller.llm_call.openai_llm import call_openai
from storyteller.llm_call.prompt_factory import get_prompt
from typing import List, Dict, Any
import pandas as pd
from enum import Enum
import base64
from PIL import Image
import io,requests
from openai import OpenAI
from .utils.html2image import convert_html_file_to_image
from storyteller.algorithm.mcts_node import ReportGenerationState
from llmx import llm, TextGenerator
from lida.components.manager import Manager
from storyteller.algorithm.utils.universalsc import run_universal_self_consistency  # Import universalsc functionality
from storyteller.algorithm.utils.unified_framework import unified_generation_framework  # Import unified framework
import time
from tqdm import tqdm
import glob
import random




class DataStorytellingAction:
    def __init__(self, action_id: str, description: str):
        self.action_id = action_id
        self.description = description

        # Add MCTS statistics attributes
        #self.Q = 0.0  # Cumulative reward
        #self.N = 0    # Visit count


    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1

            raise NotImplementedError


class Query2Chapters(DataStorytellingAction):
    def __init__(self):
        super().__init__("A1", "Define chapter structure")
        self.use_unified_framework = True  # Whether to use unified framework

    def generate_chapter_prompt(self, node, **kwargs):
        """Generate chapter prompt"""
        query = node.report.clarified_query if node.report.clarified_query else node.report.original_query
        data_context = node.report.data_context

        # Use preset prompt template
        prompt_args = {
            "QUERY": query,
            "DATA_CONTEXT": data_context
        }

        return get_prompt("Query2Chapters_test", prompt_args)

    def apply_chapters(self, node, action, cluster, **kwargs):
        """Apply chapters to child node"""
        try:
            cluster_id = cluster.get("cluster_id", "unknown")
            chapters = cluster.get("chapters", [])

            if not chapters:
                print(f"[WARNING] Cluster {cluster_id} has no chapter content, skipping")
                return None

            print(f"[INFO] Applying chapter scheme for cluster {cluster_id}")
            print(f"   Chapter structure: {chapters}")

            # Create child node
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = action
            child_node.depth = node.depth + 1

            # Clear existing chapters
            child_node.report.chapters = []

            # Add chapters
            for title in chapters:
                child_node.report.add_chapter(Chapter(title=title))

            # Set node state
            child_node.node_type = ReportGenerationState.a1

            print(f"[SUCCESS] Successfully added cluster {cluster_id} chapter scheme")
            return [child_node]

        except Exception as e:
            print(f"[ERROR] Error applying chapters: {str(e)}")
            traceback.print_exc()
            return None

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """
        Generate diverse chapter structures using unified framework based on user query and data context
        """
        if self.use_unified_framework:
            return unified_generation_framework(
                node=node,
                action=self,
                llm_kwargs=llm_kwargs,
                action_type="chapters",
                prompt_generator=self.generate_chapter_prompt,
                node_applier=self.apply_chapters,
                n=4
            )
        else:
            query = node.report.clarified_query if node.report.clarified_query else node.report.original_query
            data_context = node.report.data_context
            print(data_context)
            clusters = run_universal_self_consistency(query, data_context, llm_kwargs, n=4)
            print(f"[SUCCESS] Completed chapter clustering, got {len(clusters)} clusters")

            nodes = []
            for cluster in clusters:
                cluster_id = cluster.get("cluster_id", "unknown")
                chapters = cluster.get("chapters", [])

                if not chapters:
                    print(f"[WARNING] Cluster {cluster_id} has no chapter content, skipping")
                    continue

                print(f"[INFO] Creating child node for cluster {cluster_id}")
                print(f"   Chapter structure: {chapters}")

                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1

                child_node.report.chapters = []

                for title in chapters:
                        child_node.report.add_chapter(Chapter(title=title))

                child_node.node_type = ReportGenerationState.a1

                nodes.append(child_node)
                print(f"[SUCCESS] Successfully added cluster {cluster_id} chapter scheme")

        return nodes



class Chapters2Tasks(DataStorytellingAction):
    def __init__(self):
        super().__init__("A2", "Divide chapter task scheme according to chapter scheme")
        self.use_unified_framework = True  # Whether to use unified framework

    def generate_tasks_prompt(self, node, **kwargs):
        """Generate task prompt"""
        data_context = node.report.data_context
        query = node.report.clarified_query if node.report.clarified_query else node.report.original_query

        chapters_list = []
        for i, chapter in enumerate(node.report.chapters):
            if isinstance(chapter, dict):
                if 'title' in chapter:
                    if isinstance(chapter['title'], dict):
                        title_text = chapter['title'].get('title', '') or chapter['title'].get('text', f"Chapter{i+1}")
                    else:
                        title_text = chapter['title']
                else:
                    title_text = f"Chapter{i+1}"
            else:
                title_attr = getattr(chapter, 'title', None)
                if isinstance(title_attr, dict):
                    title_text = title_attr.get('title', '') or title_attr.get('text', f"Chapter{i+1}")
                else:
                    title_text = title_attr if title_attr else f"Chapter{i+1}"

            chapters_list.append(title_text)

        prompt_args = {
            "QUERY": query,
            "DATA_CONTEXT": data_context,
            "CHAPTERS": json.dumps(chapters_list, ensure_ascii=False)
        }

        return get_prompt("Chapters2Tasks_test", prompt_args)

    def apply_tasks(self, node, action, cluster, **kwargs):
        """Apply tasks to child node"""
        try:
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = action
            child_node.depth = node.depth + 1

            cluster_id = cluster.get("cluster_id", "unknown")
            chapters_info = cluster.get("chapters", [])

            if not chapters_info:
                print(f"[WARNING] Cluster {cluster_id} has no task content, skipping")
                return None

            print(f"[INFO] Applying task scheme for cluster {cluster_id}")

            chapter_title_to_index = {}
            for i, chapter in enumerate(child_node.report.chapters):
                if isinstance(chapter, dict):
                    if 'title' in chapter:
                        if isinstance(chapter['title'], dict):
                            title_text = chapter['title'].get('title', '') or chapter['title'].get('text', f"Chapter{i+1}")
                        else:
                            title_text = chapter['title']
                    else:
                        title_text = f"Chapter{i+1}"
                else:
                    title_attr = getattr(chapter, 'title', None)
                    if isinstance(title_attr, dict):
                        title_text = title_attr.get('title', '') or title_attr.get('text', f"Chapter{i+1}")
                    else:
                        title_text = title_attr if title_attr else f"Chapter{i+1}"

                if not isinstance(title_text, str):
                    title_text = str(title_text)

                chapter_title_to_index[title_text.lower()] = i

            chapters_with_tasks = set()

            for chapter_info in chapters_info:
                raw_title = chapter_info.get("title", "")

                if isinstance(raw_title, dict):
                    title_text = raw_title.get('title', '') or raw_title.get('text', '')
                else:
                    title_text = raw_title

                if not isinstance(title_text, str):
                    title_text = str(title_text) if title_text is not None else ""

                tasks = chapter_info.get("tasks", [])

                chapter_idx = -1
                title_lower = title_text.lower()

                if title_lower in chapter_title_to_index:
                    chapter_idx = chapter_title_to_index[title_lower]
                else:
                    for i, chapter in enumerate(child_node.report.chapters):
                        if isinstance(chapter, dict):
                            if 'title' in chapter:
                                if isinstance(chapter['title'], dict):
                                    search_title = chapter['title'].get('title', '') or chapter['title'].get('text', f"Chapter{i+1}")
                                else:
                                    search_title = chapter['title']
                            else:
                                search_title = f"Chapter{i+1}"
                        else:
                            title_attr = getattr(chapter, 'title', None)
                            if isinstance(title_attr, dict):
                                search_title = title_attr.get('title', '') or title_attr.get('text', f"Chapter{i+1}")
                            else:
                                search_title = title_attr if title_attr else f"Chapter{i+1}"

                        if not isinstance(search_title, str):
                            search_title = str(search_title)

                        search_title_lower = search_title.lower()
                        if title_lower in search_title_lower or search_title_lower in title_lower:
                            chapter_idx = i
                            break
                
                if chapter_idx >= 0 and chapter_idx < len(child_node.report.chapters):
                    chapter = child_node.report.chapters[chapter_idx]

                    # Clear existing task list
                    chapter.visualization_tasks = []

                    # Add tasks
                    for task in tasks:
                        task_id = task.get("task_id", "")
                        description = task.get("task_description", "")
                        chart_type = task.get("chart_type", ["Bar Chart"])

                        # Create task object
                        task_obj = {
                            "task_id": task_id,
                            "task_description": description,
                            "chart_type": chart_type,
                            "status": "pending",
                            "visualization_success": False
                        }

                        # Add to chapter's task list
                        if not hasattr(chapter, 'visualization_tasks'):
                            chapter.visualization_tasks = []
                        chapter.visualization_tasks.append(task_obj)

                        # Print task status
                        print(f"   - Task ID: '{task_id}'")
                        print(f"   - Task description: '{description}'")
                        print(f"   - Chart type: {chart_type}")
                        print(f"   - Status: {task_obj.get('status')}")

                    # Record chapters with assigned tasks
                    chapters_with_tasks.add(chapter_idx)

                    # Print debug info
                    print(f"[SUCCESS] Generated {len(tasks)} visualization tasks for chapter {chapter_idx+1} ({chapter.title})")
                else:
                    print(f"[ERROR] Cannot find matching chapter: {title_text}")

            # Check if all chapters have tasks
            all_chapters_have_tasks = True
            for i, chapter in enumerate(child_node.report.chapters):
                if i not in chapters_with_tasks:
                    print(f"[WARNING] Chapter {i+1} ({chapter.title}) has no tasks")
                    all_chapters_have_tasks = False

            # Only return node if all chapters have tasks
            if all_chapters_have_tasks:
                # Set node state
                child_node.node_type = ReportGenerationState.a2
                print(f"[SUCCESS] Successfully applied task scheme for cluster {cluster_id}")
                return [child_node]
            else:
                print(f"[WARNING] Task scheme for cluster {cluster_id} is incomplete, skipping")
                return None

        except Exception as e:
            print(f"[ERROR] Error applying tasks: {str(e)}")
            traceback.print_exc()
            return None

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """Generate multiple task schemes for each chapter"""
        # Only use unified framework implementation
        return unified_generation_framework(
            node=node,
            action=self,
            llm_kwargs=llm_kwargs,
            action_type="tasks",
            prompt_generator=self.generate_tasks_prompt,
            node_applier=self.apply_tasks,
            n=3  # Generate 3 different task scheme variants
        )



class Tasks2Charts(DataStorytellingAction):
    def __init__(self):
        super().__init__("A3", "Generate visualizations")
        # Save config parameters instead of actual object instances
        self.similarity_threshold = 0.90  # Similarity threshold
        self.use_similarity_check = True  # Flag whether to use similarity check
        self.use_chart2vega = True  # Flag whether to use chart2vega

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        child_node = copy.deepcopy(node)
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1

        try:
            # Initialize chart similarity detection tool (deferred until needed)
            similarity_tool = None
            if self.use_similarity_check:
                try:
                    from storyteller.algorithm.utils.ChartSimilarity import ChartSimilarity
                    similarity_tool = ChartSimilarity()
                    print("[SUCCESS] Chart similarity detection tool initialized")
                except Exception as e:
                    print(f"[WARNING] Chart similarity detection tool initialization failed: {str(e)}")
                    similarity_tool = None

            # Initialize chart2vega (deferred until needed)
            chart2vega_module = None
            if self.use_chart2vega:
                try:
                    from storyteller.algorithm.utils import chart2vega
                    chart2vega_module = chart2vega
                    print("[SUCCESS] chart2vega tool initialized")
                except Exception as e:
                    print(f"[WARNING] chart2vega tool initialization failed: {str(e)}")
                    chart2vega_module = None

            # Increment iteration number - ensure iteration number increases by 1 each time a new node is created
            current_iteration = child_node.report.current_iteration
            print(f"[INFO] Current iteration: {current_iteration}")

            # Determine current iteration and save path
            iteration_dir = os.path.join("storyteller", "output", "iterations", f"iteration_{current_iteration}")
            os.makedirs(iteration_dir, exist_ok=True)
            charts_dir = os.path.join(iteration_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)

            # Create Vega-Lite config directory
            vegalite_dir = os.path.join(iteration_dir, "vegalite_configs")
            os.makedirs(vegalite_dir, exist_ok=True)

            # Get dataset
            dataset_path = node.report.dataset_path
            df = pd.read_csv(dataset_path)

            # Iterate through all chapters
            for chapter_idx, chapter in enumerate(child_node.report.chapters):
                print(f"\n[INFO] Processing chapter {chapter_idx + 1}...")
                print(f"Chapter title: {getattr(chapter, 'title', f'Chapter{chapter_idx+1}')}")
                print(f"Number of visualization tasks in chapter: {len(getattr(chapter, 'visualization_tasks', []))}")

                # Initialize chapter chart list (if not exists)
                if not hasattr(chapter, 'charts'):
                    chapter.charts = []

                # Collect all chapter charts for similarity checking
                all_charts = []
                for ch in child_node.report.chapters:
                    if hasattr(ch, 'charts'):
                        all_charts.extend(ch.charts)

                # Iterate through all visualization tasks in chapter
                for task in chapter.visualization_tasks:
                    print(f"\n[INFO] Processing task:")
                    print(f"- Task ID: {task.get('task_id', '')}")
                    print(f"- Task description: {task.get('task_description', '')}")
                    print(f"- Chart type: {task.get('chart_type', ['Bar Chart'])[0]}")

                    task_id = task.get('task_id', "")
                    description = task.get('task_description')
                    chart_type = task.get('chart_type', ["Bar Chart"])[0]

                    # Use task ID as file name, use task description if empty
                    file_name = task_id if task_id else description
                    if not file_name:
                        file_name = f"chart_{chapter_idx}_{len(chapter.charts)}"
                    # Clean illegal characters from file name
                    file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
                    chart_path = os.path.join(charts_dir, f"{file_name}.png")

                    # Create text generator and manager (create in local scope to avoid serialization issues)
                    from lida.datamodel import Goal, Summary
                    from lida.components.manager import Manager

                    # Create Goal object
                    goal = Goal(question=task_id, visualization=description, chart_type=chart_type)

                    # Create Summary object
                    # Read data summary JSON file
                    data_summary = {}
                    json_path = os.path.join("storyteller", "dataset", "data_context.json")
                    print(f"Attempting to read data summary JSON: {json_path}")
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data_summary = json.load(f)
                        print("[SUCCESS] Successfully read data summary JSON")
                    except Exception as e:
                        print(f"[ERROR] Failed to read data summary JSON: {str(e)}")
                        # Use default values if JSON file cannot be read
                        data_summary = {
                            "name": node.report.original_query,
                            "dataset_description": node.report.data_context,
                            "fields_info": {col: {"dtype": str(dtype)} for col, dtype in zip(df.columns, df.dtypes)}
                        }

                    # Create Summary object using information extracted from JSON file
                    summary = Summary(
                        name=data_summary.get("name", "Shopping Data Analysis"),
                        file_name=dataset_path,
                        dataset_description=str(data_summary.get("dataset_description", "Shopping Dataset")),
                        field_names=list(data_summary.get("fields_info", {}).keys()) if "fields_info" in data_summary else df.columns.tolist(),
                        fields=[info.get("dtype", "unknown") for info in data_summary.get("fields_info", {}).values()] if "fields_info" in data_summary else [str(dtype) for dtype in df.dtypes.tolist()]
                    )

                    # Create custom text generator
                    text_gen = llm(
                        provider="openai",
                        model="gpt-4-32k"
                    )

                    # Create LIDA manager
                    manager = Manager(text_gen=text_gen)

                    # Generate visualization
                    print(f"Generating visualization chart for task '{description}'...")
                    visualization = manager.visualize(summary, goal, library="matplotlib")

                    # Process visualization result
                    if isinstance(visualization, list) and len(visualization) > 0:
                        visualization = visualization[0]

                    if hasattr(visualization, 'status') and visualization.status:
                        print("[SUCCESS] Successfully generated visualization")

                        # Save chart
                        if hasattr(visualization, 'savefig'):
                            visualization.savefig(chart_path)
                            print(f"[SUCCESS] Chart saved to: {chart_path}")

                            # Generate Vega-Lite config
                            try:
                                # Use chart2vega to extract Vega-Lite config
                                chart_config = self._extract_chart_config(visualization, task_id, description, df, llm_kwargs, chart2vega_module)

                                # Save Vega-Lite config
                                if "vegalite_config" in chart_config and chart_config["vegalite_config"]:
                                    vegalite_config = chart_config["vegalite_config"]
                                    vegalite_file_name = f"{file_name}.json"
                                    vegalite_path = os.path.join(vegalite_dir, vegalite_file_name)

                                    with open(vegalite_path, "w", encoding="utf-8") as f:
                                        json.dump(vegalite_config, f, ensure_ascii=False, indent=2)
                                    print(f"[SUCCESS] Vega-Lite chart config saved to: {vegalite_path}")

                                    # Generate Vega-Lite HTML visualization
                                    try:
                                        if chart2vega_module:
                                            # Create HTML output directory
                                            html_dir = os.path.join(iteration_dir, "vegalite_html")
                                            os.makedirs(html_dir, exist_ok=True)

                                            # Generate HTML file
                                            html_path = os.path.join(html_dir, f"{file_name}.html")

                                            # Create HTML viewer
                                            chart2vega_module.create_html_viewer(vegalite_config, html_path)
                                            print(f"[SUCCESS] Vega-Lite HTML visualization saved to: {html_path}")
                                    except Exception as e:
                                        print(f"[WARNING] Error generating Vega-Lite HTML: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
                            except Exception as e:
                                print(f"[WARNING] Error generating Vega-Lite config: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                vegalite_path = None

                            # Extra save chart data as CSV for subsequent analysis
                            try:
                                csv_dir = os.path.join(os.path.dirname(charts_dir), "chart_data")
                                os.makedirs(csv_dir, exist_ok=True)
                                csv_file_name = f"{file_name}.csv"
                                csv_path = os.path.join(csv_dir, csv_file_name)

                                # Try to extract actual data used from visualization object
                                if hasattr(visualization, '_data') and isinstance(visualization._data, pd.DataFrame):
                                    visualization._data.to_csv(csv_path, index=False)
                                    print(f"[SUCCESS] Chart data saved to: {csv_path}")
                                elif hasattr(visualization, 'data') and isinstance(visualization.data, pd.DataFrame):
                                    visualization.data.to_csv(csv_path, index=False)
                                    print(f"[SUCCESS] Chart data saved to: {csv_path}")
                            except Exception as e:
                                print(f"[WARNING] Error saving chart data CSV: {str(e)}")
                                traceback.print_exc()

                            # Check chart similarity
                            if similarity_tool and all_charts:
                                # Collect existing chart paths
                                existing_chart_paths = []
                                for chart in all_charts:
                                    if hasattr(chart, 'url') and chart.url:
                                        existing_chart_paths.append(chart.url)

                                if existing_chart_paths:
                                    # Use batch_compare to calculate similarity
                                    is_too_similar, max_similarity, similar_chart_path, all_similarities = similarity_tool.batch_compare(
                                        chart_path, existing_chart_paths, self.similarity_threshold
                                    )

                                    if is_too_similar:
                                        # Find most similar chart object
                                        similar_chart = None
                                        for chart in all_charts:
                                            if hasattr(chart, 'url') and chart.url == similar_chart_path:
                                                similar_chart = chart
                                                break

                                        similar_task_id = getattr(similar_chart, 'task_id', 'Unknown task') if similar_chart else 'Unknown task'

                                        print(f"[WARNING] Generated chart has too high similarity with existing chart ({max_similarity:.4f})")
                                        print(f"   - Similar chart: {similar_task_id}")

                                        # Create samechart folder
                                        samechart_dir = os.path.join(charts_dir, "samechart")
                                        os.makedirs(samechart_dir, exist_ok=True)

                                        # Move similar chart to samechart folder
                                        samechart_path = os.path.join(samechart_dir, f"{file_name}.png")

                                        try:
                                            import shutil
                                            # Move chart to samechart directory (instead of copy)
                                            shutil.move(chart_path, samechart_path)
                                            print(f"[SUCCESS] Similar chart moved to: {samechart_path}")

                                            # Output similarity info to console
                                            print(f"[INFO] Chart similarity info:")
                                            print(f"   - Similarity value: {max_similarity:.4f}")
                                            print(f"   - Similar chart task: {similar_task_id}")
                                            print(f"   - Current task: {task_id}")

                                            # Mark task as completed but chart skipped
                                            for vis_task in chapter.visualization_tasks:
                                                if vis_task.get('task_id') == task_id:
                                                    vis_task['visualization_success'] = False
                                                    vis_task['skipped_due_to_similarity'] = True
                                                    print(f"[WARNING] Task '{task_id}' skipped due to high chart similarity")
                                                    break

                                            # Skip subsequent processing for current task
                                            continue

                                        except Exception as e:
                                            print(f"[WARNING] Error moving similar chart: {str(e)}")
                                            # If move fails, continue with original path

                        # Create chart object
                        print(f"\n[INFO] Creating chart object:")
                        print(f"- Chart path: {chart_path}")
                        print(f"- Chart type: {chart_type}")
                        print(f"- Task ID: {task_id}")

                        chart = Chart(
                            url=chart_path,
                            caption="",
                            chart_type=chart_type,
                            task_id=task_id
                        )

                        # Store visualization code for subsequent modification
                        if hasattr(visualization, 'code'):
                            chart.code = visualization.code

                        # Add chart to chapter
                        if not hasattr(chapter, 'charts'):
                            chapter.charts = []
                            print("Initializing chapter's chart list")

                        chapter.charts.append(chart)
                        # Update all charts list
                        all_charts.append(chart)
                        print(f"[SUCCESS] Chart added to chapter, current chapter chart count: {len(chapter.charts)}")

                        # If processed successfully, also mark as completed
                        for vis_task in chapter.visualization_tasks:
                            if vis_task.get('task_id') == task_id:
                                vis_task['visualization_success'] = True
                                print(f"[SUCCESS] Task '{task_id}' completed successfully")
                                break
                    else:
                        print("[ERROR] Failed to generate visualization chart")
                        # Mark as completed even if failed to avoid infinite loop
                        for vis_task in chapter.visualization_tasks:
                            if vis_task.get('task_id') == task_id:
                                # Ensure visualization_success field is initialized
                                vis_task['visualization_success'] = False

                                # Save failed chart code if available
                                if hasattr(visualization, 'code'):
                                    # Create failed chart directory if not exists
                                    failed_code_dir = os.path.join(charts_dir, "failed_code")
                                    os.makedirs(failed_code_dir, exist_ok=True)

                                    # Save failed chart code to file
                                    code_file_path = os.path.join(failed_code_dir, f"{file_name}_failed.py")
                                    try:
                                        with open(code_file_path, 'w', encoding='utf-8') as f:
                                            f.write(visualization.code)
                                        print(f"[SUCCESS] Saved failed chart code to: {code_file_path}")

                                        # Record code path in task
                                        vis_task['failed_code_path'] = code_file_path

                                        # Create placeholder chart object and add to chapter even if chart generation failed
                                        # Use temporary placeholder image path or special marker to indicate failed chart
                                        placeholder_chart = Chart(
                                            url=code_file_path,
                                            caption="",
                                            chart_type=chart_type,
                                            task_id=task_id
                                        )

                                        # Add code and failure marker
                                        placeholder_chart.code = visualization.code
                                        placeholder_chart.generation_failed = True

                                        # Add chart to chapter
                                        if not hasattr(chapter, 'charts'):
                                            chapter.charts = []

                                        chapter.charts.append(placeholder_chart)
                                        all_charts.append(placeholder_chart)
                                        print(f"[SUCCESS] Added failed chart placeholder to chapter for subsequent fixing")

                                    except Exception as e:
                                        print(f"[ERROR] Error saving failed chart code or creating placeholder chart: {str(e)}")

                                print(f"[WARNING] Task '{description}' marked as completed despite failure to avoid infinite loop")
                                break
            # Set correct state
            child_node.node_type = ReportGenerationState.a3
            return [child_node]

        except Exception as e:
            print(f"[ERROR] Error processing node: {str(e)}")
            traceback.print_exc()
            # Ensure correct state is set even if exception occurs
            child_node.node_type = ReportGenerationState.a3
            return [child_node]


    def _extract_chart_config(self, visualization, task_id, description, df, llm_kwargs=None, chart2vega_module=None):
        """Extract chart config from visualization code, convert to Vega-Lite using chart2vega

        Args:
            visualization: Object containing visualization code
            task_id: Task ID
            description: Task description
            df: Data DataFrame
            llm_kwargs: LLM call parameters
            chart2vega_module: chart2vega module instance

        Returns:
            Config dict containing vegalite_config
        """
        # Initialize empty config
        result_config = {
            "title": description or "Chart",
            "vegalite_config": None
        }

        try:
            # Ensure visualization code exists
            if not hasattr(visualization, 'code'):
                raise ValueError("Visualization object does not have code attribute")

            code = visualization.code
            print("\n[INFO] Analyzing visualization code:")
            print("-" * 50)
            print(code)
            print("-" * 50)

            # Use chart2vega to directly convert Python code to Vega-Lite config
            if chart2vega_module:
                try:
                    print("\n[INFO] Using chart2vega tool to generate Vega-Lite config...")

                    # Ensure llm_kwargs parameter is correctly passed
                    if llm_kwargs is None:
                        llm_kwargs = {}
                    else:
                        # Create a copy to avoid modifying original object
                        llm_kwargs = llm_kwargs.copy()

                    # Add or ensure suitable model is set
                    if not llm_kwargs.get("model"):
                        llm_kwargs["model"] = "gpt-4-turbo"

                    # Ensure API call parameters are correct
                    # Check if base_url exists, if not try to get from environment variable
                    if not llm_kwargs.get("base_url"):
                        env_base_url = os.environ.get("OPENAI_BASE_URL")
                        if env_base_url:
                            llm_kwargs["base_url"] = env_base_url

                    # Check if api_key exists, if not try to get from environment variable
                    if not llm_kwargs.get("api_key"):
                        env_api_key = os.environ.get("OPENAI_API_KEY")
                        if env_api_key:
                            llm_kwargs["api_key"] = env_api_key

                    # Add retry logic
                    max_retries = 2
                    vegalite_config = None

                    for retry in range(max_retries):
                        try:
                            if retry > 0:
                                print(f"[WARNING] Retry {retry+1} calling chart2vega...")

                            vegalite_config = chart2vega_module.convert_python_to_vegalite(code, llm_kwargs=llm_kwargs)

                            if vegalite_config:
                                print("[SUCCESS] Successfully converted code to Vega-Lite config using LLM")
                                break
                            else:
                                print(f"[WARNING] Retry {retry+1} failed")
                        except Exception as e:
                            print(f"[WARNING] Error on retry {retry+1}: {str(e)}")

                            if retry < max_retries - 1:
                                print("[WARNING] Retrying in a moment...")
                                time.sleep(1)  # Brief delay before retry

                    # Check if Vega-Lite config was successfully obtained
                    if vegalite_config:
                        # Ensure title is set
                        if isinstance(vegalite_config, dict) and (not vegalite_config.get("title") or vegalite_config["title"] == "Chart"):
                            vegalite_config["title"] = description

                        # Save vegalite_config to result
                        result_config["vegalite_config"] = vegalite_config

                        # Output config info
                        print(f"\n[SUCCESS] Successfully generated Vega-Lite config:")
                        if isinstance(vegalite_config.get("mark"), dict):
                            print(f"- Chart type: {vegalite_config.get('mark', {}).get('type', '')}")
                        else:
                            print(f"- Chart type: {vegalite_config.get('mark', '')}")
                        print(f"- Chart title: {vegalite_config.get('title', '')}")

                        if 'encoding' in vegalite_config:
                            encoding = vegalite_config.get('encoding', {})
                            print(f"- X-axis field: {encoding.get('x', {}).get('field', '')}")
                            print(f"- Y-axis field: {encoding.get('y', {}).get('field', '')}")
                    else:
                        print("[WARNING] LLM conversion of Vega-Lite config failed")

                except Exception as e:
                    print(f"[WARNING] Error using chart2vega: {str(e)}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[WARNING] Error extracting chart config: {str(e)}")
            import traceback
            traceback.print_exc()

        return result_config

class ReviseVis(DataStorytellingAction):
    def __init__(self):
        super().__init__("A4", "Revise all visualization charts")
    
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """Modify visualization charts"""
        # Create child node
        child_node = copy.deepcopy(node)
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1

        try:
            # Iterate through all chapters
            for chapter_idx, chapter in enumerate(child_node.report.chapters):
                # Check if chapter has visualization tasks
                if not hasattr(chapter, 'visualization_tasks') or not chapter.visualization_tasks:
                    print(f"[WARNING] Chapter {chapter_idx + 1} has no visualization tasks, skipping")
                    continue

                for task in chapter.visualization_tasks:
                    # Check if task was successfully generated or skipped due to similarity
                    if task.get('visualization_success', False) == True:
                        continue

                    # Check if task was skipped due to high similarity, if so no fixing needed
                    if task.get('skipped_due_to_similarity', False) == True:
                        print(f"[WARNING] Task '{task.get('task_id', '')}' was skipped due to high similarity, no fixing needed")
                        continue

                    task_id = task.get('task_id', "")
                    description = task.get('task_description', "")

                    print(f"Modifying chart for task '{task_id}'...")

                    selected_chart = None
                    print(f"\n[INFO] Searching for chart in chapter:")
                    print(f"- Chapter title: {getattr(chapter, 'title', f'Chapter{chapter_idx+1}')}")
                    print(f"- Number of charts in chapter: {len(getattr(chapter, 'charts', []))}")

                    for c in chapter.charts:
                        print(f"- Checking chart: task_id={getattr(c, 'task_id', 'None')}")
                        if hasattr(c, 'task_id') and c.task_id == task_id:
                            selected_chart = c
                            print("[SUCCESS] Found matching chart")
                            break

                    # If no matching chart found, skip this task
                    if not selected_chart:
                        print(f"[WARNING] Cannot find chart matching task '{task_id}', skipping")
                        continue

                    try:
                        # Get data file path
                        dataset_path = node.report.dataset_path

                        # Read data
                        df = pd.read_csv(dataset_path)

                        # Create LIDA manager and text generator within the method (local variables)
                        from lida.components.manager import Manager
                        from lida.datamodel import Summary

                        # Create custom text generator (as local variable)
                        text_gen = llm(provider="openai", model="gpt-4o")
                        manager = Manager(text_gen=text_gen)

                        # Read data summary JSON file
                        data_summary = {}
                        json_path = os.path.join("storyteller", "dataset", "data_context.json")
                        print(f"Attempting to read data summary JSON: {json_path}")

                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data_summary = json.load(f)
                            print("[SUCCESS] Successfully read data summary JSON")
                        except Exception as e:
                            print(f"[ERROR] Failed to read data summary JSON: {str(e)}")
                            # Use default values if JSON file cannot be read
                            data_summary = {
                                "name": node.report.original_query,
                                "dataset_description": node.report.data_context,
                                "fields_info": {col: {"dtype": str(dtype)} for col, dtype in zip(df.columns, df.dtypes)}
                            }

                        # Create Summary object, extract necessary parameters directly from JSON file
                        summary = Summary(
                            name=data_summary.get("name", "Data Analysis"),
                            file_name=dataset_path,
                            dataset_description=str(data_summary.get("dataset_description", "Shopping Dataset")),
                            field_names=list(data_summary.get("fields_info", {}).keys()) if "fields_info" in data_summary else df.columns.tolist(),
                            fields=[info.get("dtype", "unknown") for info in data_summary.get("fields_info", {}).values()] if "fields_info" in data_summary else [str(dtype) for dtype in df.dtypes.tolist()]
                        )

                        # Check task description and chart code to decide whether to generate table instead of chart
                        chart_generation_failed = getattr(selected_chart, 'generation_failed', False)

                        # Determine whether to convert chart to table
                        # If chart generation failed, or there are other markers indicating table should be used
                        if chart_generation_failed:
                            print(f"[INFO] Detected chart generation failure, attempting to generate table to display data")
                            edit_instruction = f"""
                            Please convert this failed visualization code to generate a table. Original task description: '{description}'

                            Please follow these guidelines:
                            1. Carefully analyze the original task description to ensure the table can display the same data relationships and comparisons as the original task
                                - Analyze the variable relationships the task wants to display (for example, if comparing two groups of data, the table should contain comparisons of these two groups)
                                - Clarify the X-axis and Y-axis variables in the task, and ensure these variables have explicit columns in the table
                                - Preserve the aggregation methods required in the task (average, sum, count, etc.)

                            2. Data processing part:
                                - Preserve key data filtering, grouping, and aggregation operations from the original code
                                - If the task requires comparing multiple categories or groups, ensure all categories are in the table

                            3. Table design:
                                - Create clear row and column labels for the table, consistent with the X-axis/Y-axis naming in the original task
                                - Limit the number of data rows in the table (display at most 15 rows of key data)
                                - Format numbers appropriately (for example, keep 2 decimal places)
                                - If the original task is comparing different categories, you can add a percentage difference column

                            4. Table style:
                                - Use matplotlib's plt.table() to create the table
                                - Adjust table colors and styles to improve readability and aesthetics
                                - Set appropriate cell colors based on data type (for example, use color shades to indicate numeric magnitude)

                            5. Metadata:
                                - Use the title from the original task, and note in the title that this is a table format

                            The main goal is to ensure the table format can fully present the data insights and relationships that the original visualization task wanted to convey.
                            The final output should be a matplotlib image that can be directly saved as PNG.
                            """
                        else:
                            # If not generating table, use regular chart modification instructions
                            edit_instruction = "Modify chart errors, such as changing to a more suitable chart type, making the chart more aesthetically pleasing and clear"

                        # Use LIDA's edit function to modify chart/generate table
                        print(f"Generating {'table' if chart_generation_failed else 'chart'} for task '{description}'...")
                        edited_visualization = manager.edit(
                            code=selected_chart.code,
                            summary=summary,
                            instructions=edit_instruction,
                            library="matplotlib"
                        )

                        # Process edited visualization result
                        if edited_visualization is None:
                            print(f"[ERROR] Failed to generate {'table' if chart_generation_failed else 'chart'}: returned None")
                        elif isinstance(edited_visualization, list) and len(edited_visualization) > 0:
                            edited_visualization = edited_visualization[0]
                            print(f"[INFO] Using first edit result for processing")

                        # Check if it's a valid edit result
                        if hasattr(edited_visualization, 'status') and edited_visualization.status:
                            print(f"[SUCCESS] Successfully generated {'table' if chart_generation_failed else 'chart'}")

                            # Find the iteration directory where the current chart is located
                            original_chart_path = selected_chart.url
                            chart_dir = os.path.dirname(original_chart_path)

                            # Save the modified chart to the same directory
                            suffix = "_table" if chart_generation_failed else "_edited"
                            edited_chart_name = f"{task_id}{suffix}.png"
                            edited_chart_path = os.path.join(chart_dir, edited_chart_name)
                            
                            # Save modified chart or table
                            if hasattr(edited_visualization, 'savefig'):
                                edited_visualization.savefig(edited_chart_path)
                                print(f"[SUCCESS] {'Table' if chart_generation_failed else 'Chart'} saved to: {edited_chart_path}")

                                # Generate Vega-Lite config (only for charts, skip for tables)
                                if not chart_generation_failed:
                                    try:
                                        # Directly use the extract config logic instead of instantiating Tasks2Charts
                                        chart_config = self._extract_chart_config(edited_visualization, task_id, description, df, llm_kwargs)

                                        # Save Vega-Lite config
                                        if "vegalite_config" in chart_config and chart_config["vegalite_config"]:
                                            vegalite_config = chart_config["vegalite_config"]

                                            # Get Vega-Lite config directory
                                            vegalite_dir = os.path.join(os.path.dirname(chart_dir), "vegalite_configs")
                                            os.makedirs(vegalite_dir, exist_ok=True)

                                            # Save Vega-Lite config
                                            vegalite_file_name = f"{task_id}_edited.json"
                                            vegalite_path = os.path.join(vegalite_dir, vegalite_file_name)

                                            with open(vegalite_path, "w", encoding="utf-8") as f:
                                                json.dump(vegalite_config, f, ensure_ascii=False, indent=2)
                                            print(f"[SUCCESS] Vega-Lite chart config saved to: {vegalite_path}")

                                            # Generate HTML viewer
                                            try:
                                                # Import chart2vega (local import)
                                                from storyteller.algorithm.utils import chart2vega

                                                # Create HTML output directory
                                                html_dir = os.path.join(os.path.dirname(chart_dir), "vegalite_html")
                                                os.makedirs(html_dir, exist_ok=True)

                                                # Generate HTML file
                                                html_path = os.path.join(html_dir, f"{task_id}_edited.html")

                                                # Create HTML viewer
                                                chart2vega.create_html_viewer(vegalite_config, html_path)
                                                print(f"[SUCCESS] Vega-Lite HTML visualization saved to: {html_path}")
                                            except Exception as e:
                                                print(f"[WARNING] Error generating HTML viewer: {str(e)}")
                                                traceback.print_exc()
                                    except Exception as e:
                                        print(f"[WARNING] Error generating Vega-Lite config: {str(e)}")
                                        import traceback
                                        traceback.print_exc()

                                # Extra save chart data as CSV for subsequent analysis
                                try:
                                    csv_dir = os.path.join(os.path.dirname(chart_dir), "chart_data")
                                    os.makedirs(csv_dir, exist_ok=True)
                                    csv_file_name = f"{task_id}{suffix}.csv"
                                    csv_path = os.path.join(csv_dir, csv_file_name)

                                    # Try to extract actual data used from visualization object
                                    if hasattr(edited_visualization, '_data') and isinstance(edited_visualization._data, pd.DataFrame):
                                        edited_visualization._data.to_csv(csv_path, index=False)
                                        print(f"[SUCCESS] {'Table' if chart_generation_failed else 'Chart'} data saved to: {csv_path}")
                                    elif hasattr(edited_visualization, 'data') and isinstance(edited_visualization.data, pd.DataFrame):
                                        edited_visualization.data.to_csv(csv_path, index=False)
                                        print(f"[SUCCESS] {'Table' if chart_generation_failed else 'Chart'} data saved to: {csv_path}")
                                except Exception as e:
                                    print(f"[WARNING] Error saving {'table' if chart_generation_failed else 'chart'} data CSV: {str(e)}")
                                    traceback.print_exc()

                            # Create new chart object
                            from storyteller.algorithm.mcts_node import Chart
                            edited_chart = Chart(
                                url=edited_chart_path,
                                caption="",
                                chart_type="table" if chart_generation_failed else selected_chart.chart_type,
                                task_id=task_id
                            )
                            edited_chart.needs_caption = True
                            edited_chart.is_table = chart_generation_failed

                            # Update chart in chapter
                            for i, c in enumerate(chapter.charts):
                                if hasattr(c, 'task_id') and c.task_id == task_id:
                                    chapter.charts[i] = edited_chart
                                    # Update task status to success
                                    for vis_task in chapter.visualization_tasks:
                                        if vis_task.get('task_id') == task_id:
                                            vis_task['visualization_success'] = True
                                            vis_task['converted_to_table'] = chart_generation_failed
                                            print(f"[SUCCESS] Updated task '{task_id}' status to successfully generated {'table' if chart_generation_failed else 'chart'}")
                                            break
                                    break
                        else:
                            error_msg = edited_visualization.error if hasattr(edited_visualization, 'error') else "Unknown error"
                            print(f"[ERROR] Failed to generate {'table' if chart_generation_failed else 'chart'}: {error_msg}")
                    except Exception as e:
                        print(f"[ERROR] Error generating {'table' if getattr(selected_chart, 'generation_failed', False) else 'chart'} for task '{task_id}': {str(e)}")
                        import traceback
                        traceback.print_exc()

            # Set correct state
            child_node.node_type = ReportGenerationState.a4
            return [child_node]

        except Exception as e:
            print(f"[ERROR] Error processing node: {str(e)}")
            # If no tasks found, return empty list
            print("[ERROR] No tasks to process found")
            # Ensure correct state is set even if exception occurs
            child_node.node_type = ReportGenerationState.a4
            return [child_node]

    def _extract_chart_config(self, visualization, task_id, description, df, llm_kwargs=None):
        """Extract chart config from visualization code, convert to Vega-Lite

        Args:
            visualization: Object containing visualization code
            task_id: Task ID
            description: Task description
            df: Data DataFrame
            llm_kwargs: LLM call parameters

        Returns:
            Config dict containing vegalite_config
        """
        # Initialize empty config
        result_config = {
            "title": description or "Chart",
            "vegalite_config": None
        }

        try:
            # Ensure visualization code exists
            if not hasattr(visualization, 'code'):
                raise ValueError("Visualization object does not have code attribute")

            code = visualization.code
            print("\n[INFO] Analyzing visualization code:")
            print("-" * 50)
            print(code)
            print("-" * 50)

            # Import chart2vega (local import)
            try:
                from storyteller.algorithm.utils import chart2vega
                print("\n[INFO] Using chart2vega tool to generate Vega-Lite config...")

                # Ensure llm_kwargs parameter is correctly passed
                if llm_kwargs is None:
                    llm_kwargs = {}
                else:
                    # Create a copy to avoid modifying original object
                    llm_kwargs = llm_kwargs.copy()

                # Add or ensure suitable model is set
                if not llm_kwargs.get("model"):
                    llm_kwargs["model"] = "gpt-4-turbo"

                # Check if base_url exists, if not try to get from environment variable
                if not llm_kwargs.get("base_url"):
                    env_base_url = os.environ.get("OPENAI_BASE_URL")
                    if env_base_url:
                        llm_kwargs["base_url"] = env_base_url

                # Check if api_key exists, if not try to get from environment variable
                if not llm_kwargs.get("api_key"):
                    env_api_key = os.environ.get("OPENAI_API_KEY")
                    if env_api_key:
                        llm_kwargs["api_key"] = env_api_key

                # Add retry logic
                max_retries = 2
                vegalite_config = None

                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            print(f"[WARNING] Retry {retry+1} calling chart2vega...")

                        vegalite_config = chart2vega.convert_python_to_vegalite(code, llm_kwargs=llm_kwargs)

                        if vegalite_config:
                            print("[SUCCESS] Successfully converted code to Vega-Lite config using LLM")
                            break
                        else:
                            print(f"[WARNING] Retry {retry+1} failed")

                    except Exception as e:
                        print(f"[WARNING] Error on retry {retry+1}: {str(e)}")

                        if retry < max_retries - 1:
                            print("[WARNING] Retrying in a moment...")
                            time.sleep(1)  # Brief delay before retry

                # Check if Vega-Lite config was successfully obtained
                if vegalite_config:
                    # Ensure title is set
                    if isinstance(vegalite_config, dict) and (not vegalite_config.get("title") or vegalite_config["title"] == "Chart"):
                        vegalite_config["title"] = description

                    # Save vegalite_config to result
                    result_config["vegalite_config"] = vegalite_config

                    # Output config info
                    print(f"\n[SUCCESS] Successfully generated Vega-Lite config:")
                    if isinstance(vegalite_config.get("mark"), dict):
                        print(f"- Chart type: {vegalite_config.get('mark', {}).get('type', '')}")
                    else:
                        print(f"- Chart type: {vegalite_config.get('mark', '')}")
                    print(f"- Chart title: {vegalite_config.get('title', '')}")

                    if 'encoding' in vegalite_config:
                        encoding = vegalite_config.get('encoding', {})
                        print(f"- X-axis field: {encoding.get('x', {}).get('field', '')}")
                        print(f"- Y-axis field: {encoding.get('y', {}).get('field', '')}")
                else:
                    print("[WARNING] LLM conversion of Vega-Lite config failed")


            except Exception as e:
                print(f"[WARNING] Error using chart2vega: {str(e)}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"[WARNING] Error extracting chart config: {str(e)}")
            import traceback
            traceback.print_exc()

        return result_config

class Charts2Captions(DataStorytellingAction):
    def __init__(self):
        super().__init__("A5", "Generate captions for all visualization charts")

    def _filter_successful_charts(self, chapter):
        """Filter out successfully generated charts from chapter

        Args:
            chapter: Chapter object

        Returns:
            successful_charts: List of successfully generated charts
        """
        successful_charts = []

        # Check if chapter has charts
        if not hasattr(chapter, 'charts') or not chapter.charts:
            return successful_charts

        # Iterate through all charts in chapter
        for chart in chapter.charts:
            # Get chart task ID
            chart_task_id = getattr(chart, 'task_id', '')
            task_success = False
            
            # Check if chart already has caption
            has_caption = hasattr(chart, 'caption') and chart.caption
            
            # Find task status associated with chart from visualization tasks
            if hasattr(chapter, 'visualization_tasks'):
                for task in chapter.visualization_tasks:
                    if task.get('task_id') == chart_task_id:
                        task_success = task.get('visualization_success', False)
                        break
            
            # Only add successfully generated charts without caption
            if task_success and not has_caption:
                successful_charts.append(chart)
                print(f"[SUCCESS] Chart {chart_task_id} meets processing criteria")
            elif not task_success:
                print(f"[WARNING] Skipping chart {chart_task_id} because generation status is failed")
            elif has_caption:
                print(f"[INFO] Skipping chart {chart_task_id} because it already has caption")
                
        return successful_charts
    
    def _get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 encoding"""
        try:
            with Image.open(image_path) as img:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                # Convert to base64
                return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            print(f"[ERROR] Image conversion failed: {str(e)}")
            return None

    def call_vision_api(self, prompt, image_base64_list, **kwargs):
        """Unified vision API call handling, supports single or multiple images, auto handles rate limiting"""
        import os
        import requests
        import json
        import time
        import random
        
        # Get environment variables
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Log information
        print(f"[INFO] Environment variables: OPENAI_BASE_URL={base_url}, OPENAI_API_KEY={'set' if api_key else 'not set'}")
        
        # Construct complete API URL
        if base_url.endswith('/chat/completions'):
            url = base_url  # Already a complete URL
        elif base_url.endswith('/v1'):
            url = f"{base_url}/chat/completions"  # Add chat/completions endpoint
        else:
            # Ensure URL ends with slash
            if not base_url.endswith('/'):
                base_url += '/'
            url = f"{base_url}v1/chat/completions"  # Add v1/chat/completions path
            
        print(f"[INFO] Using API URL: {url}")
        
        # Set request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        
        # Prepare image content
        image_contents = []
        for img_base64 in (image_base64_list if isinstance(image_base64_list, list) else [image_base64_list]):
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a data visualization expert."},
            {"role": "user", "content": [{"type": "text", "text": prompt}, *image_contents]}
        ]
        
        # Set API call parameters
        model = "gpt-4-turbo"  # Use fixed model instead of getting from kwargs
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        print(f"[INFO] Calling vision API, model: {model}, temperature: {temperature}")
        
        # Configure retry parameters
        max_retries = kwargs.get("max_retries", 5)  # Increase max retry count
        base_delay = kwargs.get("base_delay", 3)   # Initial wait time (seconds)
        max_delay = kwargs.get("max_delay", 60)    # Max wait time (seconds)
        timeout = kwargs.get("timeout", 60)        # Request timeout time
        
        # Implement exponential backoff retry
        for retry in range(max_retries):
            try:
                # Create local session object instead of using global session
                session = requests.Session()
                
                # Send request
                response = session.post(url, headers=headers, json=data, timeout=timeout)
                response_json = response.json()
                
                # Close session
                session.close()
                
                # Process response
                if 'choices' in response_json and response_json['choices']:
                    return response_json['choices'][0]['message']['content'].strip()
                
                # Check if it's a rate limit error (429)
                if 'error' in response_json:
                    error = response_json['error']
                    error_code = error.get('code', '')
                    error_type = error.get('type', '')
                    error_message = error.get('message', '')
                    
                    # If it's a rate limit error, apply exponential backoff strategy
                    if error_code == '429' or '429' in error_message or 'rate limit' in error_message.lower():
                        # Print rate limit error
                        print(f"[ERROR] API returned error or no response: {response_json}")
                        
                        # Parse wait time (if provided by API)
                        wait_time = None
                        import re
                        time_matches = re.findall(r'retry after (\d+)', error_message.lower())
                        if time_matches and len(time_matches) > 0:
                            try:
                                wait_time = int(time_matches[0])
                            except ValueError:
                                pass
                        
                        # If no explicit wait time specified, use exponential backoff strategy
                        if wait_time is None:
                            # Calculate backoff time, add random jitter to avoid synchronized requests
                            delay = min(max_delay, base_delay * (2 ** retry)) + random.uniform(0, 2)
                        else:
                            # Use wait time returned by API plus 2 second buffer
                            delay = wait_time + 2
                            
                        if retry < max_retries - 1:  # No need to wait for last retry
                            print(f"[WARNING] API returned rate limit error, will retry in {delay:.1f} seconds... (attempt {retry+1}/{max_retries})")
                            time.sleep(delay)
                            continue
                    else:
                        print(f"[ERROR] API returned error: {error_type} - {error_message}")
                else:
                    print(f"[ERROR] API returned unknown format response: {response_json}")
                
            except Exception as e:
                # Handle network exceptions and other errors
                print(f"[ERROR] API call failed: {str(e)}")
                
                # Only wait if it's not the last retry
                if retry < max_retries - 1:
                    # Calculate backoff time
                    delay = min(max_delay, base_delay * (2 ** retry)) + random.uniform(0, 2)
                    print(f"[WARNING] Will retry in {delay:.1f} seconds... (attempt {retry+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                
                traceback.print_exc()
        
        print(f"[ERROR] Reached max retry count ({max_retries}), API call failed")
        return None
    
    def generate_chapter_caption_schemes(self, node, chapter, chapter_idx, charts, num_schemes=3, llm_kwargs=None):
        """Generate multiple caption schemes for all charts in a single chapter, with retry mechanism"""
        # Filter out successfully generated charts
        successful_charts = self._filter_successful_charts(chapter)
        
        # If there are no successfully generated charts in the chapter, return empty
        if not successful_charts:
            print(f"[WARNING] Chapter {chapter_idx+1} has no successfully generated charts to process")
            return []
        
        print(f"\n[INFO] Generating {num_schemes} caption schemes for chapter {chapter_idx+1}")
        print(f"Chapter title: {getattr(chapter, 'title', f'Chapter{chapter_idx+1}')}")
        print(f"Number of charts to process: {len(successful_charts)} (filtered from {len(charts)} total charts)")
        
        # Prepare chart info text and images
        charts_info = ""
        chart_images = []
        
        for i, chart in enumerate(successful_charts):
            charts_info += f"\nChart{i}:"
            charts_info += f"\n- Type: {chart.chart_type}"
            charts_info += f"\n- Task: {chart.task_id}"
            
            # Get chart image data
            image_base64 = self._get_image_base64(chart.url)
            if image_base64:
                chart_images.append(image_base64)
            else:
                print(f"[ERROR] Cannot get image data for chart {i}")
        
        if not chart_images:
            print("[ERROR] No available chart image data")
            return []
            
        # Implement retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Use template file to generate prompt
                prompt_args = {
                    "QUERY": node.original_query,
                    "CHAPTER_TITLE": getattr(chapter, 'title', f'Chapter{chapter_idx+1}'),
                    "DATA_CONTEXT": node.report.data_context,
                    "NUM_SCHEMES": str(num_schemes),
                    "CHARTS_INFO": charts_info,
                    "RETRY_NUM": str(retry + 1)  # Tell the model which attempt this is
                }
                
                # Enhance prompt
                prompt = get_prompt("chapter_captions", prompt_args)
                if retry > 0:
                    # For retries, add more explicit JSON format requirements
                    prompt += f"\n\n[IMPORTANT] This is attempt {retry+1}, please make sure to return valid JSON format. Your response must include complete JSON structure, formatted as follows:\n"
                    prompt += """
{
  "schemes": [
    {
      "scheme_id": 1,
      "captions": [
        {
          "chart_idx": 0,
          "caption": "Caption text for chart 0"
        },
        {
          "chart_idx": 1,
          "caption": "Caption text for chart 1"
        }
      ]
    },
    {
      "scheme_id": 2,
      "captions": [
        {
          "chart_idx": 0,
          "caption": "Alternative caption text for chart 0"
        },
        {
          "chart_idx": 1,
          "caption": "Alternative caption text for chart 1"
        }
      ]
    }
  ]
}
"""
                
                # Call vision API
                print(f"[INFO] Calling API to generate caption schemes for chapter {chapter_idx+1}... (attempt {retry+1}/{max_retries})")
                # Lower temperature for more deterministic results
                api_kwargs = llm_kwargs.copy() if llm_kwargs else {}
                api_kwargs['temperature'] = max(0.1, 0.7 - retry * 0.2)  # Gradually lower temperature
                response_text = self.call_vision_api(prompt, chart_images, **api_kwargs)
                
                if not response_text:
                    print(f"[ERROR] Chapter {chapter_idx+1} did not receive valid response (attempt {retry+1}/{max_retries})")
                    if retry < max_retries - 1:
                        print("Will retry in 1 second...")
                        import time
                        time.sleep(1)
                        continue
                    else:
                        return []
                
                # Parse JSON response
                print(f"[INFO] LLM response snippet: {response_text[:200]}...")
                result = self.extract_json_from_text(response_text)
                
                if result and "schemes" in result:
                    schemes = result["schemes"]
                    print(f"[SUCCESS] Successfully generated {len(schemes)} caption schemes for chapter {chapter_idx+1}")
                    return schemes
                
                print(f"[ERROR] Cannot parse chart caption schemes for chapter {chapter_idx+1} (attempt {retry+1}/{max_retries})")
                if retry < max_retries - 1:
                    print("Will retry in 1 second...")
                    import time
                    time.sleep(5)
                else:
                    print("Reached max retry count, cannot generate valid caption schemes")
                    return []
                    
            except Exception as e:
                print(f"[ERROR] Error generating chapter chart caption schemes: {str(e)} (attempt {retry+1}/{max_retries})")
                traceback.print_exc()
                if retry < max_retries - 1:
                    print("Will retry in 1 second...")
                    import time
                    time.sleep(5)
                else:
                    print("Reached max retry count, cannot generate valid caption schemes")
                    return []
        
        # All retries failed
        return []
    
    def extract_json_from_text(self, text):
        """Extract JSON from LLM response with strong error tolerance"""
        try:
            # First try to find JSON block
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON block parsing failed: {str(e)}, attempting to fix and re-parse")
                    # Try to fix common JSON format issues
                    json_str = self._fix_json(json_str)
                    return json.loads(json_str)
            
            # If no JSON block, try to find JSON object in entire text
            match = re.search(r'(\{[\s\S]*\})', text)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON object parsing failed: {str(e)}, attempting to fix and re-parse")
                    # Try to fix common JSON format issues
                    json_str = self._fix_json(json_str)
                    return json.loads(json_str)
            
            # If above methods fail, try to extract schemes section from text
            schemes_match = re.search(r'"schemes"\s*:\s*(\[[\s\S]*?\])', text)
            if schemes_match:
                schemes_str = schemes_match.group(1)
                print(f"[SUCCESS] Extracted schemes array, attempting to build complete JSON")
                try:
                    # Build a new JSON
                    new_json = f'{{"schemes": {schemes_str}}}'
                    return json.loads(new_json)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Extracted schemes parsing failed: {str(e)}")
            
            # If complete structure not found, try to manually extract each caption
            captions = re.findall(r'(?:chart_idx|chart_index)["\s:]+(\d+)[\s"]*(?:,|\})[\s\S]*?(?:caption|caption_text)["\s:]+([^"]*?)[",$}]', text)
            if captions:
                print(f"[SUCCESS] Manually extracted {len(captions)} caption entries")
                manual_scheme = {
                    "scheme_id": 1,
                    "captions": []
                }
                
                for chart_idx_str, caption in captions:
                    try:
                        chart_idx = int(chart_idx_str)
                        manual_scheme["captions"].append({
                            "chart_idx": chart_idx,
                            "caption": caption.strip()
                        })
                    except ValueError:
                        pass
                
                if manual_scheme["captions"]:
                    return {"schemes": [manual_scheme]}
            
            return None
        except Exception as e:
            print(f"[ERROR] JSON parsing error: {str(e)}")
            traceback.print_exc()
            return None
    
    def _fix_json(self, json_str):
        """Fix common JSON format issues"""
        # Fix missing comma issues
        json_str = re.sub(r'}\s*{', '},{', json_str)
        json_str = re.sub(r']\s*\[', '],[', json_str)
        
        # Fix extra commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ensure property names have quotes
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Fix escaping issues
        json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
        
        return json_str
    
    def generate_combined_nodes(self, node, all_chapter_schemes, all_chapter_groups=None, max_nodes=3):
        """Generate child node combinations - using simple strategy: all chapters use scheme n"""
        if not all_chapter_schemes:
            return []
        
        children_nodes = []
        
        # Calculate max schemes per chapter
        max_schemes = max([len(chapter_data["schemes"]) for chapter_data in all_chapter_schemes], default=0)
        
        # Strategy: all chapters use same scheme number (all use scheme 1, all use scheme 2...)
        for scheme_idx in range(min(max_schemes, max_nodes)):
            try:
                # Perform deep copy in try block to catch possible serialization errors
                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.node_type = ReportGenerationState.a5  # Correctly set node state to a5
                print(f"[INFO] Creating child node for scheme {scheme_idx+1}, setting state to: {child_node.node_type}")
                
                caption_applied = False  # Track if any caption was applied
                
                # Apply same numbered scheme to each chapter
                for chapter_data in all_chapter_schemes:
                    chapter_idx = chapter_data["chapter_idx"]
                    schemes = chapter_data["schemes"]
                    
                    # If this chapter has a scheme with the corresponding number
                    if 0 <= scheme_idx < len(schemes):
                        scheme = schemes[scheme_idx]
                        
                        # Safely get chapter
                        if 0 <= chapter_idx < len(child_node.report.chapters):
                            chapter = child_node.report.chapters[chapter_idx]
                            
                            print(f"[INFO] Applying scheme {scheme.get('scheme_id', scheme_idx+1)} for chapter {chapter_idx+1} to child node {scheme_idx+1}")
                            
                            # If there is grouping information, also add it to the chapter object
                            if all_chapter_groups and chapter_idx in all_chapter_groups:
                                # Store grouping information in chapter object
                                chapter.chart_groups = all_chapter_groups[chapter_idx]
                                print(f"[SUCCESS] Added {len(chapter.chart_groups)} chart group information for chapter {chapter_idx+1}")
                            
                            # Apply all chart captions in this scheme
                            for caption_info in scheme.get("captions", []):
                                chart_idx = caption_info.get("chart_idx")
                                caption = caption_info.get("caption", "")
                                
                                if hasattr(chapter, 'charts') and 0 <= chart_idx < len(chapter.charts):
                                    chart = chapter.charts[chart_idx]
                                    chart.caption = caption
                                    chart.needs_caption = False
                                    caption_applied = True
                                    
                                    # Update task status
                                    if hasattr(chapter, 'visualization_tasks'):
                                        for task in chapter.visualization_tasks:
                                            if task.get('task_id') == chart.task_id:
                                                task['status'] = 'completed'
                                                task['caption_generated'] = True
                                                break
                
                if caption_applied:  # Only add node if caption was applied
                    child_node.caption_strategy = f"Unified scheme {scheme_idx+1}"
                    # Reconfirm that state is set correctly
                    child_node.node_type = ReportGenerationState.a5
                    children_nodes.append(child_node)
                    print(f"[SUCCESS] Successfully created child node {scheme_idx+1}, using unified scheme {scheme_idx+1}, final state is: {child_node.node_type}")
                else:
                    print(f"[WARNING] Child node {scheme_idx+1} did not apply any caption, skipping this node")

            except Exception as e:
                print(f"[ERROR] Error creating child node for scheme {scheme_idx+1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Final check to ensure all child nodes have correct state
        for i, child in enumerate(children_nodes):
            if child.node_type != ReportGenerationState.a5:
                print(f"[WARNING] Detected child node {i+1} has incorrect state, fixing...")
                child.node_type = ReportGenerationState.a5
        
        if children_nodes:
            print(f"[INFO] Generated {len(children_nodes)} child nodes, all nodes state set to: {ReportGenerationState.a5}")
        
        return children_nodes

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """Generate captions for charts, use new evaluation and grouping method to generate relational captions for grouped charts"""
        print("\n[INFO] Starting chart caption generation task (A5)...")
        
        # Collect chapters with charts that need processing
        chapters_with_charts = []
        for chapter_idx, chapter in enumerate(node.report.chapters):
            # Filter out successfully generated charts that need captions
            successful_charts = self._filter_successful_charts(chapter)
            
            if successful_charts:
                chapters_with_charts.append({
                    "chapter_idx": chapter_idx,
                    "chapter": chapter,
                    "charts": successful_charts
                })
                print(f"[SUCCESS] Chapter {chapter_idx+1} has {len(successful_charts)} charts that need caption generation")
        
        if not chapters_with_charts:
            # No charts to process, return original node
            print("No charts need caption generation, returning original node")
            child_node = copy.deepcopy(node)  # Create a copy
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.node_type = ReportGenerationState.a5  # Ensure state is correctly set to a5
            print(f"[WARNING] No charts to process, setting node state to: {child_node.node_type}")
            return [child_node]
        
        # Generate captions for each chapter
        all_chapter_schemes = []
        all_chapter_groups = {}  # Store chart grouping information for each chapter
        
        for chapter_info in chapters_with_charts:
            chapter_idx = chapter_info["chapter_idx"]
            chapter = chapter_info["chapter"]
            charts = chapter_info["charts"]
            
            # Try new batch evaluation and grouping method
            try:
                print(f"\n[INFO] Processing chapter {chapter_idx+1} with new method")
                
                # Batch evaluate and group charts
                evaluation_result = self.evaluate_and_group_charts(node, chapter, charts)
                
                if evaluation_result and "chart_groups" in evaluation_result:
                    # Use new method - generate relational captions for each group of charts
                    chart_groups = evaluation_result["chart_groups"]
                    print(f"[SUCCESS] Chapter {chapter_idx+1} charts have been divided into {len(chart_groups)} groups")
                    
                    # Save grouping information
                    all_chapter_groups[chapter_idx] = chart_groups
                    
                    # Generate captions for each chart group
                    chapter_schemes = self.generate_group_captions(node, chapter, chart_groups, charts)
                    
                    if chapter_schemes:
                        all_chapter_schemes.append({
                            "chapter_idx": chapter_idx,
                            "schemes": chapter_schemes
                        })
                        print(f"[SUCCESS] Chapter {chapter_idx+1} successfully generated {len(chapter_schemes)} relational caption schemes")
                        continue
                    else:
                        print(f"[WARNING] Chapter {chapter_idx+1} group-level caption generation failed, will fallback to traditional method")
                else:
                    print(f"[WARNING] Chapter {chapter_idx+1} evaluation and grouping failed, will fallback to traditional method")
            except Exception as e:
                print(f"[ERROR] Error processing chapter {chapter_idx+1} with new method: {str(e)}")
                print("[WARNING] Will fallback to traditional method")
                traceback.print_exc()
            
            # Fallback strategy: use traditional method to generate captions for each chart individually
            print(f"[INFO] Processing chapter {chapter_idx+1} with traditional method")
            traditional_schemes = self.generate_chapter_caption_schemes(
                node,
                chapter,
                chapter_idx,
                charts,
                num_schemes=3,
                llm_kwargs=llm_kwargs
            )
            
            if traditional_schemes:
                all_chapter_schemes.append({
                    "chapter_idx": chapter_idx,
                    "schemes": traditional_schemes
                })
                print(f"[SUCCESS] Chapter {chapter_idx+1} successfully generated {len(traditional_schemes)} caption schemes with traditional method")
            else:
                print(f"[ERROR] Chapter {chapter_idx+1} caption generation completely failed")
        
        # Generate child node combinations
        children_nodes = self.generate_combined_nodes(node, all_chapter_schemes, all_chapter_groups)
        
        if not children_nodes:
            # If no child nodes successfully generated, create a basic node
            print("[ERROR] Cannot generate valid child node combinations, will return basic node")
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.node_type = ReportGenerationState.a5  # Ensure state is correctly set to a5
            print(f"[WARNING] Cannot generate valid child nodes, setting node state to: {child_node.node_type}")
            return [child_node]
        
        # Ensure all child nodes state is set to a5
        for child_node in children_nodes:
            child_node.node_type = ReportGenerationState.a5
            
        print(f"[SUCCESS] Successfully generated {len(children_nodes)} child nodes, all nodes state set to: {ReportGenerationState.a5}")
        return children_nodes

    def evaluate_and_group_charts(self, node, chapter, charts):
        """Batch evaluate all charts in chapter and group them
        
        Args:
            node: MCTS node
            chapter: Chapter object
            charts: List of charts
            
        Returns:
            result: Dictionary with evaluation results and grouping information, None if failed
        """
        try:
            # Collect chart images and info
            chart_images = []
            charts_info = ""
            
            for i, chart in enumerate(charts):
                image_base64 = self._get_image_base64(chart.url)
                if image_base64:
                    chart_images.append(image_base64)
                    charts_info += f"\nChart{i}: Type: {chart.chart_type}, Task: {chart.task_id}"
                else:
                    print(f"[ERROR] Cannot get image data for chart {i} ({chart.task_id})")
            
            if not chart_images:
                print("[ERROR] No available chart image data")
                return None
                
            # Build evaluation and grouping prompt
            chapter_title = getattr(chapter, 'title', f'unnamed chapter')
            prompt_args = {
                "CHAPTER_TITLE": chapter_title,
                "CHARTS_INFO": charts_info,
                "CHARTS_COUNT": len(charts),
                "QUERY": node.original_query,
                "DATA_CONTEXT": node.report.data_context
            }
            
            prompt = get_prompt("chart_evaluation_grouping", prompt_args)
            
            # Implement retry mechanism
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Call API for batch evaluation and grouping
                    print(f"[INFO] Evaluating and grouping {len(charts)} charts in chapter \"{chapter_title}\"... (attempt {retry+1}/{max_retries})")
                    
                    # Adjust temperature parameter, decrease temperature as retry count increases for more consistent results
                    temperature = max(0.2, 0.7 - retry * 0.2)
                    response = self.call_vision_api(prompt, chart_images, temperature=temperature)
                    
                    if not response:
                        print(f"[ERROR] API returned empty (attempt {retry+1}/{max_retries})")
                        if retry < max_retries - 1:
                            print("Will retry in 2 seconds...")
                            import time
                            time.sleep(2)
                            continue
                        return None
                        
                    # Parse results
                    print(f"[INFO] LLM response snippet: {response[:200]}...")
                    result = self.extract_json_from_text(response)
                    
                    if result and "chart_evaluations" in result and "chart_groups" in result:
                        # Record evaluation results
                        evaluations = result["chart_evaluations"]
                        print(f"[SUCCESS] Successfully evaluated {len(evaluations)} charts")
                        for eval_info in evaluations:
                            chart_idx = eval_info.get("chart_idx")
                            if 0 <= chart_idx < len(charts):
                                has_insight = eval_info.get("has_insight", False)
                                insight_score = eval_info.get("insight_score", 0)
                                status = "[SUCCESS] Valuable" if has_insight else "[WARNING] Not valuable"
                                print(f"  Chart {chart_idx}: {status} (score: {insight_score})")
                        
                        # Record grouping results
                        groups = result["chart_groups"]
                        print(f"[SUCCESS] Successfully divided charts into {len(groups)} groups")
                        for group in groups:
                            group_id = group.get("group_id")
                            theme = group.get("theme", "unnamed theme")
                            chart_indices = group.get("chart_indices", [])
                            print(f"  - Group {group_id}: {theme} (contains {len(chart_indices)} charts: {chart_indices})")
                        
                        return result
                    else:
                        # Format error, provide retry prompt
                        error_msg = "Parsing result incomplete" if result else "Failed to parse valid JSON result"
                        print(f"[ERROR] {error_msg} (attempt {retry+1}/{max_retries})")
                        
                        if retry < max_retries - 1:
                            print("Will retry in 2 seconds...")
                            import time
                            time.sleep(2)
                        else:
                            print("Reached maximum retry count, evaluation and grouping failed")
                            return None
                            
                except Exception as e:
                    print(f"[ERROR] API call or parsing error: {str(e)} (attempt {retry+1}/{max_retries})")
                    if retry < max_retries - 1:
                        print("Will retry in 2 seconds...")
                        import time
                        time.sleep(2)
                    else:
                        print("Reached maximum retry count, evaluation and grouping failed")
                        traceback.print_exc()
                        return None
                        
        except Exception as e:
            print(f"[ERROR] Error evaluating and grouping charts: {str(e)}")
            traceback.print_exc()
            return None

    def generate_group_captions(self, node, chapter, chart_groups, charts):
        """Generate relational captions for each group of charts
        
        Args:
            node: MCTS node
            chapter: Chapter object
            chart_groups: Chart grouping information
            charts: List of charts
            
        Returns:
            schemes: List containing caption schemes, returns empty list if failed
        """
        try:
            # Store all generated schemes
            schemes = []
            
            # Get chapter title
            chapter_title = getattr(chapter, 'title', 'unnamed chapter')
            print(f"\n[INFO] Generating captions for chart groups in chapter \"{chapter_title}\"")
            
            # Process each chart group
            for group in chart_groups:
                # Skip non-valuable chart groups
                group_theme = group.get("theme", "")
                if "non-valuable" in group_theme.lower() or "non-insightful" in group_theme.lower():
                    print(f"[WARNING] Skipping non-valuable chart group: {group_theme}")
                    continue
                    
                # Get group ID and relationship description
                group_id = group.get("group_id", 0)
                group_relationship = group.get("relationship", "These charts show related data")
                
                # Get all chart indices in this group
                chart_indices = group.get("chart_indices", [])
                if not chart_indices:
                    print(f"[WARNING] Group {group_id} has no charts, skipping")
                    continue
                
                print(f"[INFO] Processing group {group_id}: {group_theme} (contains {len(chart_indices)} charts)")
                
                # Collect group chart images
                group_charts = []
                group_images = []
                
                for idx in chart_indices:
                    if 0 <= idx < len(charts):
                        chart = charts[idx]
                        group_charts.append(chart)
                        
                        image_base64 = self._get_image_base64(chart.url)
                        if image_base64:
                            group_images.append(image_base64)
                        else:
                            print(f"[ERROR] Cannot get image data for chart {idx} in group {group_id}")
                
                if not group_images:
                    print(f"[ERROR] Group {group_id} has no available chart images, skipping")
                    continue
                
                # Build group-level caption prompt
                prompt_args = {
                    "QUERY": node.original_query,
                    "CHAPTER_TITLE": chapter_title,
                    "GROUP_THEME": group_theme,
                    "GROUP_RELATIONSHIP": group_relationship,
                    "CHARTS_COUNT": len(group_charts),
                    "DATA_CONTEXT": node.report.data_context
                }
                
                prompt = get_prompt("group_captions", prompt_args)
                
                # Implement retry mechanism
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Call API to generate caption
                        print(f"[INFO] Generating caption for group {group_id}... (attempt {retry+1}/{max_retries})")
                        
                        # Adjust temperature parameter, decrease temperature as retry count increases
                        temperature = max(0.2, 0.7 - retry * 0.2)
                        response = self.call_vision_api(prompt, group_images, temperature=temperature)
                        
                        if not response:
                            print(f"[ERROR] API for group {group_id} returned empty (attempt {retry+1}/{max_retries})")
                            if retry < max_retries - 1:
                                print("Will retry in 2 seconds...")
                                import time
                                time.sleep(2)
                                continue
                            break
                            
                        # Parse results
                        print(f"[INFO] LLM response snippet: {response[:200]}...")
                        caption_result = self.extract_json_from_text(response)
                        
                        if caption_result and "captions" in caption_result:
                            # Create caption scheme
                            scheme = {
                                "scheme_id": len(schemes) + 1,
                                "theme": caption_result.get("theme", group_theme),
                                "captions": []
                            }
                            
                            # Process captions for each chart
                            captions = caption_result["captions"]
                            print(f"[SUCCESS] Successfully generated {len(captions)} captions for group {group_id}")
                            
                            for i, chart_idx in enumerate(chart_indices):
                                if i < len(captions):
                                    caption_entry = captions[i]
                                    # Extract caption text, prioritize using chart_position match
                                    caption_text = ""
                                    
                                    # Find if there's a caption with position match
                                    for entry in captions:
                                        if entry.get("chart_position") == i:
                                            caption_text = entry.get("caption", "")
                                            break
                                    
                                    # If not found, use sequential match
                                    if not caption_text:
                                        caption_text = caption_entry.get("caption", "")
                                    
                                    scheme["captions"].append({
                                        "chart_idx": chart_idx,
                                        "caption": caption_text
                                    })
                                    
                            # Add to schemes list
                            schemes.append(scheme)
                            # Successfully generated, break out of retry loop
                            break
                            
                        else:
                            # Format error, provide retry prompt
                            error_msg = "Parsing result incomplete" if caption_result else "Failed to parse valid JSON result"
                            print(f"[ERROR] {error_msg} (attempt {retry+1}/{max_retries})")
                            
                            if retry < max_retries - 1:
                                print("Will retry in 2 seconds...")
                                import time
                                time.sleep(2)
                            else:
                                print(f"[ERROR] Reached maximum retry count, caption generation for group {group_id} failed")
                                
                    except Exception as e:
                        print(f"[ERROR] Caption generation error for group {group_id}: {str(e)} (attempt {retry+1}/{max_retries})")
                        if retry < max_retries - 1:
                            print("Will retry in 2 seconds...")
                            import time
                            time.sleep(2)
                        else:
                            print(f"[ERROR] Reached maximum retry count, caption generation for group {group_id} failed")
                            traceback.print_exc()
            
            # Return all generated schemes
            if schemes:
                print(f"[SUCCESS] Successfully generated caption schemes for {len(schemes)} chart groups")
            else:
                print("[WARNING] Failed to generate caption schemes for any chart group")
                
            return schemes
                
        except Exception as e:
            print(f"[ERROR] Error generating group-level captions: {str(e)}")
            traceback.print_exc()
            return []


class Captions2Summaries(DataStorytellingAction):
    def __init__(self):
        super().__init__("A6", "Generate summary for each chapter based on its captions")
        self.use_unified_framework = True  # Whether to use unified framework
    
    def generate_summary_prompt(self, node, chapter_idx=None, **kwargs):
        """Generate chapter summary prompt"""
        # If chapter index is specified, generate prompt for specific chapter
        if chapter_idx is not None and 0 <= chapter_idx < len(node.report.chapters):
            chapter = node.report.chapters[chapter_idx]
            chapter_title = getattr(chapter, 'title', f"Chapter{chapter_idx+1}") if not isinstance(chapter, dict) else chapter.get('title', f"Chapter{chapter_idx+1}")
            
            # Collect all charts and their captions in this chapter
            visualization_tasks = []
            if hasattr(chapter, 'visualization_tasks'):
                for task in chapter.visualization_tasks:
                    task_info = {
                        'description': task.get('task_description', ''),
                        'charts': []
                    }
                    
                    # Check if chapter has charts
                    if hasattr(chapter, 'charts') and chapter.charts:
                        # Find charts associated with task
                        for chart in chapter.charts:
                            if hasattr(chart, 'task_id') and chart.task_id == task.get('task_id'):
                                # Check if this chart task was completed successfully
                                task_success = False
                                for t in chapter.visualization_tasks:
                                    if t.get('task_id') == chart.task_id and t.get('visualization_success', False):
                                        task_success = True
                                        break
                                
                                if task_success:
                                    caption = getattr(chart, 'caption', 'No caption text')
                                    task_info['charts'].append({
                                        'caption': caption
                                    })
                    
                    if task_info['charts']:
                        visualization_tasks.append(task_info)
            
            # Prepare prompt arguments
            prompt_args = {
                "QUERY": node.original_query,
                "CHAPTER_TITLE": chapter_title,
                "visualization_tasks": json.dumps(visualization_tasks, ensure_ascii=False, indent=2)
            }
            
            return get_prompt("chapter_summary", prompt_args)
        
        # If no chapter is specified, return basic information
        return {
            "QUERY": node.report.clarified_query if node.report.clarified_query else node.report.original_query,
            "DATA_CONTEXT": node.report.data_context
        }
    
    def apply_summaries(self, node, action, cluster, **kwargs):
        """Apply chapter summaries to child nodes"""
        # Create child node
        child_node = copy.deepcopy(node)
        child_node.parent_node = node
        child_node.parent_action = action
        child_node.depth = node.depth + 1
        
        try:
            # Get summaries for each chapter from cluster
            if "chapter_summaries" in cluster:
                chapter_summaries = cluster["chapter_summaries"]
                
                # Apply chapter summaries
                success_count = 0
                for chapter_summary in chapter_summaries:
                    chapter_idx = chapter_summary.get("chapter_idx", -1)
                    summary = chapter_summary.get("summary", "")
                    
                    if 0 <= chapter_idx < len(child_node.report.chapters):
                        chapter = child_node.report.chapters[chapter_idx]
                        chapter.summary = summary
                        success_count += 1
                        print(f"✅ Applied summary for chapter {chapter_idx + 1}")
                
                # Set node state
                if success_count > 0:
                    child_node.node_type = ReportGenerationState.FINALIZED
                    return [child_node]
            
            # If no summaries were obtained from cluster, try to process them ourselves
            print("⚠️ No chapter summaries obtained from cluster, trying to process ourselves...")
            success = self.process_all_chapters(child_node, **kwargs)
            
            if success:
                # Set node state
                child_node.node_type = ReportGenerationState.FINALIZED
                return [child_node]
            else:
                print("❌ Failed to process chapter summaries")
                return None
                
        except Exception as e:
            print(f"❌ Error applying chapter summaries: {str(e)}")
            traceback.print_exc()
            return None
    
    def generate_chapter_summaries(self, node, llm_kwargs, n=3):
        """Generate multiple candidate summaries for each chapter"""
        all_chapter_summaries = []
        
        # Iterate through all chapters
        for chapter_idx, chapter in enumerate(node.report.chapters):
            # Safely get chapter title
            chapter_title = getattr(chapter, 'title', f"Chapter{chapter_idx+1}") if not isinstance(chapter, dict) else chapter.get('title', f"Chapter{chapter_idx+1}")
            
            print(f"\n📑 Generating multiple candidate summaries for chapter {chapter_idx + 1}...")
            print(f"Chapter title: {chapter_title}")
            
            # Check if chapter has charts and captions
            has_captions = False
            if hasattr(chapter, 'charts') and chapter.charts:
                for chart in chapter.charts:
                    if hasattr(chart, 'caption') and chart.caption:
                        has_captions = True
                        break
            
            if not has_captions:
                print(f"⚠️ Chapter {chapter_idx + 1} has no charts or captions, skipping")
                continue
                
            # Generate prompt for this chapter
            prompt = self.generate_summary_prompt(node, chapter_idx=chapter_idx)
            
            # Collect multiple candidate summaries for this chapter
            chapter_summaries = []
            
            for i in range(n):
                # Use different temperature for each candidate
                llm_kwargs_temp = llm_kwargs.copy()
                llm_kwargs_temp['temperature'] = 0.3 + i * 0.2  # 0.3, 0.5, 0.7
                
                print(f"🔄 Generating candidate summary {i+1}/{n} for chapter {chapter_idx + 1} (temperature: {llm_kwargs_temp['temperature']})")
                
                responses = call_openai(prompt, **llm_kwargs_temp)
                if responses:
                    summary = responses[0].strip()
                    
                    # Collect candidate summary
                    chapter_summaries.append({
                        "chapter_idx": chapter_idx,
                        "summary": summary,
                        "variant_id": i
                    })
                    
                    print(f"✅ Successfully generated candidate summary {i+1} for chapter {chapter_idx + 1}")
                else:
                    print(f"❌ Failed to generate candidate summary {i+1} for chapter {chapter_idx + 1}")
            
            # If candidate summaries were successfully generated, add to list
            if chapter_summaries:
                all_chapter_summaries.append({
                    "chapter_idx": chapter_idx,
                    "chapter_title": chapter_title,
                    "candidate_summaries": chapter_summaries
                })
        
        return all_chapter_summaries
    
    def cluster_chapter_summaries(self, all_chapter_summaries, llm_kwargs):
        """Cluster candidate summaries for each chapter and select the best summary"""
        if not all_chapter_summaries:
            return []
        
        try:
            # Prepare clustering data
            formatting_data = []
            for chapter_data in all_chapter_summaries:
                chapter_idx = chapter_data["chapter_idx"]
                chapter_title = chapter_data["chapter_title"]
                candidates = chapter_data["candidate_summaries"]
                
                # Convert to format required for clustering
                formatted_candidates = [
                    {
                        "index": candidate["variant_id"],
                        "chapter_idx": chapter_idx,
                        "summary": candidate["summary"]
                    }
                    for candidate in candidates
                ]
                
                formatting_data.append({
                    "chapter_idx": chapter_idx,
                    "chapter_title": chapter_title,
                    "candidates": formatted_candidates
                })
            
            # Use template to generate clustering prompt
            prompt_args = {
                "CHAPTER_SUMMARIES_DATA": json.dumps(formatting_data, ensure_ascii=False, indent=2)
            }
            
            clustering_prompt = get_prompt("chapter_summary_clustering", prompt_args)
            
            # Call LLM for clustering
            print("\n🔍 Performing clustering analysis on chapter summaries...")
            responses = call_openai(clustering_prompt, **llm_kwargs)
            
            if not responses:
                print("❌ Clustering analysis received no valid response")
                return []
            
            # Parse response
            clustering_response = responses[0]
            
            # Extract JSON part
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', clustering_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = clustering_response
            
            try:
                # Parse JSON
                clustering_result = json.loads(json_str)
                
                # Check if there are valid clustering results
                if "clusters" in clustering_result and clustering_result["clusters"]:
                    print(f"✅ Successfully obtained {len(clustering_result['clusters'])} clusters")
                    return clustering_result["clusters"]
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {str(e)}")
                print(f"❌ Raw response:\n{clustering_response}")
        
        except Exception as e:
            print(f"❌ Error clustering chapter summaries: {str(e)}")
            traceback.print_exc()
        
        return []
    
    def process_all_chapters(self, node, **kwargs):
        """Process all chapters and generate summaries for each chapter"""
        llm_kwargs = kwargs.get("llm_kwargs", {})
        
        try:
            # If using unified framework and have multiple candidate summaries
            if self.use_unified_framework:
                # Generate multiple candidate summaries for each chapter
                all_chapter_summaries = self.generate_chapter_summaries(node, llm_kwargs)
                
                if not all_chapter_summaries:
                    print("❌ No candidate summaries were successfully generated for any chapter")
                    return False
                
                # Cluster candidate summaries and select the best
                clusters = self.cluster_chapter_summaries(all_chapter_summaries, llm_kwargs)
                
                if not clusters:
                    print("❌ No valid clustering results obtained")
                    return False
                
                # Apply the first cluster's results
                cluster = clusters[0]
                print(f"✅ Applying summary results from cluster {cluster.get('cluster_id', 'unknown')}")
                
                # Apply chapter summaries
                chapter_summaries = cluster.get("chapter_summaries", [])
                success_count = 0
                
                for chapter_summary in chapter_summaries:
                    chapter_idx = chapter_summary.get("chapter_idx", -1)
                    summary = chapter_summary.get("summary", "")
                    
                    if 0 <= chapter_idx < len(node.report.chapters):
                        chapter = node.report.chapters[chapter_idx]
                        chapter.summary = summary
                        success_count += 1
                        print(f"✅ Applied summary for chapter {chapter_idx + 1}")
                
                return success_count > 0
            else:
                # Original logic (when not using unified framework)
                success_count = 0

                # Iterate through all chapters
                for chapter_idx, chapter in enumerate(node.report.chapters):
                    # Safely get chapter title
                    chapter_title = getattr(chapter, 'title', f"Chapter{chapter_idx+1}") if not isinstance(chapter, dict) else chapter.get('title', f"Chapter{chapter_idx+1}")
                    
                    print(f"\n📑 Processing chapter {chapter_idx + 1}: {chapter_title}")
                    
                    # Check if chapter has visualization tasks
                    if not hasattr(chapter, 'visualization_tasks') or not chapter.visualization_tasks:
                        print(f"⚠️ Chapter {chapter_idx + 1} has no visualization tasks, skipping")
                        continue
                    
                    # Collect all charts and their captions in this chapter
                    visualization_tasks = []
                    for task in chapter.visualization_tasks:
                        task_info = {
                            'description': task.get('task_description', ''),
                            'charts': []
                        }
                        
                        # Check if chapter has charts
                        if not hasattr(chapter, 'charts') or not chapter.charts:
                            continue
                            
                        # Find charts associated with task
                        for chart in chapter.charts:
                            if hasattr(chart, 'task_id') and chart.task_id == task.get('task_id'):
                                caption = getattr(chart, 'caption', 'No caption text')
                                task_info['charts'].append({
                                    'caption': caption
                                })
                        
                        # Only add tasks that have charts
                        if task_info['charts']:
                            visualization_tasks.append(task_info)
                    
                    # If no valid visualization tasks were collected, skip this chapter
                    if not visualization_tasks:
                        print(f"⚠️ Chapter {chapter_idx + 1} has no valid visualization task charts, skipping")
                        continue
                    
                    # Prepare prompt
                    prompt_args = {
                        "QUERY": node.original_query,
                        "CHAPTER_TITLE": chapter_title,
                        "visualization_tasks": json.dumps(visualization_tasks, ensure_ascii=False, indent=2)
                    }
                    
                    prompt = get_prompt("chapter_summary", prompt_args)
                    
                    # Call LLM to generate summary
                    responses = call_openai(prompt, **llm_kwargs)
                    if not responses:
                        print(f"❌ Chapter {chapter_idx + 1} received no valid response")
                        continue
                    
                    summary = responses[0].strip()
                    
                    print(f"\n📝 Summary for chapter {chapter_idx + 1}:")
                    print("-" * 50)
                    print(summary)
                    print("-" * 50)
                    
                    # Save summary to chapter
                    chapter.summary = summary
                    print(f"✅ Generated summary for chapter {chapter_idx + 1}")
                    success_count += 1

                return success_count > 0
                
        except Exception as e:
            print(f"❌ Error generating chapter summaries: {str(e)}")
            traceback.print_exc()
            return False
                
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        if self.use_unified_framework:
            # Generate multiple candidate summaries for each chapter
            all_chapter_summaries = self.generate_chapter_summaries(node, llm_kwargs)
            
            if not all_chapter_summaries:
                print("❌ No candidate summaries were successfully generated for any chapter, creating default node")
                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.node_type = ReportGenerationState.FINALIZED
                return [child_node]
            
            # Cluster candidate summaries and select the best
            clusters = self.cluster_chapter_summaries(all_chapter_summaries, llm_kwargs)
            
            if not clusters:
                print("❌ No valid clustering results obtained, creating default node")
                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.node_type = ReportGenerationState.FINALIZED
                return [child_node]
            
            # Create a child node for each cluster
            children_nodes = []
            
            for cluster_idx, cluster in enumerate(clusters):
                cluster_id = cluster.get("cluster_id", f"cluster_{cluster_idx+1}")
                
                print(f"🔄 Creating child node for cluster {cluster_id}")
                
                # Create child node
                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                
                # Apply chapter summaries
                chapter_summaries = cluster.get("chapter_summaries", [])
                success_count = 0
                
                for chapter_summary in chapter_summaries:
                    chapter_idx = chapter_summary.get("chapter_idx", -1)
                    summary = chapter_summary.get("summary", "")
                    
                    if 0 <= chapter_idx < len(child_node.report.chapters):
                        chapter = child_node.report.chapters[chapter_idx]
                        chapter.summary = summary
                        success_count += 1
                        print(f"✅ Applied summary for chapter {chapter_idx + 1} for cluster {cluster_id}")
                
                if success_count > 0:
                    # Set node state
                    child_node.node_type = ReportGenerationState.a6
                    child_node.summary_cluster_id = cluster_id
                    children_nodes.append(child_node)
                    print(f"✅ Successfully created child node for cluster {cluster_id}")
            
            # If no child nodes were created, create a default node
            if not children_nodes:
                print("❌ No valid child nodes were created, creating default node")
                child_node = copy.deepcopy(node)
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.node_type = ReportGenerationState.a6
                return [child_node]
            
            return children_nodes
        else:
            # Original implementation (kept for compatibility)
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            
            # Process all chapters' summaries
            self.process_all_chapters(child_node, llm_kwargs=llm_kwargs)
        
        # Set final state
        child_node.node_type = ReportGenerationState.a6
        
        return [child_node]
    
class ReviseNarrativeStrategy(DataStorytellingAction):
    def __init__(self):
        super().__init__("NarrativeStrategy", "Adjust report narrative strategy, reorder chapters")
        self.use_unified_framework = True  # Use unified framework
    
    def generate_narrative_prompt(self, node, **kwargs):
        """Generate narrative strategy prompt"""
        # Prepare chapter information
        chapters_info = []
        for i, chapter in enumerate(node.report.chapters):
            chapter_info = {
                "index": i,
                "title": getattr(chapter, 'title', f"Chapter{i+1}"),
                "summary": getattr(chapter, 'summary', "") if hasattr(chapter, 'summary') else ""
            }
            chapters_info.append(chapter_info)
        
        # Use template to generate prompt
        prompt_args = {
            "QUERY": node.report.clarified_query if node.report.clarified_query else node.report.original_query,
            "CHAPTERS": json.dumps(chapters_info, ensure_ascii=False, indent=2)
        }
        
        return get_prompt("revise_narrative", prompt_args)
    
    def apply_narrative_strategy(self, node, action, cluster, **kwargs):
        """Apply narrative strategy to child nodes"""
        try:
            # Create child node
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = action
            child_node.depth = node.depth + 1
            
            # Get narrative strategy and chapter order
            cluster_id = cluster.get("cluster_id", "unknown")
            strategy = cluster.get("strategy", "")
            strategy_reason = cluster.get("strategy_reason", "")
            chapter_order = cluster.get("chapter_order", [])
            
            if not chapter_order:
                print(f"⚠️ Cluster {cluster_id} has no chapter order information, skipping")
                return None
            
            print(f"📘 Applying narrative strategy plan from cluster {cluster_id}")
            print(f"   Strategy: {strategy}")
            print(f"   Reason: {strategy_reason}")
            
            # Validate chapter order
            if len(chapter_order) != len(node.report.chapters):
                print(f"⚠️ Chapter count mismatch: expected {len(node.report.chapters)}, got {len(chapter_order)}")
                return None
                
            # Reorder chapters based on new order
            new_chapters = []
            for chapter_info in chapter_order:
                original_index = chapter_info.get("original_index")
                if not isinstance(original_index, int) or original_index < 0 or original_index >= len(node.report.chapters):
                    print(f"⚠️ Invalid chapter index: {original_index}")
                    continue
                
                new_chapters.append(copy.deepcopy(node.report.chapters[original_index]))
                print(f"   - Moving chapter '{chapter_info.get('title', '')}' to new position")
                print(f"     Reason: {chapter_info.get('reason', 'Not provided')}")
            
            # If not all chapters were successfully reordered, skip this cluster
            if len(new_chapters) != len(node.report.chapters):
                print(f"⚠️ Chapter reordering incomplete, skipping this cluster")
                return None
            
            # Update report's chapter order and narrative strategy
            child_node.report.chapters = new_chapters
            child_node.report.narrative_strategy = {
                "strategy": strategy,
                "strategy_reason": strategy_reason,
                "cluster_id": cluster_id
            }
            
            # Set node state
            child_node.node_type = ReportGenerationState.REVISECHAPTERSORDERS
            
            print(f"✅ Successfully applied narrative strategy plan from cluster {cluster_id}")
            return [child_node]
            
        except Exception as e:
            print(f"❌ Error applying narrative strategy: {str(e)}")
            traceback.print_exc()
            return None
    
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """Generate multiple narrative strategy plans and cluster to select the best"""
        return unified_generation_framework(
            node=node,
            action=self,
            llm_kwargs=llm_kwargs,
            action_type="narrative",
            prompt_generator=self.generate_narrative_prompt,
            node_applier=self.apply_narrative_strategy,
            n=3 
        )



class TransitionAction(DataStorytellingAction):
    def __init__(self):
        super().__init__("Transition", "Add transition text between chapters to improve report coherence")
        self.use_unified_framework = True  # Use unified framework
    
    def generate_transition_prompt(self, node, **kwargs):
        """Generate transition text prompt"""
        # Prepare chapter information
        chapters_info = []
        for i, chapter in enumerate(node.report.chapters):
            chapter_info = {
                "index": i,
                "title": getattr(chapter, 'title', f"Chapter{i+1}"),
                "summary": getattr(chapter, 'summary', "") if hasattr(chapter, 'summary') else "",
                "charts_captions": [
                    getattr(chart, 'caption', "") for chart in getattr(chapter, 'charts', [])
                ]
            }
            chapters_info.append(chapter_info)
        
        # Use template to generate prompt
        prompt_args = {
            "QUERY": node.report.clarified_query if node.report.clarified_query else node.report.original_query,
            "CHAPTERS": json.dumps(chapters_info, ensure_ascii=False, indent=2),
            "NARRATIVE_STRATEGY": json.dumps(getattr(node.report, 'narrative_strategy', {}), ensure_ascii=False, indent=2)
        }
        
        return get_prompt("add_transitions", prompt_args)
    
    def apply_transitions(self, node, action, cluster, **kwargs):
        """Apply transition text to child nodes"""
        try:
            # Create child node
            child_node = copy.deepcopy(node)
            child_node.parent_node = node
            child_node.parent_action = action
            child_node.depth = node.depth + 1
            
            # Get transition text plan
            cluster_id = cluster.get("cluster_id", "unknown")
            transitions = cluster.get("transitions", [])
            
            if not transitions:
                print(f"⚠️ Cluster {cluster_id} has no transition text information, skipping")
                return None
            
            print(f"📝 Applying transition text plan from cluster {cluster_id}")
            
            # Apply transition text
            success_count = 0
            for transition in transitions:
                chapter_idx = transition.get("chapter_idx")
                transition_text = transition.get("transition_text", "")
                
                if not isinstance(chapter_idx, int) or chapter_idx < 0 or chapter_idx >= len(child_node.report.chapters):
                    print(f"⚠️ Invalid chapter index: {chapter_idx}")
                    continue
                
                # Add transition text to chapter
                chapter = child_node.report.chapters[chapter_idx]
                if not hasattr(chapter, 'transition'):
                    chapter.transition = ""
                
                chapter.transition = transition_text
                success_count += 1
                print(f"   ✅ Added transition text for chapter {chapter_idx + 1}")
            
            # If no transition text was successfully added, skip this cluster
            if success_count == 0:
                print(f"⚠️ No transition text was successfully added, skipping this cluster")
                return None
            
            # Set node state
            child_node.node_type = ReportGenerationState.FINALIZED
            
            print(f"✅ Successfully applied transition text plan from cluster {cluster_id}, total {success_count} transitions")
            return [child_node]
            
        except Exception as e:
            print(f"❌ Error applying transition text: {str(e)}")
            traceback.print_exc()
            return None
    
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        """Generate multiple transition text plans and cluster to select the best"""
        return unified_generation_framework(
            node=node,
            action=self,
            llm_kwargs=llm_kwargs,
            action_type="transition",
            prompt_generator=self.generate_transition_prompt,
            node_applier=self.apply_transitions,
            n=3  # Generate 3 different transition text plans
        )

    

# Fix save_chart method, make it a class method instead of standalone function
class ChartUtils:
    @staticmethod
    def save_chart(node: MCTSNode, chart_data: dict) -> str:
        """Save chart and return URL"""
        # Get current iteration number, add debug info
        current_iteration = node.report.current_iteration
        print(f"Debug: Iteration number when saving chart: {current_iteration}")
        print(f"Debug: Node type: {node.node_type}")
        print(f"Debug: Node depth: {node.depth}")
        
        # Ensure correct iteration number is used
        if current_iteration is None or current_iteration < 1:
            print("Warning: current_iteration is invalid, using default value 1")
            current_iteration = 1
        
        # Build save path
        iteration_dir = os.path.join("storyteller", "output", "iterations", f"iteration_{current_iteration}")
        charts_dir = os.path.join(iteration_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        print(f"Debug: Charts will be saved to: {charts_dir}")
        
        return charts_dir

    def get_current_iteration_dir(self):
        """Get current iteration output directory"""
        try:
            # Check if current iteration directory property exists
            if hasattr(self, 'current_iteration_dir') and self.current_iteration_dir:
                return self.current_iteration_dir
            
            # Check if output root directory property exists
            if hasattr(self, 'output_dir') and self.output_dir:
                # Find latest iteration directory
                iteration_dirs = glob.glob(os.path.join(self.output_dir, "iteration_*"))
                if iteration_dirs:
                    # Sort by creation time, get latest
                    latest_dir = max(iteration_dirs, key=os.path.getctime)
                    return latest_dir
            
            # If no output directory set, use default output directory
            default_output_dir = os.path.join("output", "mcts")
            os.makedirs(default_output_dir, exist_ok=True)
            
            # Find latest iteration directory
            iteration_dirs = glob.glob(os.path.join(default_output_dir, "iteration_*"))
            if iteration_dirs:
                latest_dir = max(iteration_dirs, key=os.path.getctime)
                return latest_dir
            
            # If no iteration directory found, create a new one
            new_dir = os.path.join(default_output_dir, f"iteration_{int(time.time())}")
            os.makedirs(new_dir, exist_ok=True)
            return new_dir
            
        except Exception as e:
            print(f"⚠️ Error getting current iteration directory: {str(e)}")
            # Return temporary directory
            temp_dir = os.path.join("output", "temp_charts")
            os.makedirs(temp_dir, exist_ok=True)
            return temp_dir




# Keep dictionary definition as module-level variable
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
    ReportGenerationState.FINALIZED: []  # Terminal state, final state after adding transitions
}



