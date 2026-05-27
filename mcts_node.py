from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from copy import deepcopy
from enum import Enum
import copy

if TYPE_CHECKING:
    from .mcts_action import DataStorytellingAction


class ReportGenerationState(Enum):
    EMPTY = "Empty"

    a1 = "a1"

    a2 = "a2"

    a3 = "a3"

    a4 = "a4"

    a5 = "a5"

    a6 = "a6"

    a7 = "a7"

    a8 = "a8"

    a9 = "a9"

    a10 = "a10"

    a11 = "a11"

    FINALIZED = "Finalized"

    REVISECHAPTERSORDERS = "ReviseChaptersOrders"

    ADDEDTRANSITIONS = "AddedTransitions"




class Chart:
    def __init__(self,
                url: str,
                caption: str,
                chart_position: str = "center",
                code: str = None,
                chart_type: str = None,
                task_id: str = None):
        """
        Initialize chart object

        Args:
            url: Chart URL address
            caption: Chart caption text
            chart_position: Chart position in page, default center
            code: Code to generate chart
            chart_type: Chart type
            task_id: Associated task ID
        """
        self.type = "chart"
        self.url = url
        self.caption = caption
        self.chart_position = chart_position
        self.code = code
        self.chart_type = chart_type
        self.task_id = task_id
        self.needs_caption = False

    def to_dict(self):
        """
        Convert chart object to dictionary for JSON serialization

        Returns:
            Dictionary containing chart information
        """
        chart_dict = {
            "type": self.type,
            "url": self.url,
            "caption": self.caption,
            "chart_position": self.chart_position,
            "code": self.code,
            "chart_type": self.chart_type,
            "task_id": self.task_id,
            "needs_caption": self.needs_caption
        }
        return chart_dict


class ChartGroup:
    def __init__(self,
                 charts_list: List[Chart],
                 caption: Optional[str] = None,
                 chart_position: str = "side-by-side",
                 caption_position: str = "below"):
        """
        Initialize chart group object

        Args:
            charts_list: List of chart objects
            caption: Shared caption for the entire chart group
        """
        self.type = "chart_group"
        self.charts_list = charts_list
        self.caption = caption
        self.chart_position = chart_position

    def to_dict(self):
        """
        Convert chart group object to dictionary for JSON serialization

        Returns:
            Dictionary containing chart group information
        """
        return {
            "type": self.type,
            "charts_list": [chart.to_dict() for chart in self.charts_list],
            "caption": self.caption,
            "chart_position": self.chart_position
        }


class Chapter:
    def __init__(self, title: str, summary: Optional[str] = None):
        """
        Initialize chapter object

        Args:
            title: Chapter title
            summary: Chapter summary
        """
        self.title = title
        self.summary = summary
        self.charts: List[Union[Chart, ChartGroup]] = []
        self.visualization_tasks = []
        self.tasks_status = {}

    def add_chart(self, chart: Chart):
        """
        Add a single chart to the chapter

        Args:
            chart: Chart object to add
        """
        self.charts.append(chart)

    def add_chart_group(self, chart_group: ChartGroup):
        """
        Add a chart group to the chapter

        Args:
            chart_group: Chart group object to add
        """
        self.charts.append(chart_group)

    def initialize_tasks_status(self):
        """Initialize all visualization tasks status to 'pending'"""
        if hasattr(self, 'visualization_tasks'):
            self.tasks_status = {}
            for i, task in enumerate(self.visualization_tasks):
                task_id = task.get('task_id', f"task_{i+1}")
                self.tasks_status[task_id] = "pending"

    def get_next_pending_task(self):
        """Get the next pending task"""
        if not hasattr(self, 'visualization_tasks') or not self.visualization_tasks:
            return None, None

        if not hasattr(self, 'tasks_status'):
            self.tasks_status = {}

        for task in self.visualization_tasks:
            task_id = task.get('task_id')
            if task_id:
                task_status = self.tasks_status.get(task_id, "pending")
                if task_status == "pending":
                    return task_id, task

        return None, None

    def mark_task_in_progress(self, task_id):
        """Mark task as in progress"""
        if not hasattr(self, 'tasks_status'):
            self.tasks_status = {}
        self.tasks_status[task_id] = "in_progress"

    def mark_task_completed(self, task_id):
        """Mark task as completed"""
        if not hasattr(self, 'tasks_status'):
            self.tasks_status = {}
        self.tasks_status[task_id] = "completed"

    def all_tasks_completed(self):
        """Check if all tasks are completed"""
        if not hasattr(self, 'visualization_tasks') or not self.visualization_tasks:
            return True

        if not hasattr(self, 'tasks_status'):
            return False

        for task in self.visualization_tasks:
            task_id = task.get('task_id')
            if task_id and self.tasks_status.get(task_id) != "completed":
                return False
        return True

    def to_dict(self):
        """
        Convert chapter object to dictionary for JSON serialization

        Returns:
            Dictionary containing chapter information
        """
        return {
            "title": self.title,
            "summary": self.summary,
            "charts": [chart.to_dict() for chart in self.charts]
        }


class Report:
    def __init__(self,
                 original_query: str,
                 dataset_path: str,
                 data_context: str = "",
                 clarified_query: str = "",
                 dataset_description: str = "",
                 task_list: List[str] = None,
                 narrative_strategy: str = "data-driven",
                 layout_strategy: str = "thematic",
                 chapters: Optional[List[Chapter]] = None):
        """
        Initialize report object

        Args:
            original_query: Original user query
            dataset_path: Dataset path
            data_context: Data summary
            clarified_query: Clarified query
            dataset_description: Dataset description
            task_list: Task list
            narrative_strategy: Narrative strategy
            layout_strategy: Layout strategy
            chapters: Chapter list
        """
        self.original_query = original_query
        self.dataset_path = dataset_path
        self.data_context = data_context
        self.clarified_query = clarified_query
        self.dataset_description = dataset_description
        self.task_list = task_list if task_list else []
        self.narrative_strategy = narrative_strategy
        self.layout_strategy = layout_strategy
        self.chapters = chapters if chapters else []
        self.full_column_names = []
        self.current_iteration = 0

    def add_chapter(self, chapter: Chapter) -> None:
        """
        Add a chapter to the report

        Args:
            chapter: Chapter object to add
        """
        self.chapters.append(chapter)

    def get_chapter_by_title(self, title: str) -> Optional[Chapter]:
        """
        Get chapter by title

        Args:
            title: Chapter title

        Returns:
            Chapter object if found, None otherwise
        """
        for chapter in self.chapters:
            if chapter.title == title:
                return chapter
        return None

    def get_chapter_titles(self) -> List[str]:
        """Get all chapter titles"""
        return [chapter.title for chapter in self.chapters]

    def get_chart_count(self) -> int:
        """Get total chart count in the report"""
        return sum(len(chapter.charts) for chapter in self.chapters)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report object to dictionary for JSON serialization

        Returns:
            Dictionary containing report information
        """
        return {
            "original_query": self.original_query,
            "data_context": self.data_context,
            "clarified_query": self.clarified_query,
            "dataset_description": self.dataset_description,
            "task_list": self.task_list,
            "dataset_path": self.dataset_path,
            "narrative_strategy": self.narrative_strategy,
            "layout_strategy": self.layout_strategy,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }

    def __str__(self) -> str:
        """String representation"""
        return f"Report(query='{self.original_query}', chapters={len(self.chapters)})"


class MCTSNode:
    def __init__(self,
                 node_type: Union[str, ReportGenerationState],
                 parent_node: Optional["MCTSNode"] = None,
                 parent_action: Optional["DataStorytellingAction"] = None,
                 depth: int = 0,
                 report: Optional[Report] = None,
                 original_query: str = "",
                 llm_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize MCTS node
        """
        if isinstance(node_type, str):
            try:
                self.node_type = ReportGenerationState[node_type]
            except KeyError:
                try:
                    self.node_type = ReportGenerationState(node_type)
                except ValueError:
                    raise ValueError(f"Invalid node_type: {node_type}")
        else:
            self.node_type = node_type

        self.parent_node = parent_node
        self.parent_action = parent_action
        self.depth = depth

        if report:
            self.report = deepcopy(report)
        elif parent_node:
            self.report = deepcopy(parent_node.report)
        else:
            raise ValueError("Either report or parent_node.report must be provided.")

        self.original_query = original_query
        self.llm_kwargs = llm_kwargs if llm_kwargs else {}

        self.children: List["MCTSNode"] = []

        self.Q = 0.0
        self.N = 0

        self.expanded = False

        self.selected_task = None
        self.selected_chapter_idx = None

        self.data_processed = False

    def add_child(self, child_node: "MCTSNode") -> None:
        """Add a child node"""
        self.children.append(child_node)

    def get_chapter_count(self) -> int:
        """Get chapter count in the report"""
        return len(self.report.chapters)

    def get_chart_count(self) -> int:
        """Get total chart count in the report"""
        return sum(len(chapter.charts) for chapter in self.report.chapters)

    def get_report_summary(self) -> Dict[str, Any]:
        """Get report summary information"""
        return {
            "query": self.original_query,
            "chapter_count": self.get_chapter_count(),
            "chart_count": self.get_chart_count(),
            "node_type": self.node_type,
            "depth": self.depth
        }

    def expand(self, action_space: List["DataStorytellingAction"]):
        """
        Expand current node, generate child nodes

        Args:
            action_space: List of available actions
        """
        if self.children:
            return

        for action in action_space:
            child_report = deepcopy(self.report)
            action.execute(child_report=child_report, llm_kwargs=self.llm_kwargs)
            child_node = MCTSNode(
                node_type=action.next_node_type,
                parent_node=self,
                parent_action=action,
                depth=self.depth + 1,
                report=child_report,
                original_query=self.original_query,
                llm_kwargs=self.llm_kwargs
            )
            self.children.append(child_node)

    def is_terminal(self):
        """
        Check if this is a terminal node

        Returns:
            True if node type is FINALIZED, False otherwise
        """
        return self.node_type == ReportGenerationState.FINALIZED

    def __str__(self) -> str:
        """String representation"""
        return f"MCTSNode(type={self.node_type}, depth={self.depth}, chapters={self.get_chapter_count()})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "node_type": self.node_type,
            "depth": self.depth,
            "report": self.report.to_dict(),
            "children_count": len(self.children),
            "Q": self.Q,
            "N": self.N
        }

    def generate_html_report(self) -> str:
        """Generate HTML report for this node"""
        pass

    def copy(self):
        """Create a deep copy of the node"""
        new_node = MCTSNode(self.node_type)
        new_node.visits = self.visits
        new_node.value = self.value
        new_node.depth = self.depth
        new_node.report = copy.deepcopy(self.report)
        return new_node