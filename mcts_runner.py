import sys
import os, json
from datetime import datetime

# Add project root directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pathlib import Path
from typing import Dict, Any
import yaml
import traceback
from storyteller.algorithm.mcts_solver import DataStorytellingMCTSSolver
from storyteller.algorithm.mcts_node import  MCTSNode, ChartGroup

class DataStorytellingRunner:
    def __init__(self, config_path: str):
        """Initialize runner"""
        # Load configuration file
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Create log directory
        self.log_dir = os.path.join(os.path.dirname(config_path), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self):
        """Run MCTS for data story generation"""
        try:
            print("\n🚀 Starting DataStorytelling MCTS...")
            solver = DataStorytellingMCTSSolver(
                original_query=self.config["query"],
                dataset_path=self.config["dataset_path"],
                output_dir=self.config["save_root_dir"],
                max_iterations=self.config["max_iterations"],
                max_depth=self.config["max_depth"],
                exploration_constant=self.config["exploration_constant"],
                data_context=self.config.get("data_context", ""),
                llm_kwargs=self.config.get("llm_kwargs", {})
            )

            # Run MCTS to generate data story
            best_node = solver.solve()
            print("\n✅ Data Story MCTS completed!")

            # 🔍 Print final decision path
            self.print_decision_path(best_node)
            
            # Generate final report (using full report generation)
            self.generate_final_report(best_node)

            return best_node

        except Exception as e:
            print(f"❌ Error occurred during execution: {e}")
            traceback.print_exc()
            return None

    def print_decision_path(self, final_node: MCTSNode):
        """Print the final decision path"""
        print("\n📖 **Final Decision Path:**")
        print("=" * 50)

        path = []
        current_node = final_node
        while current_node:
            path.append(current_node)
            current_node = current_node.parent_node
        
        # Print path from root to final node
        for step, node in enumerate(reversed(path)):
            if node.parent_action:  # If there is an action, print it
                print(f"\n🔹 Step {step + 1}: {node.parent_action.description}")
                print(f"   ⤷ Entering state: {node.node_type.name}")
            else:  # Root node has no action, only print state
                print(f"\n🔹 Initial state: {node.node_type.name}")

        # Print final report chapters
        print("\n📊 **Final Report Chapters:**")
        if hasattr(final_node, 'report') and final_node.report and final_node.report.chapters:
            for chapter in final_node.report.chapters:
                print(f"\n📌 {chapter.title}")
        else:
            print("   ⚠️ Warning: Report chapter list is empty")

        print("\n📊 **Data Story Generation Complete!**")

    def clean_caption(self, caption: str) -> str:
        """Clean chart caption text by removing any HTML tags"""
        if not caption:
            return ""
        
        # If contains complete HTML document, extract useful text content
        if '<!doctype html>' in caption.lower():
            # Return empty string as this may be an error caption
            return ""
        
        # Remove any HTML tags
        import re
        clean_text = re.sub(r'<[^>]+>', '', caption)
        return clean_text.strip()

    def generate_markdown_report(self, final_node: MCTSNode) -> str:
        """Generate Markdown format report"""
        markdown = []
        output_dir = os.path.abspath(self.config["save_root_dir"])
        
        # 1. Report title
        markdown.append("# Data Analysis Report\n")
        
        # 2. Report abstract (beginning)
        if hasattr(final_node.report, 'key_abstract') and final_node.report.key_abstract:
            markdown.append("## Abstract\n")
            markdown.append(final_node.report.key_abstract + "\n")
        
        # 3. Chapter content
        for chapter in final_node.report.chapters:
            markdown.append(f"\n## {chapter.title}\n")
            
            # Add charts and descriptions
            for chart in getattr(chapter, 'charts', []):
                if hasattr(chart, 'caption') and chart.caption:
                    markdown.append(f"\n> {chart.caption}\n")
                if hasattr(chart, 'url') and chart.url:
                    # Normalize image path
                    img_path = os.path.relpath(chart.url, output_dir)
                    img_path = img_path.replace('\\', '/')  # Use forward slashes consistently
                    markdown.append(f"\n![{chapter.title}]({img_path})\n")
            
            # Add chapter summary
            if hasattr(chapter, 'summary') and chapter.summary:
                markdown.append("\n### Chapter Summary\n")
                markdown.append(chapter.summary + "\n")
        
        # 4. Report conclusion (ending)
        if hasattr(final_node.report, 'brief_conclusion') and final_node.report.brief_conclusion:
            markdown.append("\n## Conclusion and Suggestions\n")
            markdown.append(final_node.report.brief_conclusion + "\n")
        
        return "\n".join(markdown)

    def generate_html_report(self, markdown_content: str) -> str:
        """Generate HTML format report"""
        # 1. Generate basic HTML structure
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/markdown-it@13.0.1/dist/markdown-it.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
            text-align: left;
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 2px solid #eee; 
            padding-bottom: 0.3em;
            text-align: center;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 2em;
            text-align: left;
        }}
        h3 {{ 
            color: #455a64;
            text-align: left;
        }}
        p {{ 
            margin: 1em 0;
            text-indent: 2em;
        }}
        img {{ 
            max-width: 50%;
            height: auto; 
            display: block; 
            margin: 2em auto;
            border: 1px solid #eee;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        em {{ 
            display: block;
            text-align: left;
            color: #666;
            margin: 1em 0;
            font-style: italic;
            text-indent: 2em;
            max-width: 100%;
        }}
        .chart-caption {{ 
            text-align: left;
            color: #666;
            margin: 1em 0;
            max-width: 100%;
            text-indent: 2em;
        }}
        blockquote {{
            border-left: 4px solid #42b983;
            margin: 1em 0;
            padding: 0.5em 1em;
            color: #666;
            background: #f8f8f8;
        }}
        ul, ol {{
            padding-left: 2em;
            margin: 1em 0;
        }}
        li {{
            margin: 0.5em 0;
            text-align: left;
            text-indent: 0;
        }}
        .summary p {{
            text-indent: 0;
        }}
    </style>
</head>
<body>
    <div id="content"></div>
    <script>
        // Initialize markdown-it
        var md = window.markdownit({{
            html: true,
            breaks: true,
            linkify: true
        }});
        
        // Use markdown content directly
        var content = {json.dumps(markdown_content)};
        document.getElementById('content').innerHTML = md.render(content);

        // Add class to summary section
        var summarySection = document.querySelector('h2');
        if (summarySection && summarySection.textContent.trim() === 'Abstract') {{
            var summaryDiv = document.createElement('div');
            summaryDiv.className = 'summary';
            var nextElement = summarySection.nextElementSibling;
            while (nextElement && nextElement.tagName !== 'H2') {{
                var clone = nextElement.cloneNode(true);
                summaryDiv.appendChild(clone);
                nextElement = nextElement.nextElementSibling;
            }}
            summarySection.parentNode.insertBefore(summaryDiv, summarySection.nextSibling);
        }}
    </script>
</body>
</html>"""

        return html

    def generate_final_report(self, final_node: MCTSNode):
        """Generate final complete report"""
        output_dir = self.config["save_root_dir"]
        
        try:
            # Find best iteration
            history_dir = os.path.join(output_dir, "iterations")
            best_iteration = None
            best_score = -float('inf')
            
            for d in os.listdir(history_dir):
                if not d.startswith("iteration_"):
                    continue
                
                score_file = os.path.join(history_dir, d, "score.json")
                if os.path.exists(score_file):
                    with open(score_file, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                        if score_data["score"] > best_score:
                            best_score = score_data["score"]
                            best_iteration = d
            
            if not best_iteration:
                print("⚠️ No valid iteration results found")
                return
            
            # Find best iteration directory
            best_iteration_dir = os.path.join(history_dir, best_iteration)
            
            # Copy best iteration report
            best_report_path = os.path.join(best_iteration_dir, "report.md")
            if os.path.exists(best_report_path):
                with open(best_report_path, 'r', encoding='utf-8') as f:
                    best_report_content = f.read()
                
                # Fix image path references
                best_report_content = best_report_content.replace(
                    "](charts/", 
                    f"]({os.path.join('iterations', best_iteration, 'charts')}/"
                )
                
                # Save final report
                with open(os.path.join(output_dir, "report.md"), 'w', encoding='utf-8') as f:
                    f.write(best_report_content)
                print(f"\n📝 Markdown report saved to: {os.path.join(output_dir, 'report.md')}")
                
                # Generate HTML report
                html_content = self.generate_html_report(best_report_content)
                with open(os.path.join(output_dir, "report.html"), 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"📊 HTML report saved to: {os.path.join(output_dir, 'report.html')}")
                
                print(f"✨ Best report from iteration {best_iteration}, score: {best_score:.2f}")
        
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            traceback.print_exc()

def run_from_config(config_path: str):
    """Run data story generation from config file"""
    runner = DataStorytellingRunner(config_path)
    return runner.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcts_runner.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_from_config(config_path)