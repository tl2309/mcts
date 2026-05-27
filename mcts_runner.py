import sys
import os, json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pathlib import Path
from typing import Dict, Any
import yaml
import traceback
# from storyteller.algorithm.mcts_solver import DataStorytellingMCTSSolver
from storyteller.algorithm.llm_story_generator import LLMStoryGenerator
from storyteller.algorithm.mcts_node import  MCTSNode, ChartGroup

class DataStorytellingRunner:
    def __init__(self, config_path: str):
        """初始化运行器"""
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 创建日志目录
        self.log_dir = os.path.join(os.path.dirname(config_path), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self):
        """运行 LLM 进行数据故事生成"""
        try:
            print("\n[INFO] 开始运行 DataStorytelling LLM...")
            generator = LLMStoryGenerator(
                original_query=self.config["query"],
                dataset_path=self.config["dataset_path"],
                output_dir=self.config["save_root_dir"],
                data_context=self.config.get("data_context", ""),
                llm_kwargs=self.config.get("llm_kwargs", {})
            )

            # 运行 LLM 生成数据故事
            best_node = generator.generate_story()
            print("\n[SUCCESS] 数据故事 LLM 运行完成！")

            # 打印最终决策路径
            self.print_decision_path(best_node)
            
            # 生成最终报告（使用完整版本的报告生成）
            self.generate_final_report(best_node)

            return best_node

        except Exception as e:
            print(f"[ERROR] 运行过程中出现错误: {e}")
            traceback.print_exc()
            return None

    def print_decision_path(self, final_node: MCTSNode):
        """打印最终的决策路径"""
        print("\n[INFO] **最终决策路径:**")
        print("=" * 50)

        path = []
        current_node = final_node
        while current_node:
            path.append(current_node)
            current_node = current_node.parent_node
        
        # 按照从根节点到最终节点的顺序打印路径
        for step, node in enumerate(reversed(path)):
            if node.parent_action:  # 如果有动作，打印动作
                print(f"\n[STEP] 步骤 {step + 1}: {node.parent_action.description}")
                print(f"   -> 进入状态: {node.node_type.name}")
            else:  # 根节点没有动作，只打印状态
                print(f"\n[STEP] 初始状态: {node.node_type.name}")

        # 打印最终报告的章节信息
        print("\n[INFO] **最终报告章节:**")
        if hasattr(final_node, 'report') and final_node.report and final_node.report.chapters:
            for chapter in final_node.report.chapters:
                print(f"\n[CHAPTER] {chapter.title}")
        else:
            print("   [WARNING] 报告章节列表为空")

        print("\n[INFO] **数据故事生成完成！**")

    def clean_caption(self, caption: str) -> str:
        """清理图表说明文字，移除任何 HTML 标记"""
        if not caption:
            return ""
        
        # 如果包含完整的 HTML 文档，提取有用的文本内容
        if '<!doctype html>' in caption.lower():
            # 返回空字符串，因为这种情况下可能是错误的 caption
            return ""
        
        # 移除任何 HTML 标签
        import re
        clean_text = re.sub(r'<[^>]+>', '', caption)
        return clean_text.strip()

    def generate_markdown_report(self, final_node: MCTSNode) -> str:
        """生成 Markdown 格式的报告"""
        markdown = []
        output_dir = os.path.abspath(self.config["save_root_dir"])
        
        # 1. 报告标题
        markdown.append("# 数据分析报告\n")
        
        # 2. 报告摘要（开头）
        if hasattr(final_node.report, 'key_abstract') and final_node.report.key_abstract:
            markdown.append("## 摘要\n")
            markdown.append(final_node.report.key_abstract + "\n")
        
        # 3. 章节内容
        for chapter in final_node.report.chapters:
            markdown.append(f"\n## {chapter.title}\n")
            
            # 添加图表和说明
            for chart in getattr(chapter, 'charts', []):
                if hasattr(chart, 'caption') and chart.caption:
                    markdown.append(f"\n> {chart.caption}\n")
                if hasattr(chart, 'url') and chart.url:
                    # 规范化图片路径
                    img_path = os.path.relpath(chart.url, output_dir)
                    img_path = img_path.replace('\\', '/')  # 统一使用正斜杠
                    markdown.append(f"\n![{chapter.title}]({img_path})\n")
            
            # 添加章节总结
            if hasattr(chapter, 'summary') and chapter.summary:
                markdown.append("\n### Chapter Summary\n")
                markdown.append(chapter.summary + "\n")
        
        # 4. 报告总结（结尾）
        if hasattr(final_node.report, 'brief_conclusion') and final_node.report.brief_conclusion:
            markdown.append("\n## Conclusion and Suggestions\n")
            markdown.append(final_node.report.brief_conclusion + "\n")
        
        return "\n".join(markdown)

    def generate_html_report(self, markdown_content: str) -> str:
        """生成 HTML 格式的报告"""
        # 1. 生成基本的 HTML 结构
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析报告</title>
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
        // 初始化 markdown-it
        var md = window.markdownit({{
            html: true,
            breaks: true,
            linkify: true
        }});
        
        // 直接使用 markdown 内容
        var content = {json.dumps(markdown_content)};
        document.getElementById('content').innerHTML = md.render(content);

        // 为摘要部分添加 class
        var summarySection = document.querySelector('h2');
        if (summarySection && summarySection.textContent.trim() === '摘要') {{
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
        """生成最终的完整报告"""
        output_dir = self.config["save_root_dir"]
        
        try:
            # 生成 Markdown 报告
            markdown_content = self.generate_markdown_report(final_node)
            
            # 保存 Markdown 报告
            md_path = os.path.join(output_dir, "report.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"\n Markdown 报告已保存到: {md_path}")
            
            # 生成 HTML 报告
            html_content = self.generate_html_report(markdown_content)
            html_path = os.path.join(output_dir, "report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[INFO] HTML 报告已保存到: {html_path}")
            
            print("[SUCCESS] 报告已成功生成！")
        
        except Exception as e:
            print(f"[ERROR] 保存报告时出错: {e}")
            traceback.print_exc()

def run_from_config(config_path: str):
    """从配置文件运行数据故事生成"""
    runner = DataStorytellingRunner(config_path)
    return runner.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcts_runner.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_from_config(config_path)