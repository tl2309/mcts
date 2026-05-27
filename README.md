# 📊 DataStorytelling MCTS

> 基于蒙特卡洛树搜索（MCTS）的智能数据故事生成系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 🎯 项目简介

DataStorytelling MCTS 是一个创新的数据分析和可视化系统，利用蒙特卡洛树搜索算法自动生成高质量的数据故事报告。系统能够智能地分析数据集，生成相关图表，并构建连贯的数据叙述，最终输出专业的分析报告。

### ✨ 主要特性

- 🤖 **智能数据分析**: 基于 MCTS 算法的自动化数据探索
- 📈 **自动图表生成**: 智能选择最适合的可视化方式
- 📝 **故事化叙述**: 将数据分析结果转化为连贯的故事
- 🎨 **多格式输出**: 支持 Markdown 和 HTML 格式报告
- 🔧 **灵活配置**: 可自定义分析参数和输出格式
- 🌐 **多模型支持**: 支持多种 LLM 模型（GPT、Gemini 等）

## 🚀 快速开始

### 系统要求

- Python 3.8+
- 支持的操作系统：Windows、macOS、Linux

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd storyteller

# 安装依赖（推荐使用虚拟环境）
pip install -r requirements.txt
```

### 基本使用

1. **准备数据集**
   ```bash
   # 将您的 CSV 数据文件放置在 dataset/ 目录下
   cp your_data.csv storyteller/dataset/

   # 使用generate_data_context.py对数据集生成相应的数据集上下文data_context.json
   ```

2.  ## ⚙️ 配置说明

    ### 主配置文件 (`config/config.yaml`)

    ```yaml
    # 分析查询
    query: "Employment trends across U.S. sectors since 2006"

    # 数据集路径
    dataset_path: "storyteller/dataset/us-employment.csv"

    # 输出目录
    save_root_dir: "storyteller/output"

    # MCTS 参数
    max_iterations: 5          # 最大迭代次数
    max_depth: 35             # 最大搜索深度
    exploration_constant: 1.4  # 探索常数

    # 数据上下文(你需要将所要使用的数据集上下文复制到data_context.json中)  
    data_context: "storyteller/dataset/data_context.json"

    # LLM 配置
    llm_kwargs:
    temperature: 0.7
    model: "gemini-2.0-flash"
    base_url: "https://your-api-endpoint.com/v1"
    api_key: "your-api-key"

    # 历史记录配置
    history:
    save_iterations: true
    save_dir: "iterations"
    ```
3. **运行分析**
   
   **方法 1: 一键运行（推荐）**
   ```bash
   # 设置环境变量并运行系统
   export OPENAI_API_KEY="your-api-key" && export OPENAI_BASE_URL="https://your-api-endpoint.com/v1" && python storyteller/runner/mcts_runner.py storyteller/config/config.yaml
   ```
   
   **方法 2: 分步运行**
   ```bash
   # 设置环境变量
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_BASE_URL="your-api-base-url"
   
   # 运行系统
   python storyteller/runner/mcts_runner.py storyteller/config/config.yaml
   ```
   
   **方法 3: 使用简化脚本**
   ```bash
   # 使用项目根目录的运行脚本
   python run_storyteller.py
   ```

4. **查看结果**
   ```bash
   # 结果将保存在 storyteller/output/ 目录下
   open storyteller/output/iterations/iteration_*/report.html
   ```

### 🔑 环境变量说明

| 环境变量 | 说明 | 示例值 |
|---------|------|--------|
| `OPENAI_API_KEY` | LLM API 密钥 | `sk-xxx...` |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |

**注意**: 
- 🔐 请将示例中的 API 密钥替换为您自己的密钥
- 🌐 根据您使用的 LLM 服务商调整 BASE_URL
- ⚙️ 环境变量设置会覆盖配置文件中的相应设置

## 📁 项目结构

```
storyteller/
├── algorithm/              # 核心算法模块
│   ├── mcts_solver.py     # MCTS 求解器
│   ├── mcts_node.py       # MCTS 节点定义
│   ├── mcts_action.py     # MCTS 动作定义
│   ├── evaluator.py       # 评估器
│   ├── reward.py          # 奖励函数
│   └── utils/             # 工具函数
├── config/                 # 配置文件
│   └── config.yaml        # 主配置文件
├── dataset/               # 数据集目录
│   ├── us-employment.csv  # 示例数据集
│   └── data_context.json  # 数据上下文
├── llm_call/              # LLM 调用模块
├── output/                # 输出结果目录
│   └── iterations/        # 迭代结果
├── runner/                # 运行器模块
│   ├── mcts_runner.py     # 主运行器
│   └── visualization_task.py # 可视化任务
├── templates/             # 模板文件
└── README.md              # 项目文档
```


### 重要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_iterations` | MCTS 最大迭代次数 | 5 |
| `max_depth` | 搜索树最大深度 | 35 |
| `exploration_constant` | UCB1 探索常数 | 1.4 |
| `temperature` | LLM 生成温度 | 0.7 |


## 🎨 输出格式

系统支持多种输出格式：

### 1. Markdown 报告
- 结构化的文本报告
- 包含图表引用和说明
- 适合版本控制和协作

### 2. HTML 报告
- 美观的网页格式
- 内嵌图表和交互元素
- 适合演示和分享

### 3. 图表文件
- PNG 格式的可视化图表
- 高质量输出
- 可独立使用


## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**: [您的姓名]
- **邮箱**: your.email@example.com
- **问题反馈**: [GitHub Issues](https://github.com/your-username/storyteller/issues)

## 🙏 致谢

- 感谢所有贡献者的支持
- 特别感谢开源社区提供的优秀工具和库
---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！ 