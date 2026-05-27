# 📊 DataStorytelling MCTS

> Intelligent Data Story Generation System Based on Monte Carlo Tree Search (MCTS)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 🎯 Project Overview

DataStorytelling MCTS is an innovative data analysis and visualization system that uses Monte Carlo Tree Search algorithm to automatically generate high-quality data story reports. The system can intelligently analyze datasets, generate relevant charts, and build coherent data narratives, ultimately outputting professional analysis reports.

### ✨ Key Features

- 🤖 **Intelligent Data Analysis**: Automated data exploration based on MCTS algorithm
- 📈 **Automatic Chart Generation**: Intelligent selection of the most suitable visualization methods
- 📝 **Story-driven Narrative**: Transform data analysis results into coherent stories
- 🎨 **Multi-format Output**: Support for Markdown and HTML report formats
- 🔧 **Flexible Configuration**: Customizable analysis parameters and output formats
- 🌐 **Multi-model Support**: Support for various LLM models (GPT, Gemini, etc.)

## 🚀 Quick Start

### System Requirements

- Python 3.8+
- Supported operating systems: Windows, macOS, Linux

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd storyteller

# Install dependencies (recommended to use virtual environment)
pip install -r requirements.txt
```

### Basic Usage

1. **Prepare Dataset**
   ```bash
   # Place your CSV data file in the dataset/ directory
   cp your_data.csv storyteller/dataset/

   # Use generate_data_context.py to generate the corresponding dataset context data_context.json
   ```

2. ## ⚙️ Configuration

    ### Main Config File (`config/config.yaml`)

    ```yaml
    # Analysis query
    query: "Employment trends across U.S. sectors since 2006"

    # Dataset path
    dataset_path: "storyteller/dataset/us-employment.csv"

    # Output directory
    save_root_dir: "storyteller/output"

    # MCTS parameters
    max_iterations: 5          # Maximum iterations
    max_depth: 35             # Maximum search depth
    exploration_constant: 1.4  # Exploration constant

    # Data context (you need to copy the dataset context you want to use into data_context.json)  
    data_context: "storyteller/dataset/data_context.json"

    # LLM configuration
    llm_kwargs:
    temperature: 0.7
    model: "gemini-2.0-flash"
    base_url: "https://your-api-endpoint.com/v1"
    api_key: "your-api-key"

    # History configuration
    history:
    save_iterations: true
    save_dir: "iterations"
    ```
3. **Run Analysis**
   
   **Method 1: One-click run (Recommended)**
   ```bash
   # Set environment variables and run the system
   export OPENAI_API_KEY="your-api-key" && export OPENAI_BASE_URL="https://your-api-endpoint.com/v1" && python storyteller/runner/mcts_runner.py storyteller/config/config.yaml
   ```
   
   **Method 2: Step by step**
   ```bash
   # Set environment variables
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_BASE_URL="your-api-base-url"
   
   # Run the system
   python storyteller/runner/mcts_runner.py storyteller/config/config.yaml
   ```
   
   **Method 3: Use simplified script**
   ```bash
   # Use the run script in project root directory
   python run_storyteller.py
   ```

4. **View Results**
   ```bash
   # Results will be saved in storyteller/output/ directory
   open storyteller/output/iterations/iteration_*/report.html
   ```

### 🔑 Environment Variables

| Variable | Description | Example Value |
|---------|------|--------|
| `OPENAI_API_KEY` | LLM API key | `sk-xxx...` |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |

**Note**: 
- 🔐 Please replace the example API key with your own
- 🌐 Adjust BASE_URL according to your LLM service provider
- ⚙️ Environment variable settings will override the corresponding settings in the config file

## 📁 Project Structure

```
storyteller/
├── algorithm/              # Core algorithm module
│   ├── mcts_solver.py     # MCTS solver
│   ├── mcts_node.py       # MCTS node definition
│   ├── mcts_action.py     # MCTS action definition
│   ├── evaluator.py       # Evaluator
│   ├── reward.py          # Reward function
│   └── utils/             # Utility functions
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration file
├── dataset/               # Dataset directory
│   ├── us-employment.csv  # Sample dataset
│   └── data_context.json  # Data context
├── llm_call/              # LLM call module
├── output/                # Output results directory
│   └── iterations/        # Iteration results
├── runner/                # Runner module
│   ├── mcts_runner.py     # Main runner
│   └── visualization_task.py # Visualization task
├── templates/             # Template files
└── README.md              # Project documentation
```


### Important Parameters

| Parameter | Description | Default Value |
|------|------|--------|
| `max_iterations` | MCTS maximum iterations | 5 |
| `max_depth` | Maximum search tree depth | 35 |
| `exploration_constant` | UCB1 exploration constant | 1.4 |
| `temperature` | LLM generation temperature | 0.7 |


## 🎨 Output Format

The system supports multiple output formats:

### 1. Markdown Report
- Structured text report
- Contains chart references and descriptions
- Suitable for version control and collaboration

### 2. HTML Report
- Beautiful web format
- Embedded charts and interactive elements
- Suitable for presentations and sharing

### 3. Chart Files
- PNG format visualization charts
- High-quality output
- Can be used independently


## 📄 License

This project uses MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Maintainer**: [Your Name]
- **Email**: your.email@example.com
- **Issue Reporting**: [GitHub Issues](https://github.com/your-username/storyteller/issues)

## 🙏 Acknowledgments

- Thanks to all contributors for their support
- Special thanks to the open source community for providing excellent tools and libraries
---

⭐ If this project is helpful to you, please give us a Star! 