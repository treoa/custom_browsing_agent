# Advanced Autonomous Research Agent

An intelligent multi-agent research system for conducting deep, methodical web investigations.

## Overview

This project implements the Advanced Autonomous Research Agent framework, a sophisticated multi-agent architecture designed to overcome the limitations of traditional browser-use implementations. The system focuses on conducting thorough, high-quality research across multiple domains while avoiding the common problem of premature task completion.

Key features:

- **Multi-Agent Architecture**: Specialized agents for planning, execution, analysis, critique, and synthesis
- **Advanced Memory Systems**: Short-term, working, long-term, and episodic memory for comprehensive context
- **Sophisticated Planning**: Strategic and tactical planning for thorough research
- **Quality Evaluation**: Rigorous criteria for research completeness and quality
- **Progressive Deepening**: Methodology to ensure depth rather than shallow exploration

## Components

The system consists of these main components:

1. **Agents**:
   - Executive Agent: Coordinates the overall research process
   - Research Agent: Gathers information from web sources
   - Analysis Agent: Processes and analyzes gathered information
   - Critique Agent: Evaluates quality and completeness
   - Synthesis Agent: Compiles findings into coherent outputs

2. **Memory System**:
   - Short-Term Memory: Maintains immediate context
   - Working Memory: Processes current information
   - Long-Term Memory: Stores persistent knowledge
   - Episodic Memory: Records research actions and outcomes

3. **Planning Framework**:
   - Strategic Planner: High-level research goals and strategies
   - Tactical Planner: Concrete, actionable steps

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/browsing_agent.git
   cd browsing_agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### Using the UI

Run the application:
```
python main.py --ui
```

This will launch a Gradio interface where you can:
- Enter a research query
- Select an LLM model
- Configure research parameters
- View research results

### Command Line Usage

Run research from the command line:
```
python main.py --query "Your research query" --model "gemini-2.0-pro" --max-steps 50
```

Parameters:
- `--query`: The research query to investigate
- `--model`: LLM model to use (default: "gemini-2.0-pro")
- `--max-steps`: Maximum number of research steps (default: 50)
- `--storage-path`: Path for storing research data (default: "./research_data")

## Example Queries

Try these example research queries:

- "What are the latest advancements in quantum computing?"
- "Compare renewable energy adoption rates across different countries"
- "How has artificial intelligence impacted healthcare diagnostics?"
- "What are the most effective strategies for mitigating climate change?"

## Implementation Roadmap

This project follows an implementation roadmap with five phases:

1. **Foundation Development** (Current): Core architecture, memory systems, agent framework
2. **Component Enhancement**: Specialized agent functionalities, advanced planning
3. **Integration and Optimization**: Cross-component communication, performance tuning
4. **Testing and Refinement**: Scenario testing, comparison evaluations
5. **Deployment and Scaling**: Production deployment, monitoring, continuous improvement

## License

[MIT License](LICENSE)

## Acknowledgments

- This project builds on the browser-use library for web interaction capabilities
- Inspired by successful multi-agent systems like Manus
