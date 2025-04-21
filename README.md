# Browsing Agent

An advanced autonomous research agent with browser automation capabilities for conducting web research tasks.

## Features

- Autonomous web browsing and research capabilities
- Integration with multiple LLM providers (OpenAI, Google, Anthropic, OpenRouter)
- Browser automation for gathering information from websites
- Progress tracking and research reporting
- User-friendly interface for research queries

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Chrome browser installed on your system
- API keys for at least one LLM provider

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/browsing_agent.git
   cd browsing_agent
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   # OpenAI API (for GPT models)
   OPENAI_API_KEY=your_openai_api_key

   # Google API (for Gemini models)
   GEMINI_API_KEY=your_gemini_api_key

   # Anthropic API (for Claude models)
   ANTHROPIC_API_KEY=your_anthropic_api_key

   # OpenRouter API (for access to multiple models)
   OPENROUTER_API_KEY=your_openrouter_api_key

   # Fireworks API (for alternative models)
   FIREWORKS_API_KEY=your_fireworks_api_key
   ```

   You need to include at least one API key for the agent to function.

### API Key Setup Guide

#### OpenAI API Key
1. Go to [OpenAI's platform](https://platform.openai.com/account/api-keys)
2. Create an account or log in
3. Create a new API key
4. Add to your `.env` file as `OPENAI_API_KEY=your_key_here`

#### Google (Gemini) API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or log into your Google account
3. Create an API key
4. Add to your `.env` file as `GEMINI_API_KEY=your_key_here`

#### Anthropic API Key
1. Go to [Anthropic's console](https://console.anthropic.com/)
2. Create an account or log in
3. Create a new API key
4. Add to your `.env` file as `ANTHROPIC_API_KEY=your_key_here`

#### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Create a new API key
4. Add to your `.env` file as `OPENROUTER_API_KEY=your_key_here`

## Usage

### Quick Start Example

Run the included example script to see the agent in action:

```bash
python example.py
```

This will run a sample research task about quantum computing using your configured API keys.

### Using the Web Interface

Start the web UI for more interactive research:

```bash
python main.py
```

This opens a Gradio interface where you can:
1. Select language models for research
2. Enter a research query
3. Set the maximum number of research steps
4. Start the research process and view results

## Troubleshooting

### Common Issues

- **No API keys detected**: Ensure you've added at least one valid API key to your `.env` file
- **Chrome not found**: Make sure Chrome is installed on your system
- **Import errors**: Verify all dependencies are installed using `pip install -r requirements.txt`

### Browser Issues

If Chrome fails to start automatically, you can:
1. Start Chrome manually with remote debugging enabled:
   ```bash
   chrome --remote-debugging-port=9222
   ```
2. Run the agent with the `--use-existing-browser` flag:
   ```bash
   python main.py --use-existing-browser
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
