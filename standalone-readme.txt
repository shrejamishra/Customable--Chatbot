# Standalone Sentiment Analysis Chat

A self-contained chat application with real-time sentiment analysis that requires no external APIs or services.

## Features

- Chat with a rule-based bot that responds based on your message sentiment
- Real-time sentiment analysis using VADER
- Multiple bot personality modes:
  - Standard: Natural conversational responses
  - Educational: Explains the sentiment analysis
  - Therapeutic: Focuses on emotional aspects
  - Mirror: Reflects your message sentiment
- Interactive data visualizations:
  - Line charts
  - Bar charts 
  - Area charts
  - Word clouds
- Export conversation data to CSV or Excel format
- Word frequency analysis

## Requirements

- Python 3.7+
- No external APIs or services needed!

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/standalone-sentiment-chat.git
cd standalone-sentiment-chat
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the sidebar to configure settings:
   - Choose bot personality mode
   - Select visualization type
   - Start/end chat sessions
   - Export data

4. Start chatting and observe real-time sentiment analysis!

## How It Works

The application uses:

- VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
- NLTK for natural language processing
- Streamlit for the web interface
- Plotly and Matplotlib for visualizations
- WordCloud for word frequency visualization

## Extending the Project

- Add more sophisticated response generation
- Implement custom sentiment analysis models
- Create additional visualization types
- Add more bot personality modes
- Implement conversation summarization

## License

MIT
