# Discourse Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Discourse Analyzer is a comprehensive conversation analysis toolkit that helps researchers, community managers, and communication specialists understand the dynamics, structure, and content of discussions. By combining argumentation mapping, network visualization, timeline analysis, and topic modeling, this tool provides unprecedented insights into how conversations unfold.

This prototype was developed specifically for GFOSS as part of my proposal submission. You can view the working demo at: https://gsoc25-dialogue-visualiser-gfoss-prototype.streamlit.app/

![Dashboard Preview](https://via.placeholder.com/800x450)

## 🌟 Key Features

### Argumentation Analysis
- **Argument Classification**: Automatically categorizes text into different argument types:
  - Claims (blue)
  - Counterclaims (red)
  - Evidence (green)
  - Questions (purple)
  - Agreements (cyan)
  - Disagreements (pink)
  - Clarifications (orange)
- **Evidence-to-claim ratio calculation**
- **Controversial claim detection** (claims with both agreements and disagreements)
- **Persuasive user identification**

### Network Visualization
- **Interactive user interaction graphs** showing communication patterns
- **Community detection** with colored convex hulls (Louvain method)
- **Weighted connections** based on interaction frequency
- **Sentiment-colored edges** (red: negative, blue: positive, gray: neutral)
- **Dynamic node sizing** based on:
  - Message count (activity level)
  - Centrality (importance as a connector)
  - Influence (eigenvector centrality)
- **Curved Bezier edges** with directional arrows

### Conversation Timeline
- **User activity tracking** on separate horizontal lines
- **Message type differentiation** (questions, agreements, disagreements, emotional peaks)
- **Topic shift detection** with vertical markers
- **Time gap visualization** to segment conversations
- **Interactive tooltips** with message details

### Topic Analysis
- **Automated topic extraction** using NMF on TF-IDF vectors
- **Topic evolution tracking** over time using sliding windows
- **Topic relationship visualization**
- **Key phrase extraction** from conversation content
- **User topic engagement analysis**

### Sentiment Analysis
- **Emotional tone tracking** throughout conversations
- **Sentiment trend identification** and annotation
- **Peak emotion detection**
- **Positive/negative region visualization**

### Advanced Analytics
- **Echo chamber detection** and cross-community bridge identification
- **Centrality metrics** (betweenness and eigenvector)
- **Content-based clustering** using TF-IDF and K-means
- **User participation statistics** and behavior analysis
- **Network-level metrics** quantifying overall structure

## 📊 Visualization Panels

The tool generates a multi-panel interactive dashboard:

1. **Argumentation Graph**: Shows the logical structure of discussions
2. **Network View**: Displays user interaction patterns and communities
3. **Conversation Timeline**: Visualizes message flow and types over time
4. **Topic Flow**: Tracks how conversation topics evolve and relate
5. **Sentiment Analysis**: Charts emotional tone throughout discussions

## 🔧 Tech Stack

### Core Libraries
- **pandas**: Data manipulation and management
- **NetworkX**: Graph creation and analysis
- **Plotly**: Interactive visualization
- **NLTK**: Natural language processing
- **scikit-learn**: Machine learning (TF-IDF, NMF, K-means)
- **NumPy**: Numerical operations
- **Streamlit**: Web application framework

### Analysis Components
- **Python-Louvain**: Community detection
- **SciPy**: Convex hull creation for community visualization
- **Collections module**: Efficient interaction tracking

## 📈 Output Formats

The tool generates three types of output:
1. **Interactive Visualization**: Multi-panel Plotly dashboard with hover tooltips
2. **Metrics Dictionary**: Structured data containing all calculated metrics
3. **Text Summary**: Markdown-formatted report summarizing key findings

## 🔍 Data Sources

This GFOSS prototype can analyze conversation data from multiple sources:
- **ConvoKit**: Cornell Conversational Analysis Toolkit datasets
- **YouTube discussions**: Comment threads from video discussions
- **Text discussions**: Plain text conversation logs
- **Reddit threads**: Posts and comment chains
- **Website content**: Article text with associated comments
- **Forum discussions**: Threaded conversation data

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/discourse-analyzer.git
cd discourse-analyzer

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
import pandas as pd
from discourse_analyzer import DiscourseAnalyzer

# Load your conversation data
data = pd.read_csv("your_conversation_data.csv")

# Initialize the analyzer
analyzer = DiscourseAnalyzer(data)

# Generate the full analysis dashboard
dashboard = analyzer.generate_dashboard()

# Save or display the dashboard
dashboard.write_html("conversation_analysis.html")

# Get the summary report
summary = analyzer.generate_summary()
print(summary)
```

### Starting the Web Interface

```bash
streamlit run app.py
```

## 📋 Data Format

Your input data should be a CSV or DataFrame with these columns:
- `user_id`: Identifier for the message sender
- `timestamp`: When the message was sent
- `message`: The text content of the message
- `reply_to` (optional): ID of the message being replied to
- `message_id` (optional): Unique identifier for each message

## 📚 Example

```python
from discourse_analyzer import DiscourseAnalyzer
import pandas as pd

# Sample data
data = pd.DataFrame({
    'user_id': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
    'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:05:00', 
                  '2023-01-01 10:10:00', '2023-01-01 10:15:00', 
                  '2023-01-01 10:20:00'],
    'message': ["I think we should use Python for this project.",
                "I disagree, JavaScript would be better because of its ecosystem.",
                "I agree with Alice. Python has better data science libraries.",
                "We could use Python's pandas library for data processing.",
                "Fair point about pandas, but JavaScript has great visualization libraries."],
    'reply_to': [None, 0, 0, 2, 3]
})

# Analyze the conversation
analyzer = DiscourseAnalyzer(data)
analyzer.analyze()
analyzer.generate_dashboard().write_html("example_analysis.html")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Project developed as a prototype for GFOSS as part of my GSoC 2025 proposal.

---

Made with ❤️ for conversation analysts and researchers.
