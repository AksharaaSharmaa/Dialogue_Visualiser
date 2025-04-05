# Discourse Analyzer

<div align="center">
  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-orange.svg)](https://www.python.org/downloads/)

</div>

<div align="center">
  
  <h2>🌿 A Comprehensive Conversation Analysis Toolkit 🍊</h2>
  
</div>

## 🔬 Overview

**Discourse Analyzer** empowers researchers, community managers, and communication specialists to uncover the hidden patterns within discussions. By harmoniously blending argumentation mapping, network visualization, timeline analysis, and topic modeling, this toolkit reveals the intricate tapestry of how conversations unfold.

This prototype was developed specifically for GFOSS as part of my proposal submission. Use the Sample files as dataset. You can view the working demo at: [https://gsoc25-dialogue-visualiser-gfoss-prototype.streamlit.app/](https://gsoc25-dialogue-visualiser-gfoss-prototype.streamlit.app/)

## 🌟 Key Features

### 🍊 Argumentation Analysis
- **Argument Classification**: Automatically categorizes text into different argument types:
  - Claims <span style="color:#5D9C59">■</span>
  - Counterclaims <span style="color:#DF7861">■</span>
  - Evidence <span style="color:#69C47F">■</span>
  - Questions <span style="color:#A76F6F">■</span>
  - Agreements <span style="color:#5D9C59">■</span>
  - Disagreements <span style="color:#DF7861">■</span>
  - Clarifications <span style="color:#FF9B50">■</span>
- **Evidence-to-claim ratio calculation**
- **Controversial claim detection** (claims with both agreements and disagreements)
- **Persuasive user identification**

### 🌿 Network Visualization
- **Interactive user interaction graphs** showing communication patterns
- **Community detection** with colored convex hulls (Louvain method)
- **Weighted connections** based on interaction frequency
- **Sentiment-colored edges** (orange: negative, green: positive, gray: neutral)
- **Dynamic node sizing** based on:
  - Message count (activity level)
  - Centrality (importance as a connector)
  - Influence (eigenvector centrality)
- **Curved Bezier edges** with directional arrows

### 🍊 Conversation Timeline
- **User activity tracking** on separate horizontal lines
- **Message type differentiation** (questions, agreements, disagreements, emotional peaks)
- **Topic shift detection** with vertical markers
- **Time gap visualization** to segment conversations
- **Interactive tooltips** with message details

### 🌿 Topic Analysis
- **Automated topic extraction** using NMF on TF-IDF vectors
- **Topic evolution tracking** over time using sliding windows
- **Topic relationship visualization**
- **Key phrase extraction** from conversation content
- **User topic engagement analysis**

### 🍊 Sentiment Analysis
- **Emotional tone tracking** throughout conversations
- **Sentiment trend identification** and annotation
- **Peak emotion detection**
- **Positive/negative region visualization**

### 🌿 Advanced Analytics
- **Echo chamber detection** and cross-community bridge identification
- **Centrality metrics** (betweenness and eigenvector)
- **Content-based clustering** using TF-IDF and K-means
- **User participation statistics** and behavior analysis
- **Network-level metrics** quantifying overall structure

## 📊 Visualization Panels

<div align="center">
  <table>
    <tr>
      <td align="center"><span style="color:#FF9B50">⬢</span> <b>Argumentation Graph</b><br><i>Logical structure of discussions</i></td>
      <td align="center"><span style="color:#5D9C59">⬢</span> <b>Network View</b><br><i>User interaction patterns</i></td>
    </tr>
    <tr>
      <td align="center"><span style="color:#DF7861">⬢</span> <b>Conversation Timeline</b><br><i>Message flow over time</i></td>
      <td align="center"><span style="color:#69C47F">⬢</span> <b>Topic Flow</b><br><i>Evolution of discussion themes</i></td>
    </tr>
    <tr>
      <td align="center" colspan="2"><span style="color:#A76F6F">⬢</span> <b>Sentiment Analysis</b><br><i>Emotional tone throughout discussions</i></td>
    </tr>
  </table>
</div>

## 🔧 Tech Stack

<div align="center">
  <table>
    <tr>
      <th colspan="2" style="background-color: #FF9B50; color: white;">Core Libraries</th>
    </tr>
    <tr>
      <td><b>pandas</b></td>
      <td>Data manipulation and management</td>
    </tr>
    <tr>
      <td><b>NetworkX</b></td>
      <td>Graph creation and analysis</td>
    </tr>
    <tr>
      <td><b>Plotly</b></td>
      <td>Interactive visualization</td>
    </tr>
    <tr>
      <td><b>NLTK</b></td>
      <td>Natural language processing</td>
    </tr>
    <tr>
      <td><b>scikit-learn</b></td>
      <td>Machine learning (TF-IDF, NMF, K-means)</td>
    </tr>
    <tr>
      <td><b>NumPy</b></td>
      <td>Numerical operations</td>
    </tr>
    <tr>
      <td><b>Streamlit</b></td>
      <td>Web application framework</td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <tr>
      <th colspan="2" style="background-color: #5D9C59; color: white;">Analysis Components</th>
    </tr>
    <tr>
      <td><b>Python-Louvain</b></td>
      <td>Community detection</td>
    </tr>
    <tr>
      <td><b>SciPy</b></td>
      <td>Convex hull creation for community visualization</td>
    </tr>
    <tr>
      <td><b>Collections module</b></td>
      <td>Efficient interaction tracking</td>
    </tr>
  </table>
</div>

## 📈 Output Formats

<div align="center">
<table>
  <tr>
    <td align="center" style="background-color: #FF9B50; color: white;"><b>Interactive Visualization</b></td>
    <td>Multi-panel Plotly dashboard with hover tooltips</td>
  </tr>
  <tr>
    <td align="center" style="background-color: #5D9C59; color: white;"><b>Metrics Dictionary</b></td>
    <td>Structured data containing all calculated metrics</td>
  </tr>
  <tr>
    <td align="center" style="background-color: #FF9B50; color: white;"><b>Text Summary</b></td>
    <td>Markdown-formatted report summarizing key findings</td>
  </tr>
</table>
</div>

## 🔍 Data Sources

<div align="center">
  <table>
    <tr>
      <td><span style="color:#FF9B50">●</span> <b>ConvoKit</b></td>
      <td><span style="color:#5D9C59">●</span> <b>YouTube discussions</b></td>
      <td><span style="color:#FF9B50">●</span> <b>Text discussions</b></td>
    </tr>
    <tr>
      <td><span style="color:#5D9C59">●</span> <b>Reddit threads</b></td>
      <td><span style="color:#FF9B50">●</span> <b>Website content</b></td>
      <td><span style="color:#5D9C59">●</span> <b>Forum discussions</b></td>
    </tr>
  </table>
</div>

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

<div align="center">
  <table>
    <tr>
      <th colspan="2" style="background-color: #5D9C59; color: white;">Required Data Columns</th>
    </tr>
    <tr>
      <td><b>user_id</b></td>
      <td>Identifier for the message sender</td>
    </tr>
    <tr>
      <td><b>timestamp</b></td>
      <td>When the message was sent</td>
    </tr>
    <tr>
      <td><b>message</b></td>
      <td>The text content of the message</td>
    </tr>
    <tr>
      <th colspan="2" style="background-color: #FF9B50; color: white;">Optional Data Columns</th>
    </tr>
    <tr>
      <td><b>reply_to</b></td>
      <td>ID of the message being replied to</td>
    </tr>
    <tr>
      <td><b>message_id</b></td>
      <td>Unique identifier for each message</td>
    </tr>
  </table>
</div>

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

<div align="center">
  <img src="https://img.shields.io/badge/PRs-welcome-orange.svg" alt="PRs Welcome" />
</div>

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

<div align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
</div>

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

<div align="center">
  <p>Project developed as a prototype for GFOSS as part of my GSoC 2025 proposal.</p>
</div>

---

<div align="center">
  <p style="color:#FF9B50">Made with</p>
  <p style="color:#5D9C59">❤️</p>
  <p style="color:#FF9B50">for conversation analysts and researchers</p>
</div>
