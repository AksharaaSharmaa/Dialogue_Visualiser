import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime
import praw
import trafilatura
import io
import zipfile
import yt_dlp
import srt
import requests
import pandas as pd
import json
from datetime import datetime
from youtube_comment_downloader import YoutubeCommentDownloader
import defusedxml
import openpyxl
from Network_Visualisation import create_network_visualization
from Timeline_Visualisation import create_timeline_visualization
from Topic_Visualisation import create_topic_trees
import traceback
from Argumentation_Visualisation import create_argumentation_graph

# Enhanced CSS for a more beautiful and polished UI
st.markdown("""
<style>
    /* Main theme colors - refined palette */
    :root {
        --primary-color: #FF7A00;
        --primary-dark: #E36D00;
        --secondary-color: #FF9E44;
        --background-color: #FFFFFF;
        --text-color: #333333;
        --accent-color: #FFE0C2;
        --light-accent: #FFF5EB;
        --shadow-sm: 0 4px 6px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 8px 15px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 12px 28px rgba(0, 0, 0, 0.12);
        --gradient-primary: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%);
    }
    
    /* Improved typography */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.01em;
    }
    
    /* Main content styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Header styling */
    .stApp header {
        background-color: var(--primary-color);
    }
    
    /* Button styling with enhanced design */
    .stButton>button {
        background: var(--gradient-primary);
        color: white;
        border-radius: 30px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: var(--shadow-md);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, #FF8C1A 0%, #FFAB5E 100%);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Enhanced card styling */
    .feature-card {
        background-color: white;
        border-radius: 16px;
        padding: 28px;
        box-shadow: var(--shadow-md);
        height: 100%;
        border-top: 4px solid var(--primary-color);
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        overflow: hidden;
        position: relative;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .feature-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255, 122, 0, 0.05) 0%, rgba(255, 255, 255, 0) 60%);
        z-index: 1;
        pointer-events: none;
    }
    
    /* Hero section styling */
    .hero-section {
        background: var(--gradient-primary);
        border-radius: 20px;
        padding: 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 60%);
        transform: rotate(30deg);
        z-index: 1;
        pointer-events: none;
    }
    
    /* Improved heading styling */
    h1 {
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #FF7A00, #FFA94D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8rem;
        margin-bottom: 0.2em;
        line-height: 1.1;
    }
    
    .hero-section h2, .hero-section h3 {
        color: white !important;
        font-weight: 700;
        margin-top: 0;
    }
    
    h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Animation enhancements */
    @keyframes float {
        0% {transform: translateY(0px);}
        50% {transform: translateY(-8px);}
        100% {transform: translateY(0px);}
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }
    
    .pulse {
        animation: pulse 4s ease-in-out infinite;
    }
    
    /* Feature icon styling */
    .feature-icon {
        background-color: var(--light-accent);
        width: 60px;
        height: 60px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        font-size: 24px;
        color: var(--primary-color);
        box-shadow: 0 4px 10px rgba(255, 122, 0, 0.15);
    }
    
    /* Enhanced stats card */
    .stats-card {
        background: white;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: var(--shadow-md);
        border-left: 5px solid var(--primary-color);
    }
    
    .stat-number {
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 5px;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Call to action section */
    .cta-section {
        background: linear-gradient(135deg, #FFF5EB 0%, #FFFFFF 100%);
        border-radius: 16px;
        padding: 50px 30px;
        text-align: center;
        margin: 50px 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(255, 122, 0, 0.1);
    }
    
    .cta-button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 16px 40px;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(255, 122, 0, 0.3);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-top: 20px;
        display: inline-block;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(255, 122, 0, 0.4);
    }
    
    /* Enhanced footer */
    .footer {
        background-color: var(--light-accent);
        padding: 60px 0 40px 0;
        margin-top: 80px;
        border-radius: 20px 20px 0 0;
    }
    
    .social-icon {
        background-color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-sm);
        transition: transform 0.3s ease;
        margin: 0 8px;
    }
    
    .social-icon:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    
    .email-input {
        padding: 12px 20px;
        border-radius: 30px;
        border: 1px solid rgba(255, 122, 0, 0.2);
        width: 220px;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .email-input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(255, 122, 0, 0.1);
        outline: none;
    }
    
    .submit-button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .submit-button:hover {
        background-color: var(--primary-dark);
    }
    
    /* Background styling */
    .gradient-bg {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFFFFF 100%);
        min-height: 100vh;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
    }
    
    /* Container for better spacing */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 30px;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom divider */
    .divider {
        height: 4px;
        width: 80px;
        background: var(--gradient-primary);
        border-radius: 2px;
        margin: 20px 0;
    }
    
    /* Center divider */
    .divider-center {
        margin: 20px auto;
    }
</style>
""", unsafe_allow_html=True)

def apply_custom_css():
    st.markdown("""
        <style>
/* Enhanced Streamlit Horizontal Tabs - Premium Orange & Green Theme */

/* Main tab container styling - full width and horizontal alignment */
div.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
    border-radius: 0;
    padding: 12px 12px 0 12px;
    border: none;
    box-shadow: none;
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
    display: flex;
    flex-wrap: nowrap;
    justify-content: space-between;
    align-items: center;
    overflow-x: auto;
    scrollbar-width: thin;
    -ms-overflow-style: none;
}

/* Hide scrollbar for cleaner look but maintain functionality */
div.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 4px;
    background: transparent;
}

div.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: rgba(255, 153, 51, 0.3);
    border-radius: 10px;
}

/* Individual tab styling - equal width distribution */
div.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 14px 24px;
    font-weight: 600;
    background-color: rgba(248, 248, 248, 0.9);
    border: 1px solid #e0e0e0;
    transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    font-size: 15px;
    letter-spacing: 0.3px;
    flex: 1 1 0;
    min-width: fit-content;
    margin: 0 4px;
    color: #444444;
    position: relative;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.8);
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Enhanced sliding animation for tabs */
div.stTabs [data-baseweb="tab"]:before {
    content: "";
    position: absolute;
    bottom: 0;
    left: 50%;
    right: 50%;
    height: 3px;
    background: linear-gradient(to right, #FF9933, #7CB342);
    transition: all 0.4s ease-out;
    opacity: 0;
}

div.stTabs [data-baseweb="tab"]:hover:before {
    left: 20%;
    right: 20%;
    opacity: 1;
}

/* Improved hover effect with sliding animation */
div.stTabs [data-baseweb="tab"]:hover {
    background-color: #FFF6E9;
    color: #FF7700;
    cursor: pointer;
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 6px 12px rgba(255, 119, 0, 0.15);
    z-index: 5;
}

/* Active tab styling - premium orange theme with enhanced visibility */
div.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-image: linear-gradient(135deg, #FFA94D, #FF7700);
    color: white;
    border-color: #FF8000;
    box-shadow: 0 8px 20px rgba(255, 153, 51, 0.4);
    transform: translateY(-4px) scale(1.05);
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    z-index: 10;
}

/* Enhanced green accent for active tab */
div.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, #7CB342, #9CCC65);
    box-shadow: 0 1px 4px rgba(139, 195, 74, 0.5);
    border-radius: 0 0 4px 4px;
}

/* Tab content area - transparent background and wider */
div.stTabs [data-baseweb="tab-panel"] {
    background-color: transparent;
    border: none;
    box-shadow: none;
    padding: 32px 16px;
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
}

/* Enhanced tab indicator */
div.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(to right, #FF9933, #8BC34A);
    height: 0;  /* Hidden in favor of our custom indicators */
    opacity: 0;
}

/* Improve spacing and overall tab container width */
div.stTabs {
    margin-top: 20px;
    margin-bottom: 30px;
    width: 100%;
    max-width: 100%;
    padding: 0;
}

/* Force stTabs to take full width */
section[data-testid="stSidebar"] .block-container div[data-testid="stVerticalBlock"] > div:has(> div.stTabs),
.main .block-container div[data-testid="stVerticalBlock"] > div:has(> div.stTabs) {
    width: 100%;
    max-width: 100%;
    padding: 0;
}

/* Sliding animation for tab content */
div.stTabs [data-baseweb="tab-panel"] {
    animation: slideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(15px); }
    to { opacity: 1; transform: translateX(0); }
}
        </style>
    """, unsafe_allow_html=True)

# Call this function at the beginning of your app
apply_custom_css()

st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #E6F4EA; /* Light green */
        border-right: 2px solid #c1d8c4; /* Subtle border for depth */
        box-shadow: 4px 0px 8px rgba(0, 0, 0, 0.1); /* Light shadow effect */
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #006400 !important; /* Dark green */
        font-family: 'Arial', sans-serif; /* Clean and modern font */
        line-height: 1.6; /* Improved text readability */
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #004d00 !important; /* Slightly darker green */
        font-weight: 600; /* Make headers stand out more */
        text-transform: uppercase; /* Make headers look sharper */
    }

    /* Sidebar buttons and links */
    [data-testid="stSidebar"] a {
        color: #006400 !important; /* Dark green */
        font-weight: bold;
        text-decoration: none; /* Remove underline */
        transition: color 0.3s ease, background-color 0.3s ease; /* Smooth hover transition */
    }
    [data-testid="stSidebar"] a:hover {
        color: #004d00 !important; /* Slightly darker green */
        background-color: #b3e6b3 !important; /* Soft green on hover */
        padding: 2px 8px; /* Add a subtle padding effect */
        border-radius: 5px; /* Slightly rounded corners */
    }

    /* Sidebar selection highlight */
    [data-testid="stSidebarNav"] .st-emotion-cache-1v0mbdj {
        background-color: #b3e6b3 !important; /* Soft green */
        border-radius: 10px;
        transition: background-color 0.3s ease; /* Smooth transition for selection */
    }

    /* Sidebar items (buttons, links, etc.) */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #006400 !important; /* Dark green for buttons */
        color: white !important; /* White text on buttons */
        border: none;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle button shadow */
        transition: all 0.3s ease; /* Smooth button hover effect */
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #004d00 !important; /* Darker green on hover */
        box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Fixed background gradient
st.markdown("""
<div class="gradient-bg"></div>
""", unsafe_allow_html=True)


st.title("👁〰👁 Dialogue Visualiser Tool")

st.markdown("""
This tool helps you gather and process dialogue data from various sources:
- Upload your own data files
- Scrape website content
- Extract Reddit thread discussions
- Get YouTube video transcripts and comments
""")

# Sidebar for data input and configuration
with st.sidebar:
    st.header("Data Source")
    data_source = st.selectbox(
        "Select data source",
        ["Upload File", "Website Scraping", "Reddit Thread", "YouTube Video", "Sample Data"]
    )
    
    # File upload options
    if data_source == "Upload File":
        file_format = st.selectbox(
            "File format",
            ["JSON", "CSV", "ConvoKit", "Text"]
        )
        uploaded_file = st.file_uploader(f"Upload your {file_format} file", type=["json", "csv", "txt"])
    
    # Website scraping options
    elif data_source == "Website Scraping":
        website_url = st.text_input("Website URL (include https://)")
        extraction_mode = st.radio(
            "Extraction mode",
            ["Full page content", "Article text only", "Comments section"]
        )
        scrape_button = st.button("Scrape Website")
    
    # Reddit options
    elif data_source == "Reddit Thread":
        reddit_url = st.text_input("Reddit thread URL")
        reddit_depth = st.slider("Comment depth to scrape", 1, 10, 5)
        reddit_button = st.button("Fetch Reddit Thread")
    
    # YouTube options
    elif data_source == "YouTube Video":
        youtube_url = st.text_input("YouTube video URL or ID")
        data_to_scrape = st.multiselect(
            "Data to scrape",
            ["Video Transcript", "Video Comments"],
            default=["Video Transcript", "Video Comments"]
        )
        comment_limit = st.slider("Maximum comments to scrape", 10, 500, 100)
        youtube_button = st.button("Fetch YouTube Data")
    
    # NLP Processing options
    st.header("NLP Processing")
    enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
    enable_entity = st.checkbox("Entity Recognition", value=True)
    enable_topics = st.checkbox("Topic Extraction", value=True)
    
    # Download options
    st.header("Export")
    export_format = st.selectbox(
        "Export format",
        ["CSV", "JSON", "Excel"]
    )

# Helper functions for data collection and processing

def extract_video_id(youtube_url):
    """Extract YouTube video ID from URL."""
    # Handle different URL formats
    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/watch" in youtube_url:
        return re.findall(r"v=([^&]+)", youtube_url)[0]
    elif "youtube.com/embed/" in youtube_url:
        return youtube_url.split("youtube.com/embed/")[1].split("?")[0]
    elif len(youtube_url) == 11:  # Direct video ID
        return youtube_url
    return None

def get_youtube_transcript(video_id):
    """Get transcript for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Get transcript using youtube_transcript_api
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Create DataFrame from transcript
        df = pd.DataFrame(transcript_list)
        
        # Add message ID and user columns
        df['id'] = range(1, len(df) + 1)
        df['user'] = 'YouTubeTranscript'
        df['reply_to'] = None
        
        # Convert timestamp_seconds to datetime format
        start_time = datetime.now()
        df['timestamp'] = df['start'].apply(
            lambda x: (start_time.replace(hour=0, minute=0, second=0) + pd.Timedelta(seconds=x)).isoformat()
        )
        
        # Rename columns to match your schema
        df = df.rename(columns={'start': 'timestamp_seconds', 'text': 'text'})
        
        return df
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return pd.DataFrame()

def get_youtube_comments(video_id, max_comments=100):
    """Get comments for a YouTube video using youtube-comment-downloader."""
    try:
        downloader = YoutubeCommentDownloader()
        comments_list = downloader.get_comments(f"https://www.youtube.com/watch?v={video_id}", sort_by=0)
        
        comments = []
        for i, comment in enumerate(comments_list):
            if i >= max_comments:
                break
            comments.append({
                "id": comment["cid"],
                "user": comment["author"],
                "text": comment["text"],
                "timestamp": comment["time_parsed"].isoformat(),
                "reply_to": comment["parent"] if "parent" in comment else None
            })
        
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"Error fetching YouTube comments: {str(e)}")
        return pd.DataFrame()

def scrape_website(url, mode="Article text only"):
    """Scrape content from a website."""
    try:
        # Use trafilatura for high-quality web content extraction
        downloaded = trafilatura.fetch_url(url)
        
        if mode == "Full page content":
            # Get full HTML and parse with BeautifulSoup
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Break into paragraphs
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
        elif mode == "Article text only":
            # Extract main text content using trafilatura
            text = trafilatura.extract(downloaded, include_comments=False, 
                                     include_tables=False, favor_precision=True)
            paragraphs = text.split('\n\n') if text else []
            
        elif mode == "Comments section":
            # Try to extract comments
            comments = trafilatura.extract_comments(downloaded)
            if comments:
                paragraphs = [c['text'] for c in comments]
            else:
                # Fallback to heuristic comment detection
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for common comment container classes/IDs
                comment_containers = soup.select('.comments, #comments, .comment-section, .comment')
                if comment_containers:
                    paragraphs = [c.get_text(strip=True) for c in comment_containers]
                else:
                    paragraphs = ["No comments found on this page"]
        
        # Convert to our standard message format
        messages = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():  # Skip empty paragraphs
                messages.append({
                    "id": i + 1,
                    "user": "WebsiteContent",
                    "text": paragraph,
                    "timestamp": datetime.now().isoformat(),
                    "reply_to": None
                })
        
        return pd.DataFrame(messages)
    
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return pd.DataFrame()

def scrape_reddit_thread(url, max_depth=5):
    """Scrape a Reddit thread."""
    try:
        # Embed Reddit API credentials directly
        client_id = "aJHo-iHWDvrJezMFeKEOIQ"
        client_secret = "_gdn0epI7jf94kHjjTDU1kREJ9sl3w"
        user_agent = "Dialogue_Ingestion_Tool/1.0"

        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        # Extract submission ID from URL
        if 'comments' in url:
            match = re.search(r'comments/([a-zA-Z0-9]+)/', url)
            submission_id = match.group(1) if match else None
        else:
            submission_id = url.split('/')[-1]

        if not submission_id:
            st.error("Invalid Reddit URL format.")
            return pd.DataFrame()

        submission = reddit.submission(id=submission_id)

        # Get submission data
        messages = [{
            "id": submission.id,
            "user": submission.author.name if submission.author else "[deleted]",
            "text": submission.selftext if submission.selftext else "[No text content]",
            "title": submission.title,
            "timestamp": datetime.fromtimestamp(submission.created_utc).isoformat(),
            "reply_to": None
        }]

        # Get comments
        submission.comments.replace_more(limit=None)

        def extract_comments(comments, parent_id=None, level=0):
            """Recursively extract comments up to the max depth."""
            if level > max_depth:
                return []

            result = []
            for comment in comments:
                if hasattr(comment, 'body'):  # Ensure it's a Comment object
                    result.append({
                        "id": comment.id,
                        "user": comment.author.name if comment.author else "[deleted]",
                        "text": comment.body,
                        "timestamp": datetime.fromtimestamp(comment.created_utc).isoformat(),
                        "reply_to": parent_id
                    })

                    # Recursively get replies
                    result.extend(extract_comments(comment.replies, comment.id, level + 1))

            return result

        messages.extend(extract_comments(submission.comments))

        return pd.DataFrame(messages)

    except Exception as e:
        st.error(f"Error scraping Reddit thread: {str(e)}")
        return pd.DataFrame()
    

def load_sample_data():
    """Load sample discussion data."""
    sample_data = {
        "messages": [
            {"id": 1, "user": "user1", "text": "I think we should focus on climate change solutions.", "timestamp": "2023-05-01T10:00:00", "reply_to": None},
            {"id": 2, "user": "user2", "text": "I disagree. Economic growth should be our priority right now.", "timestamp": "2023-05-01T10:05:00", "reply_to": 1},
            {"id": 3, "user": "user3", "text": "We can have both economic growth and climate solutions if we invest in green technology.", "timestamp": "2023-05-01T10:10:00", "reply_to": 2},
            {"id": 4, "user": "user1", "text": "Exactly! Green jobs could boost the economy while helping the environment.", "timestamp": "2023-05-01T10:15:00", "reply_to": 3},
            {"id": 5, "user": "user4", "text": "But the costs of transitioning are too high for many businesses.", "timestamp": "2023-05-01T10:20:00", "reply_to": 3},
            {"id": 6, "user": "user2", "text": "I agree with user4. Small businesses would struggle with new regulations.", "timestamp": "2023-05-01T10:25:00", "reply_to": 5},
            {"id": 7, "user": "user3", "text": "What if we had government subsidies to help small businesses transition?", "timestamp": "2023-05-01T10:30:00", "reply_to": 6},
            {"id": 8, "user": "user1", "text": "That's a great idea! It could make the transition more manageable.", "timestamp": "2023-05-01T10:35:00", "reply_to": 7},
            {"id": 9, "user": "user4", "text": "But then we're increasing government spending, which has its own problems.", "timestamp": "2023-05-01T10:40:00", "reply_to": 7},
            {"id": 10, "user": "user5", "text": "I'm more concerned about immediate climate impacts. We're seeing worsening natural disasters already.", "timestamp": "2023-05-01T10:45:00", "reply_to": 1},
        ]
    }
    return pd.DataFrame(sample_data["messages"])

def process_uploaded_file(uploaded_file, file_format):
    """Process an uploaded file."""
    try:
        if file_format == "JSON":
            data = json.load(uploaded_file)
            # Handle various JSON formats
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif "messages" in data:
                return pd.DataFrame(data["messages"])
            else:
                # Try to normalize nested JSON
                return pd.json_normalize(data)
                
        elif file_format == "CSV":
            return pd.read_csv(uploaded_file)
            
        elif file_format == "Text":
            # Process plain text by splitting into paragraphs
            text = uploaded_file.getvalue().decode("utf-8")
            paragraphs = text.split('\n\n')
            
            messages = []
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():  # Skip empty paragraphs
                    messages.append({
                        "id": i + 1,
                        "user": "TextContent",
                        "text": paragraph.strip(),
                        "timestamp": datetime.now().isoformat(),
                        "reply_to": None
                    })
            
            return pd.DataFrame(messages)
        
        else:
            st.error(f"Unsupported file format: {file_format}")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment(text):
    """Analyze sentiment of text."""
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return sentiment
    except:
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}

def extract_entities(text):
    """Simple entity extraction with NLTK data handling."""
    import nltk
    import os
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    
    # Set up NLTK data path and download required resources if needed
    try:
        # First try to use the resources
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # If resources are not found, download them
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Add this directory to nltk's search path
        nltk.data.path.insert(0, nltk_data_dir)
        
        # Download required resources
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        
        # Now try again
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
    
    # Simple heuristic: capitalized words not at the beginning of sentences
    entities = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        for i, word in enumerate(words):
            if (i > 0 and word[0].isupper() and 
                word.lower() not in stop_words and 
                len(word) > 1):
                entities.append(word)
    
    return list(set(entities))
    
def extract_keywords(text, top_n=5):
    """Extract keywords from text."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and punctuation
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Get top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]

def apply_nlp_processing(df):
    """Apply NLP processing to the DataFrame."""
    # Skip if no data
    if df.empty or 'text' not in df.columns:
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Apply sentiment analysis if enabled
    if enable_sentiment:
        sentiment_results = result_df['text'].apply(analyze_sentiment)
        result_df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
        result_df['sentiment_positive'] = sentiment_results.apply(lambda x: x['pos'])
        result_df['sentiment_negative'] = sentiment_results.apply(lambda x: x['neg'])
        result_df['sentiment_neutral'] = sentiment_results.apply(lambda x: x['neu'])
    
    # Apply entity recognition if enabled
    if enable_entity:
        result_df['entities'] = result_df['text'].apply(extract_entities)
    
    # Apply keyword extraction if enabled
    if enable_topics:
        result_df['keywords'] = result_df['text'].apply(extract_keywords)
    
    return result_df

def export_data(df, format="CSV"):
    """Export processed data."""
    if df.empty:
        st.error("No data to export")
        return None
    
    if format == "CSV":
        return df.to_csv(index=False).encode('utf-8')
    elif format == "JSON":
        return df.to_json(orient="records", indent=2).encode('utf-8')
    elif format == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='DialogueData')
        return output.getvalue()
    else:
        st.error(f"Unsupported export format: {format}")
        return None

# Main app logic
data_loaded = False
messages_df = None

# Process data based on source
if data_source == "Upload File" and uploaded_file is not None:
    messages_df = process_uploaded_file(uploaded_file, file_format)
    data_loaded = not messages_df.empty

elif data_source == "Website Scraping" and scrape_button:
    if website_url:
        with st.spinner('Scraping website content...'):
            messages_df = scrape_website(website_url, extraction_mode)
            data_loaded = not messages_df.empty
    else:
        st.warning("Please enter a website URL")

elif data_source == "Reddit Thread" and reddit_button:
    if reddit_url:
        with st.spinner('Fetching Reddit thread...'):
            messages_df = scrape_reddit_thread(reddit_url, reddit_depth)
            data_loaded = not messages_df.empty
    else:
        st.warning("Please enter a Reddit thread URL")

elif data_source == "YouTube Video" and youtube_button:
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            with st.spinner('Fetching YouTube data...'):
                dfs = []
                
                if "Video Transcript" in data_to_scrape:
                    transcript_df = get_youtube_transcript(video_id)
                    if not transcript_df.empty:
                        dfs.append(transcript_df)
                
                if "Video Comments" in data_to_scrape:
                    comments_df = get_youtube_comments(video_id, comment_limit)
                    if not comments_df.empty:
                        dfs.append(comments_df)
                
                if dfs:
                    messages_df = pd.concat(dfs, ignore_index=True)
                    data_loaded = True
                    
                    # Display the YouTube video
                    st.subheader("YouTube Video")
                    st.video(f"https://youtu.be/{video_id}")
        else:
            st.error("Invalid YouTube URL or ID")
    else:
        st.warning("Please enter a YouTube URL or ID")

elif data_source == "Sample Data":
    messages_df = load_sample_data()
    data_loaded = True

# Initialize processed_df at the beginning
processed_df = None

# Apply NLP processing if data is loaded
if data_loaded and messages_df is not None and not messages_df.empty:
    # Process data with NLP techniques
    with st.spinner('Applying NLP processing...'):
        processed_df = apply_nlp_processing(messages_df)
    
    # Display results
    st.subheader("Processed Data")
    
    # Basic statistics
    st.write(f"Loaded {len(processed_df)} messages from {processed_df['user'].nunique()} unique users/sources")
    

# Show data tabs
    tabs = st.tabs(["Raw Data & NLP Results", "Network Visualisation", "Timeline Visualisation", "Topic Visualisation & Sentiment Heatmap", "Argumentation Graph"])
    
    with tabs[0]:
        st.dataframe(processed_df)

        # Display NLP results
        if enable_sentiment:
            st.subheader("Sentiment Analysis")
            
            # Calculate average sentiment
            avg_sentiment = processed_df['sentiment_compound'].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.05 else ("Negative" if avg_sentiment < -0.05 else "Neutral")
            
            st.write(f"Average sentiment: {avg_sentiment:.2f} ({sentiment_label})")
            
            # Count sentiment categories
            sentiment_categories = pd.cut(
                processed_df['sentiment_compound'], 
                bins=[-1, -0.5, -0.05, 0.05, 0.5, 1],
                labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            ).value_counts().sort_index()
            
            st.write("Sentiment distribution:")
            st.write(sentiment_categories)
        
        if enable_entity:
            st.subheader("Entity Recognition")
            
            # Get all entities and count frequencies
            all_entities = []
            for entity_list in processed_df['entities']:
                all_entities.extend(entity_list)
            
            entity_counts = pd.Series(all_entities).value_counts()
            
            st.write("Top entities found:")
            st.write(entity_counts.head(10))
        
        if enable_topics:
            st.subheader("Keywords/Topics")
            
            # Get all keywords and count frequencies
            all_keywords = []
            for keyword_list in processed_df['keywords']:
                all_keywords.extend(keyword_list)
            
            keyword_counts = pd.Series(all_keywords).value_counts()
            
            st.write("Top keywords found:")
            st.write(keyword_counts.head(10))

        st.subheader("Export Data")
        
        # Allow user to download data
        export_data_bytes = export_data(processed_df, export_format)
        
        if export_data_bytes:
            file_extension = export_format.lower()
            if file_extension == "excel":
                file_extension = "xlsx"
                
            st.download_button(
                label=f"Download data as {export_format}",
                data=export_data_bytes,
                file_name=f"dialogue_data.{file_extension}",
                mime=f"application/{file_extension}"
            )
            
            # Also offer a zip bundle with all formats
            with st.expander("Download all formats (ZIP)"):
                # Create a ZIP file containing all formats
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add each format to the zip
                    for fmt in ["CSV", "JSON", "Excel"]:
                        data_bytes = export_data(processed_df, fmt)
                        ext = fmt.lower()
                        if ext == "excel":
                            ext = "xlsx"
                        zip_file.writestr(f"dialogue_data.{ext}", data_bytes)
                
                st.download_button(
                    label="Download ZIP with all formats",
                    data=zip_buffer.getvalue(),
                    file_name="dialogue_data_all_formats.zip",
                    mime="application/zip"
                )
    # In the Network Visualization tab section
    with tabs[1]:
        st.subheader("User Interaction")
        
        # Check if we have reply information for creating a network
        if not processed_df.empty and 'reply_to' in processed_df.columns:
            with st.spinner('Generating network visualization...'):
                # Around line 1161 in Demo_Application.py
                try:
                    # Make sure we're properly unpacking the return value
                    network_result = create_network_visualization(processed_df)
                    
                    # Check if it's a tuple with 2 elements as expected
                    if isinstance(network_result, tuple) and len(network_result) == 2:
                        network_fig, insights = network_result
                        st.plotly_chart(network_fig, use_container_width=True)
                        
                        # Optionally display insights
                        st.write("### Network Insights")
                        for insight in insights:
                            st.write(f"- {insight}")
                    else:
                        st.error("Network visualization function didn't return expected values")
                except Exception as e:
                    st.error(f"Error creating network visualization: {str(e)}")
                                
                # Add legend and explanation
                st.markdown("""
                ### Understanding the Network Graph
                
                **Nodes (Circles):**
                - **Size**: Represents the number of messages from that user
                - **Color**: Indicates influence (how often others reply to this user)
                - **Blue**: Less influential users
                - **Red**: More influential/central users
                
                **Edges (Lines):**
                - **Thickness**: Shows frequency of interactions between users
                - **Color**: Represents sentiment (red = negative, blue = positive)
                - **Direction**: Shows who is responding to whom
                
                **Interact with the visualization:**
                - Hover over nodes to see detailed user statistics
                - Hover over edges to see interaction details
                - Toggle visualization modes from the dropdown menu
                """)
        else:
            st.info("Network visualization requires reply relationship data. Make sure your data has a 'reply_to' column.")

    # In the Timeline Visualization tab section
    with tabs[2]:  # Conversation Timeline tab
        st.subheader("Conversation Timeline")
        
        if not processed_df.empty and 'timestamp' in processed_df.columns:
            with st.spinner('Generating timeline visualization...'):
                try:
                    # Call the visualization function - now correctly unpacking three return values
                    timeline_fig, metrics, summary_text = create_timeline_visualization(processed_df)
                    
                    # Display the plotly figure
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Create expandable sections for the detailed metrics and summary
                    with st.expander("📊 Conversation Metrics"):
                        # Display general metrics
                        st.subheader("General Information")
                        general = metrics['general']
                        cols = st.columns(3)
                        cols[0].metric("Total Messages", general['total_messages'])
                        cols[1].metric("Duration", str(general['conversation_duration']).split('.')[0])
                        cols[2].metric("Participants", len(metrics['users']))
                        
                        # User metrics table
                        st.subheader("User Activity")
                        user_data = []
                        for user in metrics['users']:
                            user_data.append({
                                "User": user['User'],
                                "Messages": user['Message_Count'],
                                "% of Total": f"{user['Percentage']:.1f}%",
                                "Avg Length": f"{user['Avg_Message_Length']:.1f}",
                                "Questions": user['Questions_Asked'],
                                "Agreements": user['Agreements'],
                                "Disagreements": user['Disagreements']
                            })
                        
                        st.dataframe(user_data, use_container_width=True)
                        
                        # Display interaction patterns
                        if 'interactions' in metrics:
                            st.subheader("Interaction Patterns")
                            int_data = metrics['interactions']
                            cols = st.columns(4)
                            cols[0].metric("Questions", int_data['questions'])
                            cols[1].metric("Agreements", int_data['agreements'])
                            cols[2].metric("Disagreements", int_data['disagreements'])
                            cols[3].metric("Emotional Peaks", int_data['emotional_peaks'])
                    
                    # Display the markdown summary
                    with st.expander("📝 Conversation Analysis Summary"):
                        st.markdown(summary_text)
                    
                    # Add legend and explanation
                    st.markdown("""
                    ### Understanding the Timeline Visualization
                    
                    **Message Markers:**
                    - **Size**: Larger markers indicate longer messages or higher emotional intensity
                    - **Color**: Different colors represent different users
                    - **Symbol**: Shows message characteristics
                    - Circle: Regular message
                    - Diamond: Question
                    - Square: Agreement
                    - X: Disagreement
                    - Star: Emotional peak/significant message
                    
                    **Timeline Elements:**
                    - **Top Panel**: Shows messages by user over time
                    - **Middle Panel**: Displays sentiment flow (blue line) throughout the conversation
                    - **Bottom Panel**: Shows message frequency over time
                    - **Vertical Dotted Lines**: Indicate topic shifts
                    - **Background Shading**: Conversation segments separated by time gaps
                    
                    **Interact with the visualization:**
                    - Hover over any element to see detailed information
                    - Click on legend items to show/hide specific components
                    - Zoom and pan to explore specific time periods
                    - Use the toolbar in the top-right to download or explore the data
                    """)
                except Exception as e:
                    st.error(f"Error creating timeline visualization: {str(e)}")
                    st.code(str(e), language="text")
        else:
            st.info("Timeline visualization requires timestamp data. Make sure your data has a 'timestamp' column.")
    

    # In the Topic Trees Visualization tab section
    with tabs[3]:  # Topic Trees tab
        st.subheader("Topic Trees & Conversation Flow Analysis")
        
        if not processed_df.empty and 'text' in processed_df.columns:
            with st.spinner('Analyzing conversation topics and generating visualizations...'):
                try:
                    # Call the topic trees visualization function
                    topic_results = create_topic_trees(processed_df)
                    
                    # Display the combined dashboard (previously in topic_tabs[0])
                    st.plotly_chart(topic_results['visualization'], use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    ### Combined Topic Analysis Dashboard
                    
                    This dashboard provides a comprehensive view of how topics evolve and relate throughout the conversation:
                    - **Topic Flow Diagram**: Shows how topics connect and transition
                    - **Topic Relationship Map**: Visualizes relationships between topics
                    - **Topic Evolution**: Shows how topics change in prominence over time
                    """)
                    
                    # Display topic descriptions (previously in topic_tabs[5])
                    st.subheader("Topic Descriptions")
                    topic_data = []
                    for i, name in enumerate(topic_results['topic_names']):
                        terms = ", ".join(topic_results['topic_terms'][i][:10])
                        count = (topic_results['topic_data']['topic_id'] == i).sum()
                        percentage = count / len(topic_results['topic_data']) * 100
                        
                        topic_data.append({
                            "Topic": name,
                            "Top Terms": terms,
                            "Message Count": count,
                            "Percentage": f"{percentage:.1f}%"
                        })
                    
                    st.dataframe(topic_data, use_container_width=True)
                    
                    # Display insights (previously in topic_tabs[5])
                    st.subheader("Key Insights")
                    for insight in topic_results['insights']:
                        st.markdown(f"- {insight}")
                    
                    # Add interactive topic explorer (previously in topic_tabs[5])
                    st.subheader("Explore Messages by Topic")
                    selected_topic = st.selectbox(
                        "Select a topic to explore:",
                        options=range(len(topic_results['topic_names'])),
                        format_func=lambda x: topic_results['topic_names'][x]
                    )
                    
                    if selected_topic is not None:
                        topic_messages = topic_results['topic_data'][
                            topic_results['topic_data']['topic_id'] == selected_topic
                        ]
                        
                        if not topic_messages.empty:
                            st.write(f"{len(topic_messages)} messages in this topic")
                            
                            # Sort by topic weight to show most relevant messages first
                            weight_col = f'topic_weight_{selected_topic}'
                            if weight_col in topic_messages.columns:
                                topic_messages = topic_messages.sort_values(by=weight_col, ascending=False)
                            
                            # Display sample messages
                            for i, row in topic_messages.head(10).iterrows():
                                user = row.get('user', 'Unknown')
                                text = row.get('text', '')
                                weight = row.get(weight_col, 0) if weight_col in row else 0
                                
                                with st.container():
                                    st.markdown(f"**{user}** (relevance: {weight:.2f})")
                                    st.markdown(f"> {text}")
                                    st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error creating topic trees visualization: {str(e)}")
                    st.code(traceback.format_exc(), language="text")
                    st.info("Tip: Topic analysis requires sufficient text data with variety. Try with a larger conversation dataset.")
        else:
            st.info("Topic analysis requires text data. Make sure your data has a 'text' column with conversation messages.")
        
        # Add explanation about topic analysis
        with st.expander("ℹ️ About Topic Analysis"):
            st.markdown("""
            ### Understanding Topic Analysis
            
            This analysis uses Natural Language Processing (NLP) to identify patterns and topics in conversation data:
            
            **What It Shows:**
            - **Topics**: Key themes and subjects discussed (extracted using NMF - Non-negative Matrix Factorization)
            - **Transitions**: How the conversation flows between different topics
            - **Evolution**: How topics emerge, develop, and fade over time
            - **Relationships**: How topics connect semantically to each other
            
            **How to Use It:**
            - Identify main discussion themes and how they connect
            - Understand how conversations evolve and branch
            - Discover subtopics and related concepts
            - Analyze topic patterns and participation
            
            **Limitations:**
            - Requires sufficient text data (at least 20+ messages)
            - Works best with substantive conversation content
            - Topic labels are generated automatically based on keywords
            
            The visualizations are interactive - hover, zoom, and click to explore different aspects of the topic analysis.
            """)

    with tabs[4]:  # Argumentation Graph tab
        st.subheader("Argumentation Structure Analysis")
        
        if not processed_df.empty and 'text' in processed_df.columns:
            with st.spinner('Analyzing argumentation structure...'):
                try:
                    # Generate the argumentation graph
                    arg_fig, arg_analysis, arg_summary = create_argumentation_graph(processed_df)
                    
                    # Display the visualization
                    st.plotly_chart(arg_fig, use_container_width=True)
                    
                    # Display the analysis summary
                    st.markdown(arg_summary)
                    
                    # Show detailed statistics in expandable section
                    with st.expander("Detailed Argumentation Statistics"):
                        if 'arg_counts' in arg_analysis:
                            st.subheader("Argument Types")
                            arg_counts = arg_analysis['arg_counts']
                            arg_df = pd.DataFrame({
                                'Type': list(arg_counts.keys()),
                                'Count': list(arg_counts.values())
                            })
                            st.bar_chart(arg_df.set_index('Type'))
                        
                        if 'controversial_claims' in arg_analysis and arg_analysis['controversial_claims']:
                            st.subheader("Controversial Claims")
                            for i, claim in enumerate(arg_analysis['controversial_claims']):
                                st.markdown(f"{i+1}. {claim}")
                    
                except Exception as e:
                    st.error(f"Error creating argumentation graph: {str(e)}")
                    st.code(traceback.format_exc(), language="text")
        else:
            st.info("Argumentation analysis requires text data. Make sure your data has a 'text' column with conversation messages.")







    # Only process NLP-related features when we have data AND the corresponding features are enabled
    if enable_sentiment and 'sentiment_compound' in processed_df.columns:
        # Calculate average sentiment
        avg_sentiment = processed_df['sentiment_compound'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.05 else ("Negative" if avg_sentiment < -0.05 else "Neutral")
        
        # Count sentiment categories
        sentiment_categories = pd.cut(
            processed_df['sentiment_compound'], 
            bins=[-1, -0.5, -0.05, 0.05, 0.5, 1],
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        ).value_counts().sort_index()

    if enable_entity and 'entities' in processed_df.columns:
        # Get all entities and count frequencies
        all_entities = []
        for entity_list in processed_df['entities']:
            all_entities.extend(entity_list)
        
        entity_counts = pd.Series(all_entities).value_counts()

    if enable_topics and 'keywords' in processed_df.columns:
        # Get all keywords and count frequencies
        all_keywords = []
        for keyword_list in processed_df['keywords']:
            all_keywords.extend(keyword_list)
        
        keyword_counts = pd.Series(all_keywords).value_counts()
    else:
        # No data loaded yet, show instructions
        for tab in tabs:
            with tab:
                if data_source != "Sample Data":  # Only show if not on sample data option
                    st.info("Please provide the necessary information and click the appropriate button to fetch data")
                    
                    # Show help text for each data source
                    if data_source == "Website Scraping":
                        st.markdown("""
                        ## Website Scraping Tips
                        - Enter a complete URL (including https://)
                        - The "Article text only" mode works best for news sites and blogs
                        - "Comments section" attempts to extract user comments if available
                        - Be aware that some websites block scraping attempts
                        """)
                    
                    elif data_source == "Reddit Thread":
                        st.markdown("""
                        ## Reddit Thread Tips
                        - Use the full URL of a Reddit post
                        - Adjust the comment depth to control how many nested comments to retrieve
                        - Thread data includes the original post and all comments
                        """)
                    
                    elif data_source == "YouTube Video":
                        st.markdown("""
                        ## YouTube Tips
                        - Enter a YouTube video URL or just the video ID
                        - Transcripts are available for most YouTube videos
                        - The comment scraping uses the YouTube API (limited without API key)
                        - For best results, choose videos with active discussions
                        """)
