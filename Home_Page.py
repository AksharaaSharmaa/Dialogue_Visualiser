import nltk
import os

# Create data directory if it doesn't exist
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)

import streamlit as st
import pandas as pd
from PIL import Image
import base64
import base64
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Discourse Lens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Create a pulsing logo element that's more visually appealing
def create_modern_logo():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;" class="floating">
        <svg width="140" height="140" viewBox="0 0 140 140">
            <defs>
                <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#FF7A00;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#FF9E44;stop-opacity:1" />
                </linearGradient>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <circle cx="70" cy="70" r="60" fill="url(#logoGradient)" opacity="0.1"></circle>
            <circle cx="70" cy="70" r="48" fill="url(#logoGradient)" opacity="0.2"></circle>
            <circle cx="70" cy="70" r="36" fill="url(#logoGradient)" opacity="0.4"></circle>
            <circle cx="70" cy="70" r="24" fill="url(#logoGradient)" opacity="0.8" filter="url(#glow)"></circle>
            <text x="70" y="77" text-anchor="middle" fill="#FFFFFF" font-size="22" font-weight="bold" letter-spacing="-1">DL</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)

# Enhanced header with a cleaner, more modern look
def render_enhanced_header():
    st.markdown("""
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <h1 style="font-size: 4rem; margin-bottom: 0.2rem;">Discourse Lens</h1>
        <p style="font-size: 1.4rem; opacity: 0.8; margin-top: 0.5rem; color: #666; font-weight: 400; max-width: 700px; margin-left: auto; margin-right: auto;">Visualizing Online Discussions with Clarity and Insight</p>
        <div class="divider divider-center"></div>
    </div>
    """, unsafe_allow_html=True)

render_enhanced_header()# Full-width hero section

import streamlit as st

# Center the button using custom HTML and CSS
import streamlit as st

st.markdown(
    """
    <style>
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .center-button {
        background-color: #ff7f0e;
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        border-radius: 30px;
        cursor: pointer;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }
    .center-button:hover {
        background-color: #e67300;
    }
    </style>
    <div class="center-container">
        <a href="Demo_Application" target="_self">
            <button class="center-button">TRY THIS DEMO APPLICATION</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Main features section with a more beautiful design
st.markdown("""
<h2 style='text-align: center; margin-top: 60px; font-size: 2.5rem;'>Key Features</h2>
<div style='text-align: center; max-width: 700px; margin: 0 auto 40px auto;'>
    <p style='color: #666; font-size: 1.2rem; line-height: 1.6;'>Explore our powerful tools designed to bring clarity to complex online conversations</p>
    <div class="divider divider-center"></div>
</div>
""", unsafe_allow_html=True)

# Three-column feature layout with enhanced cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">🌐</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Network Visualization</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">See how users interact, who influences conversations, and how ideas spread through interactive network graphs.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Identify key influencers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">📈</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Trend Analysis</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">Track how topics emerge, evolve, and fade over time with powerful trend visualization tools.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Predict emerging topics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">😊</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Emotion Tracking</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">Monitor emotional patterns and sentiment shifts throughout conversations to understand user engagement.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Improve user experience</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Second row of features
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">🔍</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Argument Analysis</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">Trace the structure of arguments and counter-arguments in complex discussions with advanced visualization.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Understand discussion flow</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">📊</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Content Analysis</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">Analyze message content to identify key themes, topics, and linguistic patterns driving engagement.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Extract meaningful insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center; margin-bottom: 25px;" class="floating">
            <div style="background: linear-gradient(135deg, #FF7A00 0%, #FF9E44 100%); width: 80px; height: 80px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 10px 20px rgba(255, 122, 0, 0.2);">
                <span style="color: white; font-size: 34px;">📱</span>
            </div>
        </div>
        <h3 style="text-align: center; font-size: 1.6rem; margin-bottom: 20px;">Cross-Platform</h3>
        <p style="text-align: center; line-height: 1.7; color: #555;">Monitor discussions across multiple platforms and channels with unified analytics and reporting.</p>
        <div style="background-color: #FFF5EB; border-radius: 10px; padding: 12px; margin-top: 25px; text-align: center;">
            <span style="color: #FF7A00; font-weight: 600;">Centralize your insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st

# Add the CSS styles with an orange background and white text
st.markdown("""
<style>
.footer {
    background: linear-gradient(135deg, #FF7A00 0%, #FF9D4D 100%);
    padding: 24px 32px;
    margin-top: 40px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.footer-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
}
.footer-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: white;
}
.footer-subtitle {
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.9);
    margin-bottom: 8px;
}
.footer-message {
    font-size: 0.9rem;
    line-height: 1.5;
    max-width: 600px;
    margin: 0 auto;
    color: rgba(255, 255, 255, 0.95);
}
.footer-attribution {
    margin-top: 16px;
    font-size: 0.85rem;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.9);
}
.heart {
    color: #fff;
    font-size: 1rem;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Now add the improved footer HTML
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <p class="footer-attribution">
            Made with <span class="heart">❤</span> by Akshara Sharma for GFOSS as part of proposal for Google Summer of Code 2025
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
