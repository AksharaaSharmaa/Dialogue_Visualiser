import streamlit as st
import json
import os
import datetime
import uuid

st.markdown("""

<style>
    /* Import Modern Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Enhanced color palette - refined colors */
    :root {
        --primary-color: #FF7D35;       /* Vibrant orange - slightly deeper */
        --primary-light: #FFA76B;       /* Lighter orange for gradients */
        --primary-dark: #E06218;        /* Darker orange for hover states */
        --secondary-color: #4CAF50;     /* Rich green */
        --secondary-light: #8ED081;     /* Light green */
        --secondary-dark: #3D8B40;      /* Darker green for hover states */
        --accent-color: #FFB673;        /* Light orange */
        --background-color: #FFFCF7;    /* Warmer off-white background */
        --text-color: #2A2A2A;          /* Deeper text color for better contrast */
        --text-light: #6B6B6B;          /* Light text color for subtitles */
        --light-accent: #F0F9EE;        /* Very light green with less saturation */
        --card-bg: #FFFFFF;             /* Card background */
        --border-radius-sm: 10px;
        --border-radius-md: 14px;
        --border-radius-lg: 20px;
        --shadow-sm: 0 4px 12px rgba(0,0,0,0.05);
        --shadow-md: 0 8px 20px rgba(0,0,0,0.08);
        --shadow-lg: 0 12px 28px rgba(0,0,0,0.12);
        --transition-standard: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Base styling */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-color);
    }
    
    /* Modern title styling */
    .main-title {
        color: var(--primary-color);
        text-align: center;
        font-size: 3.4rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        position: relative;
        padding-bottom: 10px;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.5);
    }
    
    .main-title::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-light), var(--secondary-light));
        border-radius: 4px;
        box-shadow: 0 2px 6px rgba(255, 125, 53, 0.2);
    }
    
    .subtitle {
        color: var(--text-light);
        text-align: center;
        font-size: 1.5rem !important;
        margin-bottom: 3rem !important;
        font-weight: 500 !important;
        line-height: 1.5;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Refined header styles */
    h1, h2 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: var(--secondary-color);
        font-weight: 600;
        font-size: 1.7rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        padding-bottom: 16px;
    }
    
    h3::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, var(--secondary-color), var(--secondary-light));
        border-radius: 3px;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
    }

    /* ================= ENHANCED TAB STYLING ================= */
    /* Enhanced tab navigation - wider, more beautiful with animations */
    .stTabs {
        margin-top: 1.5rem;
        margin-bottom: 2.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--light-accent);
        border-radius: var(--border-radius-lg);
        padding: 10px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.03), 0 4px 15px rgba(0,0,0,0.02);
        display: flex;
        width: 100%;
        justify-content: space-between;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        border-radius: var(--border-radius-md);
        background-color: rgba(255, 255, 255, 0.75);
        color: var(--text-light);
        font-weight: 600;
        transition: var(--transition-standard);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 14px;
        border: none !important;
        flex: 1;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 120px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-light), var(--primary-dark));
        transform: scaleX(0);
        transition: transform 0.3s ease-out;
        border-radius: 1.5px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
        color: var(--primary-dark);
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        transform: scaleX(0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, var(--primary-color), var(--primary-dark)) !important;
        color: white !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-3px);
        font-weight: 700;
        position: relative;
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        width: 80%;
        height: 100%;
        top: 0;
        left: 10%;
        background: linear-gradient(rgba(255,255,255,0.2), rgba(255,255,255,0));
        border-radius: var(--border-radius-md);
    }
    
    .stTabs [aria-selected="true"]::before {
        transform: scaleX(1);
        height: 4px;
        background: linear-gradient(90deg, #FFFFFF, rgba(255,255,255,0.5));
    }
    
    /* Fade tab panel animations */
    .stTabs [role="tabpanel"] {
        animation: fadeIn 0.4s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ================= ENHANCED FEEDBACK CONTAINER ================= */
    /* Elegant feedback container with improved styling */
    .feedback-container {
        background-color: var(--card-bg);
        border-radius: var(--border-radius-lg);
        padding: 32px;
        margin: 28px 0;
        box-shadow: var(--shadow-md);
        transition: var(--transition-standard);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .feedback-container::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 8px;
        background: linear-gradient(to bottom, var(--primary-color), var(--primary-light));
        border-radius: 4px 0 0 4px;
    }
    
    .feedback-container::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        height: 60px;
        width: 60px;
        background: radial-gradient(circle at top right, rgba(255, 167, 107, 0.15), rgba(255, 167, 107, 0) 70%);
        border-radius: 0 var(--border-radius-lg) 0 60px;
        z-index: 0;
    }
    
    .feedback-container:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg), 0 10px 40px rgba(255, 125, 53, 0.15);
    }
    
    .feedback-container h4 {
        color: var(--primary-color);
        margin-top: 0;
        font-size: 1.4rem;
        padding-bottom: 14px;
        margin-bottom: 20px;
        position: relative;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    .feedback-container h4::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, var(--light-accent), rgba(240, 249, 238, 0.1));
    }
    
    .feedback-container p {
        position: relative;
        z-index: 1;
        line-height: 1.6;
        margin-bottom: 1rem;
        text-shadow: 0 1px 0 rgba(255,255,255,0.5);
    }
    
    .feedback-container ul {
        position: relative;
        z-index: 1;
        padding-left: 20px;
    }
    
    .feedback-container ul li {
        margin-bottom: 10px;
        position: relative;
        padding-left: 5px;
    }
    
    .feedback-container ul li::marker {
        color: var(--primary-color);
        font-weight: bold;
    }
    
    /* Feedback actions wrapper */
    .feedback-actions {
        display: flex;
        gap: 12px;
        margin-top: 24px;
        justify-content: flex-end;
        position: relative;
        z-index: 1;
    }
    
    /* Refined input fields */
    .stTextInput input, .stTextArea textarea {
        border-radius: var(--border-radius-md);
        border: 1px solid rgba(0,0,0,0.08);
        padding: 16px;
        box-shadow: var(--shadow-sm);
        transition: var(--transition-standard);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 16px;
        background-color: var(--card-bg);
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary-light);
        box-shadow: 0 0 0 3px rgba(255,139,61,0.15);
        transform: translateY(-2px);
    }
    
    /* Enhanced slider styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 2rem;
        margin-bottom: 3rem;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background: linear-gradient(145deg, var(--primary-color), var(--primary-dark));
        color: white;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: var(--border-radius-sm);
        font-size: 14px;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(145deg, var(--primary-color), var(--primary-dark));
        border: 3px solid white;
        box-shadow: var(--shadow-sm);
        width: 22px;
        height: 22px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.1);
        box-shadow: var(--shadow-md);
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stTrack"] {
        background: linear-gradient(90deg, var(--secondary-light), var(--primary-light));
        height: 8px;
        border-radius: 4px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Modern button styling */
    .stButton button {
        background: linear-gradient(145deg, var(--primary-color), var(--primary-dark));
        color: white;
        border-radius: var(--border-radius-md);
        padding: 12px 30px;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 12px rgba(224, 98, 24, 0.25);
        transition: var(--transition-standard);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 15px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(224, 98, 24, 0.35);
    }
    
    .stButton button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(224, 98, 24, 0.3);
    }
    
    .stButton button:hover::before {
        opacity: 1;
    }
    
    /* Refined notification messages */
    .stSuccess {
        background-color: var(--light-accent);
        border: none;
        padding: 22px;
        border-radius: var(--border-radius-md);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .stSuccess::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 6px;
        background: linear-gradient(to bottom, var(--secondary-color), var(--secondary-light));
    }
    
    .stError {
        background-color: #FFF5F5;
        border: none;
        padding: 22px;
        border-radius: var(--border-radius-md);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .stError::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 6px;
        background: linear-gradient(to bottom, #FF5A5A, #FF8080);
    }
    
    /* Refined decorative elements */
    .decoration-top {
        position: fixed;
        top: -140px;
        right: -140px;
        width: 500px;
        height: 500px;
        border-radius: 50%;
        background: radial-gradient(circle at center, rgba(255, 167, 107, 0.15) 0%, rgba(255, 139, 61, 0) 70%);
        z-index: -1;
        animation: float 20s infinite alternate ease-in-out;
    }
    
    .decoration-bottom {
        position: fixed;
        bottom: -140px;
        left: -140px;
        width: 500px;
        height: 500px;
        border-radius: 50%;
        background: radial-gradient(circle at center, rgba(142, 208, 129, 0.15) 0%, rgba(142, 208, 129, 0) 70%);
        z-index: -1;
        animation: float 15s infinite alternate-reverse ease-in-out;
    }
    
    /* Additional decorative element */
    .decoration-middle {
        position: fixed;
        top: 40%;
        left: -180px;
        width: 400px;
        height: 400px;
        border-radius: 50%;
        background: radial-gradient(circle at center, rgba(255, 182, 115, 0.12) 0%, rgba(255, 182, 115, 0) 70%);
        z-index: -1;
        animation: float 25s infinite alternate ease-in-out;
    }
    
    .decoration-accent {
        position: fixed;
        bottom: 30%;
        right: -120px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle at center, rgba(76, 175, 80, 0.08) 0%, rgba(76, 175, 80, 0) 70%);
        z-index: -1;
        animation: float 30s infinite alternate-reverse ease-in-out;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(15px, -15px) scale(1.03); }
        100% { transform: translate(-15px, 15px) scale(0.97); }
    }
    
    /* Modern card layout for options */
    .option-card {
        background-color: var(--card-bg);
        border-radius: var(--border-radius-lg);
        padding: 28px;
        margin: 18px 0;
        box-shadow: var(--shadow-md);
        transition: var(--transition-standard);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.6);
    }
    
    .option-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--secondary-color), var(--secondary-light));
        border-radius: 3px 3px 0 0;
    }
    
    .option-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg), 0 15px 35px rgba(61, 139, 64, 0.12);
    }
    
    /* Elegant scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, var(--accent-color), var(--primary-light));
        border-radius: 5px;
        border: 2px solid var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(to bottom, var(--primary-light), var(--primary-color));
    }
    
    /* Enhanced tooltip styling */
    .stTooltipIcon {
        color: var(--primary-light) !important;
        transition: color 0.2s ease;
    }
    
    .stTooltipIcon:hover {
        color: var(--primary-color) !important;
    }
    
    /* Checkbox and radio refinements */
    [data-testid="stCheckbox"], [data-testid="stRadio"] {
        padding: 12px;
        border-radius: var(--border-radius-sm);
        transition: background-color 0.2s ease;
    }
    
    [data-testid="stCheckbox"]:hover, [data-testid="stRadio"]:hover {
        background-color: var(--light-accent);
    }
    
    /* Select box styling */
    .stSelectbox label, .stMultiSelect label {
        color: var(--text-color);
        font-weight: 500;
    }
    
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: var(--card-bg);
        border-radius: var(--border-radius-md);
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: var(--shadow-sm);
        transition: var(--transition-standard);
    }
    
    .stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
        border-color: var(--primary-light);
        box-shadow: 0 0 0 3px rgba(255,139,61,0.1);
    }
    
    /* Form field labels */
    .stTextInput label, .stTextArea label, .stSlider label {
        font-weight: 500;
        color: var(--text-color);
        font-size: 1rem;
        margin-bottom: 8px;
    }
    
    /* Radio button enhancement */
    .stRadio > div {
        padding: 14px;
        background-color: var(--card-bg);
        border-radius: var(--border-radius-md);
        box-shadow: var(--shadow-sm);
        transition: var(--transition-standard);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .stRadio > div:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* Additional accent separators */
    .accent-separator {
        height: 3px;
        background: linear-gradient(90deg, var(--primary-light), var(--light-accent), var(--primary-light));
        margin: 45px 0;
        border-radius: 1.5px;
        position: relative;
    }
    
    .accent-separator::before {
        content: '';
        position: absolute;
        width: 40px;
        height: 40px;
        background: radial-gradient(circle at center, rgba(255, 167, 107, 0.1) 0%, rgba(255, 167, 107, 0) 70%);
        border-radius: 50%;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    
    /* Feedback badges for ratings */
    .feedback-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: var(--border-radius-sm);
        font-weight: 600;
        font-size: 14px;
        margin-right: 10px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-sm);
        transition: var(--transition-standard);
    }
    
    .feedback-badge:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .feedback-badge-positive {
        background: linear-gradient(145deg, var(--secondary-light), var(--secondary-color));
        color: white;
    }
    
    .feedback-badge-neutral {
        background: linear-gradient(145deg, #F0F0F0, #E0E0E0);
        color: var(--text-color);
    }
    
    .feedback-badge-negative {
        background: linear-gradient(145deg, #FF8080, #FF5A5A);
        color: white;
    }
    
    /* Enhanced feedback block styling */
    .feedback-block {
        position: relative;
        background-color: var(--card-bg);
        border-radius: var(--border-radius-lg);
        padding: 28px;
        margin: 24px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        transition: var(--transition-standard);
        border-left: 6px solid var(--primary-color);
        overflow: hidden;
    }
    
    .feedback-block::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle at top right, rgba(255, 167, 107, 0.1), rgba(255, 167, 107, 0) 70%);
        border-radius: 0 0 0 150px;
        z-index: 0;
    }
    
    .feedback-block:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .feedback-block-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        position: relative;
        z-index: 1;
    }
    
    .feedback-score {
        font-size: 24px;
        font-weight: 700;
        color: var(--primary-color);
        margin-right: 14px;
        background: linear-gradient(145deg, var(--primary-color), var(--primary-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .feedback-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.3px;
    }
    
    .feedback-metadata {
        font-size: 14px;
        color: var(--text-light);
        margin-top: 4px;
    }
    
    .feedback-content {
        position: relative;
        z-index: 1;
        line-height: 1.6;
        color: var(--text-color);
    }
    
    .feedback-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    
    .feedback-tag {
        background-color: var(--light-accent);
        color: var(--secondary-dark);
        font-size: 12px;
        font-weight: 600;
        padding: 5px 10px;
        border-radius: 20px;
        transition: var(--transition-standard);
    }
    
    .feedback-tag:hover {
        background-color: var(--secondary-light);
        color: white;
        transform: translateY(-2px);
    }
</style>

<div class="decoration-top"></div>
<div class="decoration-middle"></div>
<div class="decoration-bottom"></div>
<div class="decoration-accent"></div>

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

# File to store feedback
FEEDBACK_FILE = "feedback_data.json"

def load_feedback():
    """Load feedback from JSON file"""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted file
            return []
    return []

def save_feedback(feedback_list):
    """Save feedback to JSON file"""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)

def main():
    st.markdown("### We value your feedback! Please share your thoughts about our application.")
    
    # Load existing feedback
    feedback_list = load_feedback()
    
    # Create tabs for submission and viewing
    tab1, tab2 = st.tabs(["Submit Feedback", "View Feedback"])
    
    with tab1:
        # Create a container for the feedback form
        with st.container():
            # Text area for feedback
            feedback = st.text_area("Your Feedback", height=150, 
                                   placeholder="Please enter your feedback here...")
            
            # Optional name field
            name = st.text_input("Your Name (Optional)")
            
            # Rating system
            rating = st.slider("Rate your experience", 1, 5, 3)
            
            # Submit button
            submit_button = st.button("Submit Feedback")
            
            # Process the feedback when submitted
            if submit_button:
                if feedback:
                    # Create a unique ID for the feedback
                    feedback_id = str(uuid.uuid4())
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    feedback_entry = {
                        "id": feedback_id,
                        "feedback": feedback,
                        "name": name if name else "Anonymous",
                        "rating": rating,
                        "timestamp": timestamp
                    }
                    
                    # Add new feedback to the list
                    feedback_list.append(feedback_entry)
                    
                    # Save the updated list to file
                    save_feedback(feedback_list)
                    
                    display_feedback(feedback, name, rating)
                    st.success("Your feedback has been saved!")
                else:
                    st.error("Please enter your feedback before submitting.")
    
    with tab2:
        if feedback_list:
            st.markdown("## All Feedback")
            
            for i, entry in enumerate(feedback_list):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Feedback:** {entry['feedback']}")
                        st.markdown(f"**From:** {entry['name']}")
                        st.markdown(f"<small>Submitted on: {entry['timestamp']}</small>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**Rating:** {'⭐' * entry['rating']}")
                    
                    with col3:
                        if st.button("Delete", key=f"delete_{entry['id']}"):
                            feedback_list.remove(entry)
                            save_feedback(feedback_list)
                            st.rerun()
                
                # Add a separator between feedback entries
                if i < len(feedback_list) - 1:
                    st.markdown("---")
        else:
            st.info("No feedback submitted yet.")

def display_feedback(feedback, name, rating):
    """Display the submitted feedback"""
    st.markdown("## Your Submitted Feedback:")
    
    # Create a styled container for the feedback
    feedback_container = st.container()
    with feedback_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Feedback:** {feedback}")
            if name:
                st.markdown(f"**From:** {name}")
            else:
                st.markdown("**From:** Anonymous")
        
        with col2:
            st.markdown(f"**Rating:** {'⭐' * rating}")
    
    # Add some styling to make the feedback box stand out
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"]:has(div.element-container:has(p:contains("Feedback:"))):nth-of-type(n) {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()