def frontend():
    import streamlit as st
    
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
