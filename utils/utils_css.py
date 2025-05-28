
# Custom CSS for better styling
def better_styling_css():
    return """
    <style>
        .main-header {
            font-size: 2.5rem !important;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .sub-header {
            font-size: 1.8rem !important;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .section-header {
            font-size: 1.4rem !important;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .highlight-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .success-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .info-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .warning-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .metric-container {
            text-align: center;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .metric-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .metric-label {
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
        }
        .divider {
            height: 1px;
            margin: 2rem 0;
        }
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            border-radius: 0.5rem 0.5rem 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        /* Button styling */
        .stButton > button {
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s;
        }
        .stButton > button:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        /* Input styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 0.5rem;
            border: 2px solid;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        .stSelectbox > div > div {
            border-radius: 0.5rem;
            border: 2px solid;
        }
    </style>
    """
