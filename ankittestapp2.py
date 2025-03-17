import streamlit as st
import re
import os
import time
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import yfinance as yf
import datetime
from sklearn.cluster import KMeans
from PIL import Image
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from requests_oauthlib import OAuth2Session
import bcrypt

# -------------------- CONSTANTS AND CONFIG --------------------
PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{8,}$')
STOCK_MAP = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA"
}

# Google OAuth Configuration
GOOGLE_CLIENT_ID = "replace your cilent id"#due to github security region i removed my own google cilent id and cilent secret
GOOGLE_CLIENT_SECRET = "replace your cilent  secret id "#
REDIRECT_URI = "http://localhost:8501"

AUTHORIZATION_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# -------------------- INITIALIZATION --------------------
def init_session_state():
    required_states = {
        "users": {},
        "failed_attempts": {},
        "authenticated": False,
        "current_user": "Guest",
        "oauth_state": None
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_custom_styles():
    st.markdown("""
        <style>
        .main {background-color: #f5f7fa;}
        h1, h2, h3, h4 {color: #2c3e50;}
        .stButton>button {color: white; background-color: #4CAF50; border-radius: 8px;}
        </style>
    """, unsafe_allow_html=True)

# -------------------- SECURITY UTILITIES --------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode(), stored_hash.encode())

# -------------------- GOOGLE AUTH UTILITIES --------------------
def get_google_auth():
    return OAuth2Session(
        GOOGLE_CLIENT_ID,
        scope=SCOPES,
        redirect_uri=REDIRECT_URI
    )

def handle_google_auth():
    if 'code' in st.query_params and not st.session_state.authenticated:
        try:
            google = get_google_auth()
            token = google.fetch_token(
                TOKEN_URL,
                client_secret=GOOGLE_CLIENT_SECRET,
                authorization_response=st.query_params['code'],
            )
            id_info = id_token.verify_oauth2_token(
                token['id_token'],
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
            
            st.session_state.update({
                "authenticated": True,
                "current_user": id_info["email"],
                "oauth_state": None
            })
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")

# -------------------- AUTHENTICATION PAGE --------------------
def show_auth_page():
    st.title("🔐 Authentication")
    handle_google_auth()

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 🌐 Google Login")
        google = get_google_auth()
        authorization_url, state = google.authorization_url(
            AUTHORIZATION_BASE_URL,
            access_type="offline",
            prompt="select_account"
        )
        st.session_state.oauth_state = state
        st.markdown(f"""
            <a href="{authorization_url}" target="_self" style="
                display: inline-flex;
                align-items: center;
                padding: 0.5rem 1rem;
                background: #4285F4;
                color: white;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" 
                     width="20" style="margin-right: 10px;">
                Sign in with Google
            </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📧 Email Login")
        with st.expander("Login/Register"):
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type='password', key="login_password")
                
                if st.button("Login"):
                    handle_email_login(email, password)
            
            with tab2:
                new_email = st.text_input("New Email", key="register_email")
                new_pass = st.text_input("New Password", type='password', key="register_password")
                if st.button("Create Account"):
                    handle_email_registration(new_email, new_pass)

def handle_email_login(email, password):
    if email in st.session_state.users:
        stored_hash = st.session_state.users[email]['password']
        if verify_password(stored_hash, password):
            st.session_state.update({
                "authenticated": True,
                "current_user": email
            })
            st.rerun()
        else:
            st.error("Invalid password!")
    else:
        st.error("Email not registered!")

def handle_email_registration(email, password):
    if not PASSWORD_PATTERN.match(password):
        st.error("Password must have: 8+ chars, 1 uppercase, 1 lowercase, 1 number, 1 special char")
    elif email in st.session_state.users:
        st.error("Email already registered!")
    else:
        hashed_pw = hash_password(password)
        st.session_state.users[email] = {'password': hashed_pw}
        st.success("Account created! Please login")

# -------------------- MAIN APPLICATION PAGES --------------------
def main_app_interface():
    st.sidebar.title(f"🏦 BFSI OCR - {st.session_state.current_user}")
    
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state.update({
            "authenticated": False,
            "current_user": "Guest"
        })
        st.rerun()
    
    nav_options = ("🏠 Home", "📄 Document Analysis", "🎓 Student Loan")
    navigation = st.sidebar.radio("📂 Navigate to", nav_options)

    if navigation == "🏠 Home":
        show_home_page()
    elif navigation == "📄 Document Analysis":
        show_document_analysis()
    elif navigation == "🎓 Student Loan":
        show_student_loan_page()

def show_home_page():
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>🏦 BFSI OCR & Financial Insights</h1>", unsafe_allow_html=True)
    st.markdown("""### 🚀 Features
        - ✅ OCR-based Document Analysis  
        - ✅ Student Loan Eligibility  
        - ✅ Stock Market Visualizations  
        - ✅ Clustering & Visualization from CSV  
    """)

# -------------------- DOCUMENT ANALYSIS FUNCTIONS --------------------
def show_document_analysis():
    st.markdown("## 📄 Document Analysis")
    analysis_type = st.selectbox("Select Analysis Type", ["Supervised", "Semi-Supervised", "Unsupervised"])
    
    if analysis_type == "Supervised":
        handle_supervised_analysis()
    elif analysis_type == "Semi-Supervised":
        handle_semi_supervised_analysis()
    else:
        handle_unsupervised_analysis()

def handle_supervised_analysis():
    doc_type = st.selectbox("Choose Document Type", ["Bank Statement", "Invoice", "Payslip", "Profit and Loss"])
    uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file and st.button("Run OCR Analysis"):
        process_ocr_document(uploaded_file)

def process_ocr_document(uploaded_file):
    with st.spinner("Processing OCR..."):
        img = Image.open(uploaded_file).convert("RGB")
        extracted_text = pytesseract.image_to_string(img, lang='eng')
    
    st.success("✅ Extraction complete!")
    display_ocr_results(extracted_text)

def display_ocr_results(text):
    with st.expander("📝 Extracted Text"):
        st.text_area("Extracted Text", text, height=300, key="extracted_text_area")

    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)

    with st.expander("📊 Word Frequency Table"):
        st.dataframe(freq_df)

    display_visualizations(freq_df)

def display_visualizations(freq_df):
    top_n = min(10, len(freq_df))
    top_words = freq_df.head(top_n)

    st.markdown("#### 📈 Word Frequency - Bar Chart")
    st.pyplot(create_bar_chart(top_words))

    st.markdown("#### 🟢 Word Frequency - Pie Chart")
    st.pyplot(create_pie_chart(top_words))

def create_bar_chart(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["Word"], data["Frequency"], color=plt.cm.Pastel1.colors)
    ax.invert_yaxis()
    ax.set(xlabel="Frequency", title=f"Top {len(data)} Words Frequency")
    return fig

def create_pie_chart(data):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(data["Frequency"], labels=data["Word"], autopct='%1.1f%%',
           colors=plt.cm.Pastel2.colors, startangle=140)
    ax.set_title(f"Top {len(data)} Words Distribution")
    return fig

# -------------------- STUDENT LOAN PAGE --------------------
def show_student_loan_page():
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎓 Student Loan Recommendation</h1>", unsafe_allow_html=True)
    with st.form("loan_form"):
        name, age, tenth, twelfth, income, category = get_loan_inputs()
        if st.form_submit_button("Check Eligibility"):
            check_eligibility(name, age, tenth, twelfth, income, category)

def get_loan_inputs():
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", key="loan_name")
        age = st.number_input("Age", min_value=16, max_value=60, key="loan_age")
        tenth_score = st.number_input("10th Grade Score (%)", min_value=0.0, max_value=100.0, key="tenth_score")
    with col2:
        twelfth_score = st.number_input("12th Grade Score (%)", min_value=0.0, max_value=100.0, key="twelfth_score")
        family_income = st.number_input("Family Income (INR)", min_value=0, key="family_income")
        category = st.selectbox("Category", ["Undergraduate", "Postgraduate", "Abroad Studies"], key="loan_category")
    return name, age, tenth_score, twelfth_score, family_income, category

def check_eligibility(name, age, tenth, twelfth, income, category):
    st.markdown("### Eligibility Result")
    eligibility = True
    reasons = []
    
    if tenth < 75:
        eligibility = False
        reasons.append("10th score below 75%")
    if twelfth < 75:
        eligibility = False
        reasons.append("12th score below 75%")
        
    income_limits = {
        "Undergraduate": 1500000,
        "Postgraduate": 2000000,
        "Abroad Studies": 2500000
    }
    if income > income_limits.get(category, 1500000):
        eligibility = False
        reasons.append(f"Income exceeds limit for {category}")
        
    age_limits = {
        "Undergraduate": 25,
        "Postgraduate": 30,
        "Abroad Studies": 35
    }
    if age > age_limits.get(category, 25):
        eligibility = False
        reasons.append(f"Age exceeds limit for {category}")

    if eligibility:
        display_loan_offers(name, category)
    else:
        st.error(f"Sorry {name}, you are not eligible due to:")
        for reason in reasons:
            st.error(f"- {reason}")

def display_loan_offers(name, category):
    st.success(f"🎉 Congratulations {name}, you are eligible for {category} education loan!")
    st.markdown("### 📜 Bank Offers")
    
    loan_offers = {
        "Undergraduate": [
            {"Bank": "SBI Bank", "Interest Rate": "8.5%", "Max Loan": "15 Lakh", "Tenure": "7 Years"},
            {"Bank": "HDFC Bank", "Interest Rate": "9.0%", "Max Loan": "20 Lakh", "Tenure": "10 Years"},
            {"Bank": "Axis Bank", "Interest Rate": "9.5%", "Max Loan": "25 Lakh", "Tenure": "12 Years"}
        ],
        "Postgraduate": [
            {"Bank": "ICICI Bank", "Interest Rate": "8.0%", "Max Loan": "30 Lakh", "Tenure": "12 Years"},
            {"Bank": "Kotak Bank", "Interest Rate": "8.2%", "Max Loan": "35 Lakh", "Tenure": "15 Years"},
            {"Bank": "Bank of Baroda", "Interest Rate": "8.7%", "Max Loan": "40 Lakh", "Tenure": "15 Years"}
        ],
        "Abroad Studies": [
            {"Bank": "Axis Bank", "Interest Rate": "7.5%", "Max Loan": "1.5 Crore", "Tenure": "20 Years"},
            {"Bank": "IDFC First Bank", "Interest Rate": "7.8%", "Max Loan": "2 Crore", "Tenure": "20 Years"},
            {"Bank": "Yes Bank", "Interest Rate": "8.2%", "Max Loan": "1.75 Crore", "Tenure": "18 Years"}
        ]
    }
    
    for offer in loan_offers.get(category, []):
        with st.expander(f"{offer['Bank']} Offer"):
            st.markdown(f"**Interest Rate**: {offer['Interest Rate']}")
            st.markdown(f"**Maximum Loan Amount**: {offer['Max Loan']}")
            st.markdown(f"**Repayment Tenure**: {offer['Tenure']}")
            st.markdown(f"**Special Features**: {get_special_features(offer['Bank'])}")

def get_special_features(bank):
    features = {
        "SBI Bank": "No collateral required for loans up to 15 Lakh",
        "HDFC Bank": "Flexible repayment options after 1 year moratorium",
        "ICICI Bank": "Interest-only payments during study period",
        "Axis Bank": "Free forex card with international transactions",
        "Kotak Bank": "Career counseling services included",
        "Bank of Baroda": "Low processing fee of 0.5%",
        "IDFC First Bank": "Currency hedging options available",
        "Yes Bank": "Airport pickup service for international students"
    }
    return features.get(bank, "Contact bank for special features")

# -------------------- SEMI-SUPERVISED ANALYSIS --------------------
def handle_semi_supervised_analysis():
    st.header("📈 Live & Past Week Stock Market Analysis")
    stock_choice = st.selectbox("Select Stock", list(STOCK_MAP.keys()), key="stock_select")
    ticker_symbol = STOCK_MAP[stock_choice]

    try:
        today = datetime.date.today()
        one_week_ago = today - datetime.timedelta(days=7)
        ticker = yf.Ticker(ticker_symbol)

        display_live_price(ticker)
        display_historical_data(ticker_symbol, one_week_ago, today, stock_choice)
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

def display_live_price(ticker):
    todays_data = ticker.history(period="1d", interval="1m")
    if not todays_data.empty:
        latest_price = todays_data['Close'].iloc[-1]
        st.metric(label=f"💰 Live {ticker.info['shortName']} Price", value=f"${latest_price:.2f}")

def display_historical_data(ticker_symbol, start_date, end_date, stock_name):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        return st.warning(f"No historical data available for {stock_name}")

    st.subheader(f"📊 {stock_name} Stock Prices - Past Week")
    st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-', color='blue')
    ax.set(title=f"{stock_name} Closing Price (Last 7 Days)", xlabel="Date", ylabel="Price ($)")
    ax.grid(True)
    st.pyplot(fig)

# -------------------- UNSUPERVISED ANALYSIS --------------------
def handle_unsupervised_analysis():
    st.header("📊 CSV Clustering & Visualization")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")

    if uploaded_csv:
        process_clustering(uploaded_csv)

def process_clustering(uploaded_csv):
    df = pd.read_csv(uploaded_csv)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return st.warning("⚠️ The CSV must have at least 2 numeric columns for clustering.")

    st.subheader("Uploaded CSV Data")
    st.dataframe(df)

    k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3, key="cluster_slider")
    perform_clustering(numeric_df, k)

def perform_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data)

    st.subheader("📋 Data with Cluster Labels")
    st.dataframe(data)

    display_cluster_visualizations(data, k)

def display_cluster_visualizations(data, k):
    st.subheader("📊 Cluster Visualization (Scatter Plot)")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data["Cluster"], cmap='viridis')
    ax.set(xlabel=data.columns[0], ylabel=data.columns[1], title=f"KMeans Clustering (K={k})")
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    st.subheader("🟢 Cluster Distribution (Pie Chart)")
    cluster_counts = data["Cluster"].value_counts().sort_index()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
               autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    ax_pie.set_title("Cluster Distribution")
    st.pyplot(fig_pie)

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    init_session_state()
    apply_custom_styles()
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        main_app_interface()