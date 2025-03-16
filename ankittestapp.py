import streamlit as st
import re
from PIL import Image
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import yfinance as yf
import datetime
from sklearn.cluster import KMeans
from streamlit.components.v1 import html
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import os
import json
from urllib.parse import urlparse, parse_qs

# -------------------- CONSTANTS AND CONFIG --------------------
PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{8,}$')
STOCK_MAP = {
    "Apple": "AAPL",
    "Google": "GOOG",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA"
}

# Google OAuth Configuration
CLIENT_CONFIG = {
    "web": {
        "client_id": "replace with cilend id",
        "client_secret": "replace with cilent secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8501"]
    }
}
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", 
          "https://www.googleapis.com/auth/userinfo.profile"]
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # For local development only

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="BFSI OCR", layout="wide")

# -------------------- HELPER FUNCTIONS --------------------
def init_session_state():
    """Initialize all required session state variables"""
    required_states = {
        "users": {},
        "authenticated": False,
        "current_user": "Guest",
        "google_auth_processed": False,
        "auth_flow": None
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_custom_styles():
    st.markdown("""
        <style>
        .main {background-color: #f5f7fa;}
        h1, h2, h3, h4 {color: #2c3e50;}
        .stButton>button {color: white; background-color: #4CAF50; border-radius: 8px; height: 3em; width: 100%;}
        .stTextInput>div>div>input, .stNumberInput>div>div>input {border-radius: 8px;}
        </style>
    """, unsafe_allow_html=True)

def get_authorization_url():
    """Generate Google OAuth authorization URL"""
    flow = Flow.from_client_config(
        client_config=CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=CLIENT_CONFIG['web']['redirect_uris'][0]
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    st.session_state.auth_flow = flow
    return authorization_url

def verify_google_token(code):
    """Verify Google OAuth token and return user info"""
    try:
        flow = st.session_state.auth_flow
        flow.fetch_token(code=code)
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            Request(),
            CLIENT_CONFIG['web']['client_id']
        )
        return id_info
    except Exception as e:
        st.error(f"Token verification failed: {str(e)}")
        return None

def handle_authentication():
    """Authentication page with Google and traditional login"""
    st.title("🔐 Authentication")
    
    # Google Login
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 🌐 Google Login")
        authorization_url = get_authorization_url()
        html(f"""
            <a href="{authorization_url}" target="_top">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" 
                     width="30" style="margin-right: 10px;">
                Sign in with Google
            </a>
        """, height=50)

    # Traditional Login
    with col2:
        st.markdown("### 📧 Email Login")
        with st.expander("Login/Register"):
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                username = st.text_input("Username")
                password = st.text_input("Password", type='password')
                if st.button("Sign In"):
                    if username in st.session_state.users:
                        if st.session_state.users[username]['password'] == password:
                            st.session_state.update({
                                "authenticated": True,
                                "current_user": username
                            })
                            st.rerun()
                        else:
                            st.error("Incorrect password!")
                    else:
                        st.error("User not found!")
            
            with tab2:
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type='password')
                if st.button("Create Account"):
                    if not PASSWORD_PATTERN.match(new_pass):
                        st.error("Password must contain 8+ chars with uppercase, lowercase, number, and special char")
                    elif new_user in st.session_state.users:
                        st.error("Username already exists!")
                    else:
                        st.session_state.users[new_user] = {'password': new_pass}
                        st.success("Account created! Please login")

# ... [Keep
 
                          
def create_bar_chart(data, title, color_scheme='Pastel1'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["Word"], data["Frequency"], color=plt.cm.get_cmap(color_scheme).colors)
    ax.invert_yaxis()
    ax.set(xlabel="Frequency", title=title)
    return fig

def create_pie_chart(data, title, color_scheme='Pastel2'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(data["Frequency"], labels=data["Word"], autopct='%1.1f%%',
           colors=plt.cm.get_cmap(color_scheme).colors, startangle=140)
    ax.set_title(title)
    return fig                      

# -------------------- MAIN APP FUNCTIONALITY --------------------
def main_app():
    st.sidebar.title(f"🏦 BFSI OCR - {st.session_state.current_user}")
    
    # Add Sign Out button
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state.update({
            "authenticated": False,
            "current_user": "Guest",
            "google_auth_processed": False
        })
        st.rerun()
    
    nav_options = ("🏠 Home", "📄 Document Analysis", "🎓 Student Loan")
    navigation = st.sidebar.radio("📂 Navigate to", nav_options)

    if navigation == "🏠 Home":
        render_home()
    elif navigation == "📄 Document Analysis":
        render_document_analysis()
    elif navigation == "🎓 Student Loan":
        render_student_loan()

def render_home():
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>🏦 BFSI OCR & Financial Insights</h1>", unsafe_allow_html=True)
    st.markdown("""### 🚀 Features
            - ✅ OCR-based Document Analysis  
        - ✅ Student Loan Eligibility  
        - ✅ Stock Market Visualizations  
        - ✅ Clustering & Visualization from CSV  
    """)

def render_document_analysis():
    st.markdown("## 📄 Document Analysis")
    analysis_type = st.selectbox("Select Analysis Type", ["Supervised", "Semi-Supervised", "Unsupervised"])

    if analysis_type == "Supervised":
        handle_supervised()
    elif analysis_type == "Semi-Supervised":
        handle_semi_supervised()
    else:
        handle_unsupervised()

def handle_supervised():
    doc_type = st.selectbox("Choose Document Type", ["Bank Statement", "Invoice", "Payslip", "Profit and Loss"])
    uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])

    if uploaded_file and st.button("Run OCR Extraction & Analysis"):
        process_ocr(uploaded_file)

def process_ocr(uploaded_file):
    with st.spinner("Processing OCR..."):
        img = Image.open(uploaded_file).convert("RGB")
        extracted_text = pytesseract.image_to_string(img, lang='eng')
    
    st.success("✅ Extraction complete!")
    display_ocr_results(extracted_text)

def display_ocr_results(text):
    with st.expander("📝 Extracted Text"):
        st.text_area("Extracted Text", text, height=300)

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
    st.pyplot(create_bar_chart(top_words, f"Top {top_n} Words Frequency (Bar Chart)"))

    st.markdown("#### 🟢 Word Frequency - Pie Chart")
    st.pyplot(create_pie_chart(top_words, f"Top {top_n} Words Distribution (Pie Chart)"))

def handle_semi_supervised():
    st.header("📈 Live & Past Week Stock Market Analysis (Semi-Supervised)")
    stock_choice = st.selectbox("Select Stock", list(STOCK_MAP.keys()))
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

    st.subheader(f"📊 {stock_name} Stock Prices - Past 7 Days")
    st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-', color='blue')
    ax.set(title=f"{stock_name} Closing Price (Last 7 Days)", xlabel="Date", ylabel="Price ($)")
    ax.grid(True)
    st.pyplot(fig)

def handle_unsupervised():
    st.header("📊 CSV Clustering & Visualization (Unsupervised)")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_csv:
        process_clustering(uploaded_csv)

def process_clustering(uploaded_csv):
    df = pd.read_csv(uploaded_csv)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return st.warning("⚠️ The CSV must have at least 2 numeric columns for clustering.")

    st.subheader("Uploaded CSV Data")
    st.dataframe(df)

    k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)
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

def render_student_loan():
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎓 Student Loan Recommendation</h1>", unsafe_allow_html=True)
    with st.form("loan_form"):
        name, age, tenth, twelfth, income, category = get_loan_inputs()
        if st.form_submit_button("Check Eligibility"):
            check_eligibility(name, age, tenth, twelfth, income, category)

def get_loan_inputs():
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=16, max_value=60)
        tenth_score = st.number_input("10th Grade Score (%)", min_value=0.0, max_value=100.0)
    with col2:
        twelfth_score = st.number_input("12th Grade Score (%)", min_value=0.0, max_value=100.0)
        family_income = st.number_input("Family Income (INR)", min_value=0)
        category = st.selectbox("Category", ["Undergraduate", "Postgraduate", "Abroad Studies"])
    return name, age, tenth_score, twelfth_score, family_income, category

def check_eligibility(name, age, tenth, twelfth, income, category):
    st.markdown("### Eligibility Result")
    if tenth >= 60 and twelfth >= 60 and age <= 35:
        display_loan_offers(name)
    else:
        st.error(f"Sorry {name}, you are not eligible based on the provided information.")

def display_loan_offers(name):
    st.success(f"🎉 Congratulations {name}, you are eligible for an education loan!")
    st.markdown("### 📜 Bank Offers")
    sample_offers = [
        {"Bank": "SBI Bank", "Interest Rate": "8.5%", "Max Loan": "10 Lakh", "Tenure": "5 Years"},
        {"Bank": "ICICI Bank", "Interest Rate": "9%", "Max Loan": "7 Lakh", "Tenure": "7 Years"},
        {"Bank": "HDFC Bank", "Interest Rate": "7.5%", "Max Loan": "12 Lakh", "Tenure": "10 Years"},
    ]
    for offer in sample_offers:
        st.markdown(f"**🏦 Bank**: {offer['Bank']}")
        st.markdown(f"**💰 Interest Rate**: {offer['Interest Rate']}")
        st.markdown(f"**📈 Max Loan**: {offer['Max Loan']}")
        st.markdown(f"**🕒 Tenure**: {offer['Tenure']}")
        st.markdown("---")
 
    
 
 
 
 
 
# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    init_session_state()
    apply_custom_styles()
    
    # Handle Google OAuth callback
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.authenticated and not st.session_state.google_auth_processed:
        try:
            code = query_params['code'][0]
            user_info = verify_google_token(code)
            if user_info:
                st.session_state.update({
                    "authenticated": True,
                    "current_user": user_info['email'],
                    "google_auth_processed": True
                })
                # Clear query parameters
                st.query_params.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")

    if not st.session_state.authenticated:
        handle_authentication()
    else:
        main_app()