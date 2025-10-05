import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with your color scheme
st.markdown("""
    <style>
    /* Main color scheme */
    :root {
        --shadowed-green: #1C2529;
        --mint-green: #A1D1B1;
        --accent-light: #C8E6D0;
        --text-dark: #1C2529;
        --text-light: #ffffff;
    }
    
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: var(--mint-green) !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: var(--mint-green) !important;
        font-weight: 600 !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, var(--mint-green) 0%, var(--accent-light) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(28, 37, 41, 0.15);
        margin: 1rem 0;
        border: 2px solid var(--shadowed-green);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(28, 37, 41, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--shadowed-green) 0%, #2a3f47 100%) !important;
        color: var(--mint-green) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 12px rgba(28, 37, 41, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a3f47 0%, var(--shadowed-green) 100%) !important;
        box-shadow: 0 8px 16px rgba(28, 37, 41, 0.4) !important;
        transform: translateY(-2px);
        color: var(--text-light) !important;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > div,
    .stSelectbox > div > div > div,
    .stTextInput > div > div > div {
        padding: 0.5rem !important;
        background-color: var(--shadowed-green) !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > div >:focus,
    .stSelectbox > div > div > div:focus,
    .stTextInput > div > div > div:focus {
        border-color: var(--shadowed-green) !important;
        box-shadow: 0 0 0 2px rgba(161, 209, 177, 0.3) !important;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 12px !important;
        border-left: 5px solid var(--shadowed-green) !important;
        background-color: var(--accent-light) !important;
        padding: 1rem !important;
        box-shadow: 0 4px 8px rgba(28, 37, 41, 0.1);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--mint-green) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--mint-green) !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling - MINT GREEN TEXT ON DARK BACKGROUND */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--shadowed-green) 0%, #2a3f47 100%) !important;
    }
    
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--mint-green) !important;
    }
    
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--mint-green) !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--mint-green), transparent);
        margin: 2rem 0;
    }
    
    /* DataFrame styling */
    .dataframe {
        border: 2px solid var(--mint-green) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: var(--mint-green) !important;
        color: var(--mint-green) !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        padding: 0.5rem !important;
        border-bottom: 1px solid var(--accent-light) !important;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, var(--mint-green) 0%, var(--accent-light) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--shadowed-green);
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(28, 37, 41, 0.15);
    }
    
    /* Success/Error message boxes with shadowed green background */
    .element-container:has(.stSuccess) {
        background: linear-gradient(135deg, var(--shadowed-green) 0%, #2a3f47 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 5px solid #28a745;
    }

    .element-container:has(.stSuccess) * {
        color: var(--shadowed-green) !important;
    }

    .element-container:has(.stError) {
        background: linear-gradient(135deg, var(--shadowed-green) 0%, #2a3f47 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 5px solid #dc3545;
    }

    .element-container:has(.stError) * {
        color: var(--shadowed-green) !important;
    }
        
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--mint-green) !important;
        color: var(--shadowed-green) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    /* Section headers - MINT GREEN TEXT ON DARK BACKGROUND */
    .section-header {
        background: linear-gradient(90deg, var(--shadowed-green) 0%, #2a3f47 100%);
        color: var(--mint-green) !important;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 4px 8px rgba(28, 37, 41, 0.2);
        border-left: 5px solid var(--mint-green);
    }
    
    .section-header h2,
    .section-header h3 {
        color: var(--mint-green) !important;
    }
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background-color: var(--mint-green);
        color: var(--shadowed-green);
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Dark info box - MINT GREEN TEXT */
    .dark-info-box {
        background: linear-gradient(135deg, var(--shadowed-green) 0%, #2a3f47 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--mint-green);
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(28, 37, 41, 0.3);
        color: var(--mint-green) !important;
    }
    
    .dark-info-box p,
    .dark-info-box li,
    .dark-info-box h4 {
        color: var(--mint-green) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model():
    with open('customer_churn_xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    return model, encoder

try:
    model, encoder = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'customer_churn_xgboost_model.pkl' and 'encoder.pkl' are in the same directory.")
    st.stop()

# Title and description with custom header
st.markdown('<h1>Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class='info-box'>
<p style='text-align: center; font-size: 1.1rem; margin: 0; color: #1C2529;'>
<b>AI-Powered Predictive Analytics</b> | Identify at-risk customers and take proactive retention actions
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Create two columns for input
st.markdown("<div class='section-header'><h2 style='margin: 0;'>Customer Information</h2></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Demographics & Engagement")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Customer's age")
    gender = st.selectbox("Gender", options=['Male', 'Female'], help="Customer's gender")
    
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24, help="How long the customer has been with the company")
    usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=30, value=15, help="How often customer uses the service")
    last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=5, help="Days since last customer interaction")

with col2:
    st.markdown("### 💼 Subscription & Financial")
    subscription_type = st.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'], help="Current subscription plan")
    contract_length = st.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'], help="Contract duration")
    
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0, help="Total amount spent by customer")
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=90, value=0, help="Average payment delay in days")
    support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=2, help="Number of support calls made")

st.divider()

# Prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True)

if predict_button:
    try:
        # Prepare input data
        ohe_data = list(encoder.transform([[gender, subscription_type, contract_length]])[0])
        input_array = [age, tenure, usage_frequency, support_calls, payment_delay, 
                      total_spend, last_interaction] + ohe_data
        input_array = np.array(input_array).reshape((1, -1))
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        # Display results
        st.markdown("<div class='section-header'><h2 style='margin: 0;'>Prediction Results</h2></div>", unsafe_allow_html=True)
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            churn_prob = prediction_proba[1] * 100
            st.metric(
                label="Churn Probability",
                value=f"{churn_prob:.1f}%",
                delta=f"{churn_prob - 50:.1f}%" if churn_prob > 50 else f"{churn_prob - 50:.1f}%",
                delta_color="inverse"
            )
        
        with metric_col2:
            risk_emoji = "🔴" if prediction == 1 else "🟢"
            st.metric(
                label="⚡ Risk Level",
                value=f"{risk_emoji} {'High Risk' if prediction == 1 else 'Low Risk'}"
            )
        
        with metric_col3:
            retention_prob = prediction_proba[0] * 100
            st.metric(
                label="Retention Probability",
                value=f"{retention_prob:.1f}%"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display prediction result with color coding
        if prediction == 1:
            # Custom error message with shadowed green background
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1C2529 0%, #2a3f47 100%); 
                        border-radius: 12px; 
                        padding: 1.5rem; 
                        border-left: 5px solid #dc3545;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(28, 37, 41, 0.3);'>
                <h3 style='color: #A1D1B1; margin: 0;'>⚠️ HIGH RISK ALERT: This customer is likely to churn!</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='section-header'><h3 style='margin: 0;'>Recommended Retention Actions</h3></div>", unsafe_allow_html=True)
            
            actions_col1, actions_col2 = st.columns(2)
            
            with actions_col1:
                st.markdown("""
                <div class='metric-card'>
                <h4 style='color: #1C2529; margin-top: 0;'>🚨 Immediate Actions</h4>
                <ul style='color: #1C2529;'>
                <li><b>Priority Contact:</b> Reach out within 24 hours</li>
                <li><b>Personal Touch:</b> Assign dedicated account manager</li>
                <li><b>Special Offer:</b> Provide exclusive retention discount</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with actions_col2:
                st.markdown("""
                <div class='metric-card'>
                <h4 style='color: #1C2529; margin-top: 0;'>Strategic Moves</h4>
                <ul style='color: #1C2529;'>
                <li><b>Loyalty Program:</b> Enroll in VIP rewards tier</li>
                <li><b>Service Audit:</b> Review and optimize their plan</li>
                <li><b>Feedback Loop:</b> Schedule satisfaction survey</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.markdown("<div class='section-header'><h3 style='margin: 0;'>Key Risk Factors Identified</h3></div>", unsafe_allow_html=True)
            
            risk_factors = []
            
            if payment_delay > 15:
                risk_factors.append(("🔴 High Payment Delay", f"{payment_delay} days", "Critical"))
            if support_calls > 5:
                risk_factors.append(("🟠 Frequent Support Issues", f"{support_calls} calls", "High"))
            if usage_frequency < 5:
                risk_factors.append(("🟡 Low Engagement", f"{usage_frequency} uses", "Medium"))
            if last_interaction > 30:
                risk_factors.append(("🟠 Inactive Customer", f"{last_interaction} days ago", "High"))
            if tenure < 6:
                risk_factors.append(("🟡 New Customer Risk", f"{tenure} months", "Medium"))
            
            if risk_factors:
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                for idx, (factor, value, severity) in enumerate(risk_factors):
                    col = [risk_col1, risk_col2, risk_col3][idx % 3]
                    with col:
                        # Custom warning box with mint green on shadowed green
                        col.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1C2529 0%, #2a3f47 100%); 
                                    border-radius: 10px; 
                                    padding: 1rem; 
                                    border-left: 4px solid #ffc107;
                                    margin: 0.5rem 0;
                                    box-shadow: 0 3px 6px rgba(28, 37, 41, 0.2);'>
                            <p style='color: #A1D1B1; margin: 0; font-weight: 600;'>{factor}</p>
                            <p style='color: #A1D1B1; margin: 0.5rem 0; font-size: 1.1rem;'>{value}</p>
                            <p style='color: #A1D1B1; margin: 0; font-size: 0.9rem;'>Severity: {severity}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #1C2529 0%, #2a3f47 100%); 
                            border-radius: 12px; 
                            padding: 1rem; 
                            border-left: 5px solid #17a2b8;
                            margin: 1rem 0;
                            box-shadow: 0 4px 8px rgba(28, 37, 41, 0.3);'>
                    <p style='color: #A1D1B1; margin: 0;'>No major risk factors identified. Churn prediction based on pattern analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            # Custom success message with shadowed green background
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1C2529 0%, #2a3f47 100%); 
                        border-radius: 12px; 
                        padding: 1.5rem; 
                        border-left: 5px solid #28a745;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(28, 37, 41, 0.3);'>
                <h3 style='color: #A1D1B1; margin: 0;'>✅ LOW RISK: Customer is likely to stay!</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='section-header'><h3 style='margin: 0;'>Growth & Retention Strategy</h3></div>", unsafe_allow_html=True)
            
            strategy_col1, strategy_col2 = st.columns(2)
            
            with strategy_col1:
                st.markdown("""
                <div class='metric-card'>
                <h4 style='color: #1C2529; margin-top: 0;'>Strengthen Relationship</h4>
                <ul style='color: #1C2529;'>
                <li><b>Regular Updates:</b> Keep them informed of new features</li>
                <li><b>Appreciation:</b> Send thank you message or reward</li>
                <li><b>Feedback:</b> Request testimonial or review</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with strategy_col2:
                st.markdown("""
                <div class='metric-card'>
                <h4 style='color: #1C2529; margin-top: 0;'>Upsell Opportunities</h4>
                <ul style='color: #1C2529;'>
                <li><b>Premium Features:</b> Introduce advanced capabilities</li>
                <li><b>Referral Program:</b> Encourage friend invitations</li>
                <li><b>Cross-sell:</b> Offer complementary services</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        
        # Customer Profile Summary
        st.markdown("<div class='section-header'><h3 style='margin: 0;'>Customer Profile Summary</h3></div>", unsafe_allow_html=True)
        
        profile_data = {
            "Metric": ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"],
            "Value": [age, f"{tenure} months", usage_frequency, support_calls, f"{payment_delay} days", f"${total_spend:.2f}", f"{last_interaction} days ago"],
            "Status": [
                "✅ Optimal" if 25 <= age <= 55 else "⚠️ Monitor",
                "✅ Loyal" if tenure > 12 else "⚠️ New",
                "✅ Active" if usage_frequency > 10 else "⚠️ Low",
                "✅ Satisfied" if support_calls < 5 else "⚠️ Issues",
                "✅ On-time" if payment_delay < 10 else "⚠️ Delayed",
                "✅ High Value" if total_spend > 300 else "⚠️ Low Spend",
                "✅ Engaged" if last_interaction < 15 else "⚠️ Inactive"
            ]
        }
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# Sidebar with additional information
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Dashboard Info</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='dark-info-box'>
    <p style='margin: 0;'>This dashboard uses an <b>XGBoost ML model</b> trained on historical customer data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Performance")
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.markdown("<span class='badge'>Accuracy: 93%</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge'>Recall: 99%</span>", unsafe_allow_html=True)
    with perf_col2:
        st.markdown("<span class='badge'>Precision: 90%</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge'>F1-Score: 94%</span>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Enter customer details
    2. Click prediction button
    3. Review risk assessment
    4. Execute retention strategy
    """)
    
    st.divider()
    
    st.markdown("### Feature Guide")
    with st.expander("View Details"):
        st.markdown("""
        - **Tenure**: Relationship duration
        - **Usage**: Service utilization rate
        - **Support**: Service interactions
        - **Payment**: Payment behavior
        - **Spend**: Financial value
        - **Interaction**: Engagement recency
        """)
    
    st.divider()
    
    st.markdown("""
    <div class='dark-info-box' style='text-align: center;'>
    <p style='margin: 0; font-size: 0.9rem;'>
    Built using Streamlit & XGBoost 
    </p>
    <p style='margin: 0; font-size: 0.9rem;'> 
    Project by - Ayush Goswami
    </p>
    </div>
    """, unsafe_allow_html=True)
