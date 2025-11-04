import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hospital Analytics Dashboard", layout="wide"
)

# --- HEADER ---
st.markdown("""
<div style="
    text-align: center;
    padding: 30px 0;
">
    <h1 style="margin-bottom: 10px;">🏥 Hospital Analytics Dashboard</h1>
    <p style="font-size: 18px; color: #555;">
        A simple and friendly dashboard to understand hospital capacity,
        staffing utilisation, and patient service demand.
    </p>
</div>
""", unsafe_allow_html=True)

# --- QUICK LINKS / HINTS ---
st.markdown("### 🔍 Quick Navigation")
st.write("""
- 📈 **Overview** – Key hospital metrics  
- 🧑‍⚕️ **Staff Analytics** – Workforce & attendance patterns  
- 🛏️ **Bed & Service Utilisation** – Demand vs capacity  
- 📊 **Patient Flow** – Service congestion & turnaround  
- 🤖 **Demand Prediction** – Machine learning forecast
""")

# --- FOOTER ---
st.markdown("""
<br><hr style="opacity:0.2;">
<div style="text-align: center; color: #777; font-size: 14px; padding: 10px 0;">
Made with ❤️ for better hospital planning & insight.
</div>
""", unsafe_allow_html=True)
