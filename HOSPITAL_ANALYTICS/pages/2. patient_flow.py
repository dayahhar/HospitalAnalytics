import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_services(data_dir='data'):
    df = load_csv(f"{data_dir}/services_weekly.csv")
    # ensure services.week is numeric (week numbers like 1,2,3...)
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
    return df

def app():
    st.set_page_config(page_title="Patient Flow", layout="wide")
    st.title("Patient Flow")
    st.markdown("""
    Analyze patient flow trends across services and identify bottlenecks 
    using both **static (25%)** and **dynamic (4-week rolling)** thresholds.
    """)

    df = load_services()
    if df.empty:
        st.warning("No service data available.")
        return

    # 🧮 Compute refusal rate and bottleneck flags
    df["refusal_rate"] = df["patients_refused"] / df["patients_request"]
    df["rolling_mean"] = df.groupby("service")["refusal_rate"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df["rolling_std"] = df.groupby("service")["refusal_rate"].transform(lambda x: x.rolling(4, min_periods=1).std())
    
    # Static (25%) OR dynamic threshold
    df["is_bottleneck"] = (df["refusal_rate"] > 0.25) | (df["refusal_rate"] > df["rolling_mean"] + df["rolling_std"])

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    services = sorted(df['service'].dropna().unique().tolist())
    selected_service = st.sidebar.selectbox("Select Service", options=["All"] + services)
    week_min, week_max = int(df['week'].min()), int(df['week'].max())
    week_range = st.sidebar.slider("Select Week Range", week_min, week_max, (week_min, week_max))

    df_filtered = df.copy()
    if selected_service != "All":
        df_filtered = df_filtered[df_filtered['service'] == selected_service]
    df_filtered = df_filtered[(df_filtered['week'] >= week_range[0]) & (df_filtered['week'] <= week_range[1])]

    # --- Alerts ---
    alert_weeks = df_filtered[df_filtered['is_bottleneck']]
    if not alert_weeks.empty:
        st.error(f"🚨 Bottlenecks detected in {len(alert_weeks)} week(s)! Check the chart below.")
    else:
        st.success("✅ No bottlenecks detected in the selected range.")

    # --- Line Chart ---
    fig = px.line(
        df_filtered, 
        x='week', 
        y=['patients_request', 'patients_admitted', 'patients_refused'],
        title=f"Patient Admissions Over Time {'for ' + selected_service if selected_service != 'All' else ''}",
        labels={'value': 'Number of Patients', 'variable': 'Metric'}
    )
    fig.update_layout(legend_title_text='Metrics')

    # Highlight bottleneck weeks in red
    for _, row in df_filtered[df_filtered['is_bottleneck']].iterrows():
        fig.add_vrect(
            x0=row['week'] - 0.5, x1=row['week'] + 0.5,
            fillcolor="red", opacity=0.2, line_width=0
        )
    st.plotly_chart(fig, use_container_width=True)

    # --- Bottleneck Table ---
    st.subheader("🚨 Bottleneck Details")
    if not alert_weeks.empty:
        alert_display = alert_weeks[['week', 'service', 'patients_request', 'patients_refused', 'refusal_rate']].copy()
        alert_display['refusal_rate'] = alert_display['refusal_rate'].apply(lambda x: f"{x:.2%}")
        st.dataframe(alert_display)

        # --- CSV Export ---
        csv = alert_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Alert Data (CSV)",
            data=csv,
            file_name='bottleneck_alerts.csv',
            mime='text/csv'
        )
    else:
        st.info("No bottlenecks to display.")

app()