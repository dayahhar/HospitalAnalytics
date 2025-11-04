import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_all(data_dir='data'):
    staff = load_csv(f"{data_dir}/staff.csv")
    patients = load_csv(f"{data_dir}/patients.csv")
    services = load_csv(f"{data_dir}/services_weekly.csv")
    schedules = load_csv(f"{data_dir}/staff_schedule.csv")

    if 'week' in services.columns:
        services['week'] = pd.to_numeric(services['week'], errors='coerce').astype('Int64')
    if 'week' in schedules.columns:
        schedules['week'] = pd.to_numeric(schedules['week'], errors='coerce').astype('Int64')

    if 'arrival_date' in patients.columns:
        patients['arrival_date'] = pd.to_datetime(patients['arrival_date'], errors='coerce')
        patients['arrival_week'] = patients['arrival_date'].dt.isocalendar().week.astype('Int64')
    else:
        patients['arrival_week'] = pd.NA

    return staff, patients, services, schedules

def refusal_tint(rate):
    if pd.isna(rate):
        return 'background: rgba(255,255,255,0.6); color: #111;'
    if rate < 0.15:
        return 'background: rgba(220, 255, 220, 0.7); border: 1px solid rgba(34,135,56,0.12); color: #0b6623;'
    if rate <= 0.30:
        return 'background: rgba(255, 250, 205, 0.8); border: 1px solid rgba(204,170,50,0.12); color: #8a6d00;'
    return 'background: rgba(255, 230, 230, 0.9); border: 1px solid rgba(180,60,60,0.12); color: #7a1f1f;'

def render_card(title, value, subtitle='', tint_style='', width='100%'):
    card_html = f"""
    <div style="{tint_style} padding:14px; border-radius:10px; backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); width:{width};">
        <div style='font-size:14px; color: rgba(0,0,0,0.6);'>{title}</div>
        <div style='font-size:26px; font-weight:700; margin-top:6px;'>{value}</div>
        <div style='font-size:12px; color: rgba(0,0,0,0.45); margin-top:6px;'>{subtitle}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def app():
    st.set_page_config(page_title="Bed and Service Utilisation", layout="wide")
    st.title("Bed and Service Utilisation")

    staff, patients, services, schedules = load_all()

    week_vals =[]
    if 'week' in services.columns:
        week_vals.extend([int(w) for w in services['week'].dropna().unique()])
    if 'week' in schedules.columns:
        week_vals.extend([int(w) for w in schedules['week'].dropna().unique()])
    if 'arrival_week' in patients.columns:
        week_vals.extend([int(w) for w in patients['arrival_week'].dropna().unique()])

    week_vals = sorted(list(set(week_vals))) if week_vals else [1]
    week_min, week_max = min(week_vals), max(week_vals)

    st.sidebar.subheader("Filter by Week Range")
    week_range = st.sidebar.slider('Select Week Range', min_value=int(week_min), max_value=int(week_max), value=(int(week_min), int(week_max)))
    selected_weeks = list(range(week_range[0], week_range[1] + 1))

    services_list = services['service'].unique().tolist() if 'service' in services.columns else []
    selected_services = st.sidebar.selectbox('Service filter (All = hospital)', options=['All'] + sorted(services_list), index=0)

    svc = services.copy()
    if 'week' in svc.columns:
        svc = svc[svc['week'].isin(selected_weeks)]
    if selected_services != 'All':
        svc = svc[svc['service'] == selected_services]

    # KPI calculations
    total_beds = int(svc['available_beds'].sum()) if 'available_beds' in svc.columns else 0
    total_requests = int(svc['patients_request'].sum()) if 'patients_request' in svc.columns else 0
    total_admitted = int(svc['patients_admitted'].sum()) if 'patients_admitted' in svc.columns else 0
    total_refused = int(svc['patients_refused'].sum()) if 'patients_refused' in svc.columns else 0


    bed_utilization = (total_admitted / total_beds) if total_beds else np.nan
    admission_rate = (total_admitted / total_requests) if total_requests else np.nan
    refusal_rate = (total_refused / total_requests) if total_requests else np.nan

    # KPI cards layout (we have 7 cards; split into two rows)
    st.markdown('<div style="display:flex; gap:12px; flex-wrap:wrap;">', unsafe_allow_html=True)
    
    # Card 1: Total Available Beds
    render_card('Total Available Beds', f"{total_beds:,}", subtitle='Sum of available_beds in selected range', tint_style='background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.04);')
   
    # Card 2: Total Patient Requests
    render_card('Total Patient Requests', f"{total_requests:,}", subtitle='Sum of patients_request', tint_style='background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.04);')
    
    # Card 3: Total Admitted
    render_card('Total Admitted', f"{total_admitted:,}", subtitle='Sum of patients_admitted', tint_style='background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.04);')
    
    # Card 4: Total Refused
    render_card('Total Refused', f"{total_refused:,}", subtitle='Sum of patients_refused', tint_style='background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.04);')
    
    # Card 5: Bed Utilization Rate
    tint_bed = refusal_tint(refusal_rate)
    render_card('Bed Utilization Rate', f"{bed_utilization:.2%}" if pd.notna(bed_utilization) else 'N/A', subtitle='Admitted / Available Beds', tint_style='background: rgba(240,248,255,0.65); border: 1px solid rgba(0,0,0,0.04);')
    
    # Card 6: Admission Rate
    render_card('Admission Rate', f"{admission_rate:.2%}" if pd.notna(admission_rate) else 'N/A', subtitle='Admitted / Requests', tint_style='background: rgba(245,250,240,0.7); border: 1px solid rgba(0,0,0,0.04);')
    
    # Card 7: Refusal Rate (color-coded)
    render_card('Refusal Rate', f"{refusal_rate:.2%}" if pd.notna(refusal_rate) else 'N/A', subtitle='Refused / Requests', tint_style=refusal_tint(refusal_rate))
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Heatmap: service x week occupancy ratio (admitted / available_beds)
    st.header('Heatmap — Occupancy ratio (admitted / available_beds)')
    if 'week' in services.columns and 'available_beds' in services.columns and 'patients_admitted' in services.columns:
        heat = services.copy()
        if selected_services != 'All':
            heat = heat[heat['service'] == selected_services]
        heat = heat[heat['week'].isin(selected_weeks)]
        if heat.empty:
            st.info('No heatmap data for selected filters')
        else:
            heat['occupancy'] = heat['patients_admitted'] / heat['available_beds'].replace({0: np.nan})
            pivot = heat.pivot_table(index='service', columns='week', values='occupancy', aggfunc='mean', fill_value=np.nan)

            # display as heatmap
            fig_h = px.imshow(pivot,
                              labels=dict(x='Week', y='Service', color='Occupancy'),
                              aspect='auto', text_auto=False)
            fig_h.update_traces(hovertemplate='Service: %{y}<br>Week: %{x}<br>Occupancy: %{z:.2%}<extra></extra>')
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info('services_weekly.csv missing necessary columns for heatmap (week/available_beds/patients_admitted).')

    st.divider()

    # Detailed service table + grouped bar chart
    st.header('Service Breakdown')
    if 'service' in svc.columns:
        svc_summary = svc.groupby('service').agg({
            'available_beds': 'sum',
            'patients_request': 'sum',
            'patients_admitted': 'sum',
            'patients_refused': 'sum',
            'patient_satisfaction': 'mean'
        }).reset_index()
        svc_summary['bed_utilization'] = (svc_summary['patients_admitted'] / svc_summary['available_beds']).replace([np.inf, -np.inf], np.nan)
        svc_summary['admission_rate'] = (svc_summary['patients_admitted'] / svc_summary['patients_request']).replace([np.inf, -np.inf], np.nan)
        svc_summary['refusal_rate'] = (svc_summary['patients_refused'] / svc_summary['patients_request']).replace([np.inf, -np.inf], np.nan)

        st.subheader('Service summary table')
        st.dataframe(svc_summary.sort_values('patients_request', ascending=False))

        st.subheader('Requests vs Admitted vs Refused (per service)')
        fig_s = px.bar(svc_summary.sort_values('patients_request', ascending=False), x='service',
                       y=['patients_request', 'patients_admitted', 'patients_refused'], barmode='group')
        st.plotly_chart(fig_s, use_container_width=True)

        # highlight top congested services
        svc_summary['overflow'] = svc_summary['patients_request'] - svc_summary['available_beds']
        top_congested = svc_summary.sort_values('overflow', ascending=False).head(5)
        st.subheader('Top 5 congested services (requests - beds)')
        st.dataframe(top_congested[['service', 'patients_request', 'available_beds', 'overflow', 'refusal_rate']])
    else:
        st.info('No service column found in services_weekly.csv')

    st.markdown('---')
    st.write('- Color thresholds for refusal rate: < 15% = green, 15–30% = amber, > 30% = red.')
    st.write('- Developed by Nur Hidayah Abdul Rahman.')

app()