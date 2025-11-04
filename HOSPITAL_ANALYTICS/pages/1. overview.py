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

    # parse arrival_date / discharge_date if present
    if 'arrival_date' in patients.columns:
        patients['arrival_date'] = pd.to_datetime(patients['arrival_date'], errors='coerce')
        # derive week number (ISO week number) as integer
        patients['arrival_week'] = patients['arrival_date'].dt.isocalendar().week.astype('Int64')
    else:
        patients['arrival_week'] = pd.NA

    # ensure services.week is numeric (week numbers like 1,2,3...)
    if 'week' in services.columns:
        services['week'] = pd.to_numeric(services['week'], errors='coerce').astype('Int64')
    if 'week' in schedules.columns:
        schedules['week'] = pd.to_numeric(schedules['week'], errors='coerce').astype('Int64')

    return staff, patients, services, schedules

def compute_kpis_from_services(services_df, patients_df, week_mask=None):
    df = services_df.copy()
    if week_mask is not None:
        df = df[week_mask]

    total_beds = int(df['available_beds'].sum()) if 'available_beds' in df.columns else 0
    total_requested = int(df['patients_request'].sum()) if 'patients_request' in df.columns else 0
    total_admitted = int(df['patients_admitted'].sum()) if 'patients_admitted' in df.columns else 0
    total_refused = int(df['patients_refused'].sum()) if 'patients_refused' in df.columns else 0

    admission_rate = (total_admitted / total_requested) if total_requested else np.nan
    refusal_rate = (total_refused / total_requested) if total_requested else np.nan

    return {
        'total_beds': total_beds,
        'total_requested': total_requested,
        'total_admitted': total_admitted,
        'total_refused': total_refused,
        'admission_rate': admission_rate,
        'refusal_rate': refusal_rate
    }

def compute_workload_by_role(services_df, schedules_df, selected_weeks):
    """Compute workload per service and role.
    workload = patients_admitted (aggregated by service & week range)
    / count of staff present (by service & role over same weeks)
    Returns a dataframe suitable for table and bar chart.
    """
    # aggregate admitted per service across chosen weeks
    s = services_df.copy()
    s = s[s['week'].isin(selected_weeks)] if 'week' in s.columns else s
    admitted_by_service = s.groupby('service', dropna=False)['patients_admitted'].sum().reset_index()
    admitted_by_service = admitted_by_service.rename(columns={'patients_admitted': 'patients_admitted_sum'})

    # count staff present per service & role across selected weeks
    sched = schedules_df.copy()
    sched = sched[sched['week'].isin(selected_weeks)] if 'week' in sched.columns else sched
    # consider present as truthy (1/True)
    if 'present' in sched.columns:
        sched_present = sched[sched['present'].astype(bool)]
    else:
        sched_present = sched

    staff_count = sched_present.groupby(['service', 'role']).agg(staff_present_count=('staff_id', 'nunique')).reset_index()

    # merge admitted with staff counts (service-level cross join with roles)
    merged = staff_count.merge(admitted_by_service, on='service', how='left')
    merged['patients_admitted_sum'] = merged['patients_admitted_sum'].fillna(0)

    # compute workload; avoid division by zero
    merged['workload_per_staff'] = merged.apply(
        lambda row: row['patients_admitted_sum'] / row['staff_present_count'] if row['staff_present_count'] > 0 else np.nan,
        axis=1
    )

    return merged

def app():
    st.set_page_config(page_title="Hospital Overview", layout="wide")
    st.title("Hospital Overview")

    staff, patients, services, schedules = load_all()

    st.session_state["data"] = {
        "staff": staff,
        "patients": patients,
        "services": services,
        "schedules": schedules
    }

    week_sources = []
    if 'week' in services.columns:
        week_sources.append(services['week'].dropna().unique().tolist())
    if 'week' in schedules.columns:
        week_sources.append(schedules['week'].dropna().unique().tolist())
    if 'arrival_week' in patients.columns:
        week_sources.append(patients['arrival_week'].dropna().unique().tolist())

    all_weeks = sorted(list({int(w) for group in week_sources for w in group if pd.notna(w)})) if week_sources else [1]
    if not all_weeks:
        st.error('No week data available in datasets.')
        return
    week_min, week_max = min(all_weeks), max(all_weeks)

    st.sidebar.header('Filters')
    week_range = st.sidebar.slider('Select Week Range', min_value=int(week_min), max_value=int(week_max), value=(int(week_min), int(week_max)))
    selected_weeks = list(range(week_range[0], week_range[1] + 1))

    missing_weeks = [w for w in selected_weeks if w not in (services['week'].dropna().unique().tolist() if 'week' in services.columns else [])]
    if missing_weeks:
        st.warning(f"Warning: No service data available for weeks: {missing_weeks}")

    week_mask = services['week'].isin(selected_weeks) if 'week' in services.columns else None
    kpis = compute_kpis_from_services(services, patients, week_mask=week_mask)

    if 'arrival_week' in patients.columns:
        patients_in_range = patients[patients['arrival_week'].isin(selected_weeks)]
    else:
        patients_in_range = patients
    total_admissions = len(patients_in_range)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Admissions (patient rows)', f"{total_admissions:,}")
    c2.metric('Total Beds (sum available)', f"{kpis['total_beds']:,}")
    c3.metric('Admission Rate', f"{kpis['admission_rate']:.2%}" if pd.notna(kpis['admission_rate']) else 'N/A')
    c4.metric('Refusal Rate', f"{kpis['refusal_rate']:.2%}" if pd.notna(kpis['refusal_rate']) else 'N/A')

    st.divider()

    st.header('Weekly Trends of Patient Admissions')
    if 'week' in services.columns:
        df_ts = services[services['week'].isin(selected_weeks)].copy()
        if df_ts.empty:
            st.info('No service data available for the selected weeks to display trends.')
        else:
            df_ts_agg = df_ts.groupby('week').agg({
                'patients_admitted': 'sum',
                'patients_request': 'sum',
                'patients_refused': 'sum'
            }).reset_index().sort_values('week')

            fig = px.line(df_ts_agg, x='week', y=['patients_admitted', 'patients_request', 'patients_refused'],
                          labels={'value': 'Number of Patients', 'week': 'Week', 'variable': 'Metric'},
                          title='Weekly Patient Admissions, Requests, and Refusals')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No week data available in services dataset to display trends.')
    st.divider()

    st.header('Workload by Service')
    if 'service' in services.columns:
        svc = services[services['week'].isin(selected_weeks)].copy() if 'week' in services.columns else services.copy()
        svc_summary = svc.groupby('service').agg({
            'available_beds': 'sum',
            'patients_request': 'sum',
            'patients_admitted': 'sum',
            'patients_refused': 'sum'
        }).reset_index()

        svc_summary['admission_rate']=(svc_summary['patients_admitted'] / svc_summary['patients_request']).replace([np.inf, -np.inf], np.nan)
        svc_summary['refusal_rate']=(svc_summary['patients_refused'] / svc_summary['patients_request']).replace([np.inf, -np.inf], np.nan)

        st.subheader('Service Summary Table')
        st.dataframe(svc_summary.sort_values('patients_request', ascending=False))

        st.subheader('Service admitted vs requests vs refused')
        fig2 = px.bar (svc_summary.sort_values('patients_request', ascending=False),
                       x='service', y=['patients_admitted', 'patients_request', 'patients_refused'],barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info('No service data available in services dataset to display service summary.')
    
    st.divider()

    st.header('Workload by Role')
    workload_df = compute_workload_by_role(services, schedules, selected_weeks)

    if workload_df.empty:
        st.info('No workload data available for the selected weeks to display workload by role.')
    else:
        st.subheader('Workload Table')
        st.dataframe(workload_df.sort_values(['service', 'role']))

        st.subheader('Workload Visualisation')
        fig_w = px.bar(workload_df, x='service', y='workload_per_staff', color='role', barmode='group',
                      labels={'workload_per_staff': 'Patients per Staff (admitted)', 'service': 'Service', 'role': 'Role'},
                      title='Workload per Staff by Role and Service')
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")
    st.write("Notes:")
    st.write('- Total Admissions counts patient rows by arrival_date (filtered by weeks).')
    st.write('- Workload uses unique staff_id present per service & role across selected weeks.')
    st.write('- Missing weeks warning appears if services_weekly.csv has no rows for some selected weeks.')

app()
