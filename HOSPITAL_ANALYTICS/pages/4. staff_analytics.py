import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Staff Analytics", layout="wide")

st.title("Staff Analytics")

if "data" not in st.session_state or "staff" not in st.session_state["data"] or "schedules" not in st.session_state["data"]:
    st.error("⚠️ Data not loaded. Please go to the **Overview** page first.")
    st.stop()

staff = st.session_state["data"]["staff"]
schedules = st.session_state["data"]["schedules"]

if schedules.empty or staff.empty:
    st.warning("No staff or schedule data available.")
    st.stop()

if "staff_id" in schedules.columns and "staff_id" in staff.columns:
    schedules = schedules.merge(
        staff[["staff_id", "staff_name", "service"]],
        on="staff_id",
        how="left",
        suffixes=("", "_staff")
    )

    # Prefer staff info from the merged columns
    schedules["staff_name"] = schedules["staff_name_staff"].fillna(schedules.get("staff_name"))
    schedules["service"] = schedules["service_staff"].fillna(schedules.get("service"))

    # Drop the extra columns if they exist
    schedules = schedules.drop(columns=[c for c in schedules.columns if c.endswith("_staff") or c.endswith("_x") or c.endswith("_y")], errors="ignore")

st.sidebar.header('Filters')

# Try to get week range dynamically from schedules
if "week" in schedules.columns and not schedules["week"].dropna().empty:
    week_min = int(schedules["week"].min())
    week_max = int(schedules["week"].max())
else:
    week_min, week_max = 1, 52  # default fallback if no week data

# Sidebar slider for selecting week range
week_range = st.sidebar.slider(
    'Select Week Range',
    min_value=week_min,
    max_value=week_max,
    value=(week_min, week_max)
)
selected_weeks = list(range(week_range[0], week_range[1] + 1))

# Filter schedules by selected weeks
if "week" in schedules.columns:
    schedules = schedules[schedules["week"].isin(selected_weeks)]

# Optional: show how many rows remain after filtering
st.caption(f"Filtered data includes {len(schedules)} schedule records for weeks {week_range[0]}–{week_range[1]}.")

st.subheader("📊 Staff Attendance Summary")

if all(col in schedules.columns for col in ["staff_id", "staff_name", "present"]):
    attendance_summary = (
        schedules.groupby(["staff_id", "staff_name"])["present"]
        .agg(["sum", "count"])
        .reset_index()
    )
    attendance_summary["attendance_rate"] = attendance_summary["sum"] / attendance_summary["count"]

    st.dataframe(
        attendance_summary[["staff_name", "sum", "attendance_rate"]]
        .sort_values("attendance_rate", ascending=False)
        .rename(columns={"sum": "Days Present"})
    )
else:
    st.warning("Columns `staff_id`, `staff_name`, or `present` missing in schedules data.")

if all(col in schedules.columns for col in ["staff_name", "week", "present"]):
    st.subheader("🗓️ Attendance Heatmap by Week")

    pivot = schedules.pivot_table(
        index="staff_name", columns="week", values="present", aggfunc="sum", fill_value=0
    )

    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Week", y="Staff Name", color="Days Present"),
        aspect="auto",
        color_continuous_scale="Blues",
        title="Staff Weekly Attendance Heatmap",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("Not enough data to generate attendance heatmap.")


if "service" in schedules.columns:
    st.subheader("🏥 Average Attendance by Service")

    service_attendance = schedules.groupby("service")["present"].mean().reset_index()
    fig_service = px.bar(
        service_attendance,
        x="service",
        y="present",
        labels={"present": "Average Attendance Rate"},
        title="Average Attendance Rate by Service",
        color="present",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_service, use_container_width=True)
else:
    st.warning("`service` column missing from schedules data.")


if "staff_morale" in schedules.columns:
    st.subheader("💡 Correlation: Attendance vs Staff Morale")

    fig_corr = px.scatter(
        schedules,
        x="staff_morale",
        y="present",
        color="service" if "service" in schedules.columns else None,
        trendline="ols",
        title="Attendance vs Staff Morale",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('---')
st.write('Developed by Nur Hidayah Abdul Rahman.')