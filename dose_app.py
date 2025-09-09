import os
import json
from datetime import datetime, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# --- Pharmacokinetic config ---
HALF_LIFE_HOURS = 15.0
DECAY_CONSTANT  = np.log(2) / HALF_LIFE_HOURS
TIME_STEP       = 0.1  # hours

# --- Persistence config ---
DATA_FILE = "doses.json"

def load_doses():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_doses(doses):
    with open(DATA_FILE, "w") as f:
        json.dump(doses, f)

def concentration_curve(t0, amt, t):
    conc = np.zeros_like(t)
    mask = t >= t0
    conc[mask] = amt * np.exp(-DECAY_CONSTANT * (t[mask] - t0))
    return conc

def find_peaks_and_troughs(t, total, dose_times):
    peaks, troughs = [], []
    for i, dt in enumerate(dose_times):
        pc = np.interp(dt, t, total)
        peaks.append((dt, pc))
        next_dt = dose_times[i+1] if i+1 < len(dose_times) else t[-1]
        mask = (t > dt) & (t < next_dt)
        if np.any(mask):
            idx = np.argmin(total[mask])
            troughs.append((t[mask][idx], total[mask][idx]))
    return peaks, troughs

# --- Streamlit setup ---
st.set_page_config(layout="wide")
st.title("Interactive Dose Decay & Steady-State Build-Up")

# load or init
if "doses" not in st.session_state:
    st.session_state.doses = load_doses()
if "deleted_stack" not in st.session_state:
    st.session_state.deleted_stack = []

# base datetime for t=0: Monday April 21 at 8:30
year = datetime.now().year
base = datetime(year, 4, 21, 8, 30)

# Mobile toggle
mobile = st.checkbox("Mobile view")

# ---- Controls ----
with st.expander("Controls", expanded=not mobile):
    # initialize dose_time
    if "dose_time" not in st.session_state:
        st.session_state.dose_time = max((d["Time"] for d in st.session_state.doses), default=0.0)

    st.session_state.dose_time = st.number_input(
        "Dose time (h)",
        value=st.session_state.dose_time,
        min_value=0.0, step=0.1
    )
    dose_time = st.session_state.dose_time

    if mobile:
        if st.button("Next Booster (+5h)"):
            st.session_state.dose_time += 5.0
        if st.button("Next Initial (+19h)"):
            st.session_state.dose_time += 19.0
        if st.button("Undo Delete"):
            if st.session_state.deleted_stack:
                restored = st.session_state.deleted_stack.pop()
                st.session_state.doses.extend(restored)
                st.session_state.doses.sort(key=lambda d: d["Time"])
                save_doses(st.session_state.doses)
    else:
        c1, c2, _, c5 = st.columns([1,1,6,1])
        with c1:
            if st.button("Next Booster (+5h)"):
                st.session_state.dose_time += 5.0
        with c2:
            if st.button("Next Initial (+19h)"):
                st.session_state.dose_time += 19.0
        with c5:
            if st.button("Undo Delete"):
                if st.session_state.deleted_stack:
                    restored = st.session_state.deleted_stack.pop()
                    st.session_state.doses.extend(restored)
                    st.session_state.doses.sort(key=lambda d: d["Time"])
                    save_doses(st.session_state.doses)

    # dose amount selection
    dose_choice = st.selectbox("Dose type", ["Initial (40 mg)", "Booster (8 mg)", "Custom"])
    if dose_choice == "Initial (40 mg)":
        dose_amt = 40.0
    elif dose_choice == "Booster (8 mg)":
        dose_amt = 8.0
    else:
        dose_amt = st.number_input("Custom amount (mg)", min_value=0.0, step=1.0, value=10.0)

    apply_meth = st.checkbox("Apply L-Methionine to next dose?")
    meth_amt   = st.number_input("L-Methionine (mg)", min_value=0.0, step=1.0, value=5.0)

    if mobile:
        if st.button("Add Dose"):
            st.session_state.doses.append({
                "Time": dose_time,
                "Amount": dose_amt,
                "L-Methionine Value": meth_amt if apply_meth else 0.0
            })
            save_doses(st.session_state.doses)
        if st.button("Undo Last Dose"):
            if st.session_state.doses:
                st.session_state.doses.pop()
                save_doses(st.session_state.doses)
        st.download_button(
            "Download History",
            json.dumps(st.session_state.doses, indent=2),
            "doses.json", "application/json"
        )
        if st.button("Clear All Doses"):
            st.session_state.deleted_stack.clear()
            st.session_state.doses.clear()
            save_doses(st.session_state.doses)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Add Dose"):
                st.session_state.doses.append({
                    "Time": dose_time,
                    "Amount": dose_amt,
                    "L-Methionine Value": meth_amt if apply_meth else 0.0
                })
                save_doses(st.session_state.doses)
        with c2:
            if st.button("Undo Last Dose"):
                if st.session_state.doses:
                    st.session_state.doses.pop()
                    save_doses(st.session_state.doses)
        with c3:
            st.download_button(
                "Download History",
                json.dumps(st.session_state.doses, indent=2),
                "doses.json", "application/json"
            )
        with c4:
            if st.button("Clear All Doses"):
                st.session_state.deleted_stack.clear()
                st.session_state.doses.clear()
                save_doses(st.session_state.doses)

    uploaded = st.file_uploader("Upload doses.json to restore", type="json")
    if uploaded is not None:
        try:
            new = json.load(uploaded)
            if isinstance(new, list):
                st.session_state.doses = new
                st.session_state.deleted_stack.clear()
                save_doses(new)
                st.success("History restored.")
            else:
                st.error("Invalid JSON structure.")
        except:
            st.error("Error parsing JSON.")

    show_legend = st.checkbox("Show Legend", value=True)
    show_edit   = st.checkbox("Edit Dose History")

# ---- Editable dose table ----
if show_edit:
    df = pd.DataFrame(st.session_state.doses)[["Time","Amount","L-Methionine Value"]].astype(float)
    df.insert(1, "Date & Time",
              df["Time"].apply(lambda x:
                  (base + timedelta(hours=x)).strftime("%a %m/%d %I:%M %p")
              ))
    df.insert(0, "Delete?", False)

    column_config = {
        "Time": {"hidden": True},
        "Delete?": {"type": "boolean"},
        "Date & Time": {"type": "text"},
        "Amount": {"type": "numeric"},
        "L-Methionine Value": {"type": "numeric"}
    }

    edited = st.data_editor(
        df,
        column_config=column_config,
        num_rows="dynamic",
        key="dose_editor",
        use_container_width=True,
        height=(300 if mobile else 600)
    )

    new_list, deleted = [], []
    for _, row in edited.iterrows():
        if row["Delete?"]:
            deleted.append({
                "Time": float(row["Time"]),
                "Amount": float(row["Amount"]),
                "L-Methionine Value": float(row["L-Methionine Value"])
            })
        else:
            parts = row["Date & Time"].split()
            mon, day = map(int, parts[1].split("/"))
            hour, minute = map(int, parts[2].split(":"))
            ampm = parts[3].upper()
            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
            dt_new = base.replace(month=mon, day=day, hour=hour, minute=minute)
            offset = max((dt_new - base).total_seconds()/3600.0, 0.0)
            new_list.append({
                "Time": offset,
                "Amount": float(row["Amount"]),
                "L-Methionine Value": float(row["L-Methionine Value"])
            })

    if deleted:
        st.session_state.deleted_stack.append(deleted)

    if new_list != st.session_state.doses:
        st.session_state.doses = new_list
        save_doses(new_list)

# ---- Plotting ----
doses = st.session_state.doses
if not doses:
    st.info("No doses yet.")
    st.stop()

t0 = min(d["Time"] for d in doses)
t1 = max(d["Time"] for d in doses) + 4 * HALF_LIFE_HOURS
t = np.arange(t0, t1, TIME_STEP)
x_times = [base + timedelta(hours=hh) for hh in t]

# ---- compute zoom ranges centered on last dose ----
# find the time (in hours) of the last dose
last_dose_h = max(d["Time"] for d in doses)
# convert to a datetime
center_time = base + timedelta(hours=last_dose_h)

# total span of data
full_start, full_end = x_times[0], x_times[-1]
span = full_end - full_start

# desktop: 2× zoom → span/2 window (so half of full span each side)
half_window = span / 4

# mobile: 3× zoom → span/3 window (so one-sixth of full span each side)
third_window = span / 6

# pick window based on mobile toggle
if mobile:
    zoom_range = [center_time - third_window, center_time + third_window]
else:
    zoom_range = [center_time - half_window,  center_time + half_window]

total = np.zeros_like(t)
fig = go.Figure()

for d in doses:
    dt, amt, mval = d["Time"], d["Amount"], d["L-Methionine Value"]
    if mval > 0:
        total = np.maximum(total - concentration_curve(dt, mval, t), 0.0)
    curve = concentration_curve(dt, amt, t)
    total += curve
    lbl = (base + timedelta(hours=dt)).strftime("%a (%m/%d) %I:%M %p")

    fig.add_trace(go.Scatter(
        x=x_times, y=curve, mode='lines', line=dict(dash='dash'),
        name=f"{amt:.0f} mg @ {dt:.1f}h ({lbl})",
        hovertemplate='%{y:.1f} mg at %{x|%a %m/%d %I:%M %p}<extra></extra>'
    ))

fig.add_trace(go.Scatter(
    x=x_times, y=total, mode='lines', line=dict(width=3, color='black'),
    name='Total',
    hovertemplate='%{y:.1f} mg at %{x|%a %m/%d %I:%M %p}'
))

peaks, troughs = find_peaks_and_troughs(t, total, sorted(d["Time"] for d in doses))
for x_val, y_val in peaks:
    fig.add_trace(go.Scatter(
        x=[base + timedelta(hours=x_val)], y=[y_val],
        mode='markers+text', marker=dict(color='red', size=8),
        text=[f"{y_val:.1f} mg"], textposition='top center',
        showlegend=False
    ))
for x_val, y_val in troughs:
    fig.add_trace(go.Scatter(
        x=[base + timedelta(hours=x_val)], y=[y_val],
        mode='markers+text', marker=dict(symbol='x', color='blue', size=8),
        text=[f"{y_val:.1f} mg"], textposition='bottom center',
        showlegend=False
    ))

fig.add_hline(y=60, line_dash="dot", line_color="red",
              annotation_text="Max dose", annotation_position="top right",
              annotation_font_size=14, annotation_font_color="black")
fig.add_hline(y=32.5, line_dash="dot", line_color="green",
              annotation_text="Min dose", annotation_position="top right",
              annotation_font_size=14, annotation_font_color="black")

# Configure legend color and position
if mobile:
    legend_cfg = dict(
        orientation='h',
        yanchor='top',
        y=-1.0,            # below the rangeslider
        xanchor='center',
        x=0.5,
        font=dict(color='darkgrey')
    )
else:
    legend_cfg = dict(font=dict(color='darkgrey'))

fig.update_layout(
    title=dict(
        text=('    Dose Decay & Steady-State Build-Up' if mobile else 'Interactive Dose Decay & Steady-State Build-Up'),
        x=0.435, xanchor='center',
        y=0.95, yanchor='top',
        font=dict(size=(19 if mobile else 30), color='black'),
        pad=dict(b=5)
    ),
    font=dict(color='black'),       # all non-legend text black
    showlegend=show_legend,
    legend=legend_cfg,
    paper_bgcolor='white',
    plot_bgcolor='rgba(230,230,230,1)',
    height= (1700 if mobile else 800),
    margin=dict(l=40, r=20, t=(140 if mobile else 100), b=80),
    dragmode='pan',
    xaxis=dict(
        type='date',
        range=zoom_range,
        tickformat='%a (%m/%d)<br>%I:%M %p',
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor='black',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='black',
        linecolor='black',
        tickfont=dict(color='black'),
        title_font=dict(color='black')
    ),
    yaxis=dict(
        title='Amount (mg)',
        showgrid=True,
        gridcolor='black',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='black',
        linecolor='black',
        tickfont=dict(color='black'),
        title_font=dict(color='black')
    )
)

st.plotly_chart(fig, use_container_width=True)
