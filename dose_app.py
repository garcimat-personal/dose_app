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
TIME_STEP       = 0.1   # hours

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

# load or init persisted doses
if "doses" not in st.session_state:
    st.session_state.doses = load_doses()

# fix t=0 to Monday, April 21, at 8:30 AM of this year
year = datetime.now().year
base = datetime(year, 4, 21, 8, 30)

# ---- Controls ----
with st.expander("Controls", expanded=True):
    # initialize default time to last dose if present
    if "dose_time" not in st.session_state:
        st.session_state.dose_time = max((d["Time"] for d in st.session_state.doses), default=0.0)

    # Dose time input
    st.session_state.dose_time = st.number_input(
        "Dose time (h)",
        value=st.session_state.dose_time,
        min_value=0.0,
        step=0.1
    )
    col1, col2, _ = st.columns([1,1,8])
    with col1:
        if st.button("Next Booster (+5h)"):
            st.session_state.dose_time += 5.0
    with col2:
        if st.button("Next Initial (+19h)"):
            st.session_state.dose_time += 19.0
    dose_time = st.session_state.dose_time

    # Dose amount
    dose_choice = st.selectbox("Dose type", ["Initial (40 mg)", "Booster (8 mg)", "Custom"])
    if dose_choice == "Initial (40 mg)":
        dose_amt = 40.0
    elif dose_choice == "Booster (8 mg)":
        dose_amt = 8.0
    else:
        dose_amt = st.number_input("Custom amount (mg)", min_value=0.0, step=1.0, value=10.0)

    # L-Methionine
    apply_meth = st.checkbox("Apply L-Methionine to next dose?")
    meth_amt   = st.number_input("L-Methionine (mg)", min_value=0.0, step=1.0, value=5.0)

    # Action buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Add Dose"):
            st.session_state.doses.append({
                "Time": dose_time,
                "Amount": dose_amt,
                "L-Methionine Value": meth_amt if apply_meth else 0.0
            })
            save_doses(st.session_state.doses)
            st.session_state.dose_time = dose_time
    with c2:
        if st.button("Undo Last Dose"):
            if st.session_state.doses:
                st.session_state.doses.pop()
                save_doses(st.session_state.doses)
    with c3:
        st.download_button(
            "Download History",
            json.dumps(st.session_state.doses, indent=2),
            "doses.json",
            "application/json"
        )
    with c4:
        if st.button("Clear All Doses"):
            st.session_state.doses = []
            save_doses(st.session_state.doses)

    # Upload
    uploaded = st.file_uploader("Upload doses.json to restore", type="json")
    if uploaded:
        try:
            new = json.load(uploaded)
            if isinstance(new, list):
                st.session_state.doses = new
                save_doses(new)
                st.success("History restored.")
            else:
                st.error("Invalid JSON structure.")
        except:
            st.error("Error parsing JSON.")

    # Toggles
    show_legend = st.checkbox("Show Legend", value=True)
    show_edit   = st.checkbox("Edit Dose History")

# ---- Editable dose table (conditional) ----
if show_edit:
    df = pd.DataFrame(st.session_state.doses)[["Time","Amount","L-Methionine Value"]].astype(float)
    df.insert(
        1,
        "Date & Time",
        df["Time"]
          .apply(lambda x: (base + timedelta(hours=x)).strftime("%a %m/%d %-I:%M %p"))
    )
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        key="dose_editor",
        use_container_width=True
    )
    if not edited[["Time","Amount","L-Methionine Value"]].equals(df[["Time","Amount","L-Methionine Value"]]):
        new_list = []
        for _, row in edited.iterrows():
            new_list.append({
                "Time": float(row["Time"]),
                "Amount": float(row["Amount"]),
                "L-Methionine Value": float(row["L-Methionine Value"])
            })
        st.session_state.doses = new_list
        save_doses(new_list)

# ---- Build data for plotting ----
doses = st.session_state.doses
if not doses:
    st.info("No doses yet. Use the controls above.")
    st.stop()

t0 = min(d["Time"] for d in doses)
t1 = max(d["Time"] for d in doses) + 4 * HALF_LIFE_HOURS
t = np.arange(t0, t1, TIME_STEP)
x_times = [base + timedelta(hours=hh) for hh in t]

total = np.zeros_like(t)
fig = go.Figure()

# plot each dose curve
for d in doses:
    dt, amt, mval = d["Time"], d["Amount"], d["L-Methionine Value"]
    if mval > 0:
        neg = np.zeros_like(t)
        mask = t >= dt
        neg[mask] = mval * np.exp(-DECAY_CONSTANT * (t[mask] - dt))
        total = np.maximum(total - neg, 0.0)
    curve = concentration_curve(dt, amt, t)
    total += curve
    clock_lbl = (base + timedelta(hours=dt)).strftime('%a (%m/%d) %-I:%M %p')
    legend_lbl = f"{amt:.0f} mg @ {dt:.1f}h ({clock_lbl})"
    fig.add_trace(go.Scatter(
        x=x_times, y=curve, mode='lines',
        name=legend_lbl, line=dict(dash='dash'),
        hovertemplate='%{y:.1f} mg at %{x|%a %m/%d %I:%M %p}<extra></extra>'
    ))

# plot total
total = np.maximum(total, 0.0)
fig.add_trace(go.Scatter(
    x=x_times, y=total, mode='lines',
    name='Total', line=dict(width=3, color='black'),
    hovertemplate='%{y:.1f} mg at %{x|%a %m/%d %I:%M %p}<extra></extra>'
))

# peaks & troughs
peaks, troughs = find_peaks_and_troughs(t, total, sorted(d["Time"] for d in doses))
fig.add_trace(go.Scatter(
    x=[base + timedelta(hours=p[0]) for p in peaks],
    y=[p[1] for p in peaks], mode='markers+text', name='Peaks',
    marker=dict(color='red', size=8),
    text=[f'{p[1]:.1f} mg' for p in peaks], textposition='top center'
))
fig.add_trace(go.Scatter(
    x=[base + timedelta(hours=tr[0]) for tr in troughs],
    y=[tr[1] for tr in troughs], mode='markers+text', name='Troughs',
    marker=dict(color='blue', symbol='x', size=8),
    text=[f'{tr[1]:.1f} mg' for tr in troughs], textposition='bottom center'
))

# add threshold lines
fig.add_hline(y=60, line_dash="dot", line_color="red",
              annotation_text="Max dose", annotation_position="top right",
              annotation_font_size=14, annotation_font_color="black")
fig.add_hline(y=32, line_dash="dot", line_color="green",
              annotation_text="Min dose", annotation_position="top right",
              annotation_font_size=14, annotation_font_color="black")

# layout
fig.update_layout(
    title={
        'text': 'Interactive Dose Decay & Steady-State Build-Up',
        'x': 0.5, 'xanchor': 'center',
        'y': 0.95, 'yanchor': 'top',
        'font': {'size': 30, 'color': 'black'},
        'pad': {'b': 5}
    },
    font=dict(color='black'),
    showlegend=show_legend,
    legend=dict(font=dict(color='rgba(0,0,0,0.7)')),
    paper_bgcolor='white',
    plot_bgcolor='rgba(230,230,230,1)',
    height=800,
    margin=dict(l=40, r=20, t=100, b=40),
    dragmode='pan',
    xaxis=dict(
        title='Clock Time',
        title_font=dict(color='black'),
        tickfont=dict(color='black'),
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        type='date',
        tickformat='%a (%m/%d)<br>%I:%M %p',
        rangeslider=dict(visible=True),
        range=[x_times[0], x_times[0] + timedelta(days=((t1 - t0)//24) + 1)]
    ),
    yaxis=dict(
        title='Amount (mg)',
        title_font=dict(color='black'),
        tickfont=dict(color='black'),
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True
    )
)

st.plotly_chart(fig, use_container_width=True)
