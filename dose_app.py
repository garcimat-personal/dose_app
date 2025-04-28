import os
import json
from datetime import datetime, timedelta

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
st.title("Dose Decay & Steady-State Build-Up")

# load or initialize doses
if "doses" not in st.session_state:
    st.session_state.doses = load_doses()

# Base datetime for clock labels (today at 8:30 AM)
base = datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)

# ---- Controls ----
with st.expander("Controls", expanded=True):
    dose_time = st.number_input(
        "Dose time (h)", min_value=0.0, step=0.1, value=0.0
    )

    dose_choice = st.selectbox(
        "Dose type",
        ["Initial (40 mg)", "Booster (8 mg)", "Custom"]
    )
    if dose_choice == "Initial (40 mg)":
        dose_amt = 40.0
    elif dose_choice == "Booster (8 mg)":
        dose_amt = 8.0
    else:
        dose_amt = st.number_input(
            "Custom amount (mg)", min_value=0.0, step=1.0, value=10.0
        )

    apply_meth = st.checkbox("Apply L-Methionine to next dose?")
    meth_amt   = st.number_input(
        "L-Methionine (mg)", min_value=0.0, step=1.0, value=5.0
    )

    # Four columns: Add, Undo, Download, Clear
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Add Dose"):
            st.session_state.doses.append({
                "time": dose_time,
                "amount": dose_amt,
                "meth_value": meth_amt if apply_meth else 0.0
            })
            save_doses(st.session_state.doses)
            st.session_state.apply_meth = False
    with col2:
        if st.button("Undo Last Dose"):
            if st.session_state.doses:
                st.session_state.doses.pop()
                save_doses(st.session_state.doses)
    with col3:
        data_str = json.dumps(st.session_state.doses, indent=2)
        st.download_button(
            "Download History",
            data=data_str,
            file_name="doses.json",
            mime="application/json"
        )
    with col4:
        if st.button("Clear All Doses"):
            st.session_state.doses = []
            save_doses(st.session_state.doses)

    # Upload prior history (below the four buttons)
    uploaded = st.file_uploader("Or upload a doses.json to restore", type="json")
    if uploaded is not None:
        try:
            new_doses = json.load(uploaded)
            if isinstance(new_doses, list):
                st.session_state.doses = new_doses
                save_doses(new_doses)
                st.success("History restored from upload.")
            else:
                st.error("Invalid format: expected a JSON array.")
        except Exception:
            st.error("Failed to parse uploaded JSON.")

# ---- Plotting ----
doses = st.session_state.doses
if not doses:
    st.info("No doses yet. Add one via the controls above.")
    st.stop()

# build numeric time axis
t0 = min(d["time"] for d in doses)
t1 = max(d["time"] for d in doses) + 4*HALF_LIFE_HOURS
t  = np.arange(t0, t1, TIME_STEP)

# plotting
fig, ax = plt.subplots(figsize=(10, 5))
total = np.zeros_like(t)

for d in doses:
    dt, amt, mval = d["time"], d["amount"], d["meth_value"]

    # subtract remnant if methionine was flagged
    if mval > 0:
        neg = np.zeros_like(t)
        mask = t >= dt
        neg[mask] = mval * np.exp(-DECAY_CONSTANT * (t[mask] - dt))
        total = np.maximum(total - neg, 0.0)

    # add the full dose decay curve
    curve = concentration_curve(dt, amt, t)
    total += curve

    # compute 12-hour clock label
    clock_label = (base + timedelta(hours=dt)).strftime('%-I:%M %p')
    legend_label = f"{amt:.0f} mg @ {dt:.1f} h ({clock_label})"

    ax.plot(t, curve, '--', label=legend_label)

# final clip & plot total
total = np.maximum(total, 0.0)
ax.plot(t, total, '-', lw=2, color='black', label="Total")

# annotate peaks & troughs
sorted_times = sorted(d["time"] for d in doses)
peaks, troughs = find_peaks_and_troughs(t, total, sorted_times)
for x, y in peaks:
    ax.plot(x, y, 'ro')
    ax.text(x, y, f'{y:.1f} mg', ha='center', va='bottom', fontsize='x-small')
for x, y in troughs:
    ax.plot(x, y, 'bx')
    ax.text(x, y, f'{y:.1f} mg', ha='center', va='top', fontsize='x-small')

# format x-axis with 12-hour clock labels
def hour_to_label(x, pos):
    dt = base + timedelta(hours=x)
    return dt.strftime('%-I:%M %p')

ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
ax.xaxis.set_major_formatter(FuncFormatter(hour_to_label))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

ax.set_xlabel("Clock Time")
ax.set_ylabel("Amount (mg)")
ax.set_title("Dose Decay & Steady-State Build-Up")
ax.legend(loc='upper right', fontsize='small')

st.pyplot(fig)
