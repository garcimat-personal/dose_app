import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Pharmacokinetic config ---
HALF_LIFE_HOURS = 15.0
DECAY_CONSTANT  = np.log(2) / HALF_LIFE_HOURS
TIME_STEP       = 0.1  # hours

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

# --- Session state initialization ---
if "doses" not in st.session_state:
    st.session_state.doses = []  # each entry: {"time","amount","meth_value"}

st.set_page_config(layout="wide")
st.title("Dose Decay & Steady-State Build-Up")

# ---- Controls in main expander ----
with st.expander("Controls", expanded=True):
    dose_time = st.number_input("Dose time (h)", min_value=0.0, step=0.1, value=0.0)

    dose_choice = st.selectbox("Dose type",
        ["Initial (40 mg)", "Booster (8 mg)", "Custom"])
    if dose_choice == "Initial (40 mg)":
        dose_amt = 40.0
    elif dose_choice == "Booster (8 mg)":
        dose_amt = 8.0
    else:
        dose_amt = st.number_input("Custom amount (mg)", min_value=0.0, step=1.0, value=10.0)

    apply_meth = st.checkbox("Apply L-Methionine to next dose?")
    meth_amt   = st.number_input("L-Methionine (mg)", min_value=0.0, step=1.0, value=5.0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Dose"):
            # record dose; meth_amt only if flagged
            mval = meth_amt if apply_meth else 0.0
            st.session_state.doses.append({
                "time": dose_time,
                "amount": dose_amt,
                "meth_value": mval
            })
            st.session_state.apply_meth = False
    with col2:
        if st.button("Undo Last Dose"):
            if st.session_state.doses:
                st.session_state.doses.pop()

# ---- Plot or message ----
doses = st.session_state.doses

if not doses:
    st.info("No doses entered yet. Use the Controls above to add your first dose.")
else:
    # construct time axis safely now that we know doses is non-empty
    t0 = min(d["time"] for d in doses)
    t1 = max(d["time"] for d in doses) + 4*HALF_LIFE_HOURS
    t  = np.arange(t0, t1, TIME_STEP)

    total = np.zeros_like(t)
    fig, ax = plt.subplots(figsize=(10,5))

    # process each dose chronologically
    for d in doses:
        dt, amt, mval = d["time"], d["amount"], d["meth_value"]

        # subtract decaying negative impulse if mval > 0
        if mval > 0:
            neg = np.zeros_like(t)
            mask = t >= dt
            neg[mask] = mval * np.exp(-DECAY_CONSTANT*(t[mask] - dt))
            total = np.maximum(total - neg, 0.0)

        # add the full dose decay
        curve = concentration_curve(dt, amt, t)
        total += curve
        ax.plot(t, curve, '--', label=f"{amt:.0f} mg @ {dt:.1f} h")

    # final clip
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

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Amount (mg)")
    ax.set_title("Dose Decay & Steady-State Build-Up")
    ax.legend(loc='upper right', fontsize='small')
    st.pyplot(fig)
