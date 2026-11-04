import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import Model as M

st.set_page_config(page_title="NBA Shot Make Probability", page_icon="🏀", layout="wide")


# ---------- cached loading: runs once, reused on every interaction ----------
@st.cache_data
def get_data() -> pd.DataFrame:
    return M.load_data("training.csv.gz")


@st.cache_resource
def get_model() -> dict:
    return M.train(get_data())


def fg_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(col, observed=True)["outcome"]
        .agg(fg="mean", shots="size")
        .reset_index()
    )


FG_AXIS = alt.Axis(format="%", title="FG%")
TIP = [alt.Tooltip("fg:Q", format=".1%", title="FG%"), alt.Tooltip("shots:Q", format=",")]

df = get_data()
out = get_model()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
st.sidebar.caption("Apply to the Overview, Shot Chart and Defense tabs.")
types = st.sidebar.multiselect("Shot type", sorted(df["shottype"].unique()),
                               default=sorted(df["shottype"].unique()))
states = st.sidebar.multiselect("Game state", sorted(df["gamestate"].unique()),
                                default=sorted(df["gamestate"].unique()))
pressure = st.sidebar.radio("Defense", ["All", "Contested", "Open"], horizontal=True)

f = df[df["shottype"].isin(types) & df["gamestate"].isin(states)]
if pressure != "All":
    f = f[f["contested"] == (pressure == "Contested")]

st.title("🏀 NBA Shot Make Probability")
st.caption(f"Logistic regression on {len(df):,} tracked NBA shots · by Terrance Dessaure")

if f.empty:
    st.warning("No shots match these filters.")
    st.stop()

tab_over, tab_chart, tab_def, tab_model, tab_sim = st.tabs(
    ["Overview", "Shot Chart", "Defense", "Model", "Shot Simulator"]
)

# ---------- Overview ----------
with tab_over:
    c = st.columns(4)
    c[0].metric("Shots", f"{len(f):,}")
    c[1].metric("FG%", f"{f['outcome'].mean():.1%}")
    c[2].metric("Avg distance", f"{f['distance'].mean():.1f} ft")
    c[3].metric("Model AUC", f"{out['metrics']['auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("FG% by shot type")
        st.altair_chart(
            alt.Chart(fg_by(f, "shottype")).mark_bar().encode(
                x=alt.X("fg:Q", axis=FG_AXIS),
                y=alt.Y("shottype:N", sort="-x", title=None),
                tooltip=["shottype"] + TIP,
            ),
        )
    with right:
        st.subheader("FG% by distance")
        d = f[f["distance"] <= 35].assign(ft=lambda x: x["distance"].round())
        d = fg_by(d, "ft").query("shots >= 200")
        st.altair_chart(
            alt.Chart(d).mark_line(point=True).encode(
                x=alt.X("ft:Q", title="Distance (ft)"),
                y=alt.Y("fg:Q", axis=FG_AXIS),
                tooltip=["ft"] + TIP,
            ),
            
        )
    st.caption("Efficiency drops fast outside the paint, then flattens through the mid-range "
               "— the reason teams trade long twos for threes.")

# ---------- Shot chart ----------
def court_lines() -> pd.DataFrame:
    """Half-court outline, paint, 3PT line and rim as ordered polylines."""
    r, hoop_y = 23.75, 5.25
    t = np.linspace(-np.arcsin(22 / r), np.arcsin(22 / r), 60)
    arc_y = hoop_y + r * np.cos(t)
    three = [(-22, 0)] + list(zip(r * np.sin(t), arc_y)) + [(22, 0)]
    rim = [(0.75 * np.sin(a), hoop_y + 0.75 * np.cos(a)) for a in np.linspace(0, 2 * np.pi, 20)]
    parts = {
        "outline": [(-25, 0), (-25, 47), (25, 47), (25, 0), (-25, 0)],
        "paint": [(-8, 0), (-8, 19), (8, 19), (8, 0)],
        "three": three,
        "rim": rim,
    }
    return pd.DataFrame(
        [(name, i, x, y) for name, pts in parts.items() for i, (x, y) in enumerate(pts)],
        columns=["part", "i", "x", "y"],
    )


with tab_chart:
    st.subheader("Where shots go in")
    min_shots = st.slider("Hide zones with fewer than N shots", 10, 500, 50, step=10)
    size = 2  # ft per zone
    z = f[f["court_y"].between(0, 47) & f["court_x"].between(-25, 25)].assign(
        bx=lambda x: np.floor(x["court_x"] / size) * size,
        by=lambda x: np.floor(x["court_y"] / size) * size,
    )
    z = z.groupby(["bx", "by"])["outcome"].agg(fg="mean", shots="size").reset_index()
    z = z[z["shots"] >= min_shots].assign(bx2=lambda x: x.bx + size, by2=lambda x: x.by + size)

    xs = alt.Scale(domain=[-25, 25])
    ys = alt.Scale(domain=[0, 47])
    zones = alt.Chart(z).mark_rect().encode(
        x=alt.X("bx:Q", scale=xs, axis=None), x2="bx2",
        y=alt.Y("by:Q", scale=ys, axis=None), y2="by2",
        color=alt.Color("fg:Q", title="FG%", legend=alt.Legend(format="%"),
                        scale=alt.Scale(scheme="redyellowgreen", domain=[0.25, 0.65], clamp=True)),
        tooltip=TIP,
    )
    lines = alt.Chart(court_lines()).mark_line(color="gray").encode(
        x=alt.X("x:Q", scale=xs), y=alt.Y("y:Q", scale=ys), detail="part", order="i"
    )
    st.altair_chart((zones + lines).properties(width=520, height=490))
    st.caption("Each square is a 2×2 ft zone. Both baskets are folded onto one half-court.")

# ---------- Defense ----------
with tab_def:
    st.subheader("How much does the closest defender matter?")
    d = f[f["closestdefdist"] <= 12].assign(
        ft=lambda x: x["closestdefdist"].round(),
        shot=lambda x: np.where(x["three"], "3PT", "2PT"),
    )
    d = d.groupby(["ft", "shot"])["outcome"].agg(fg="mean", shots="size").reset_index()
    st.altair_chart(
        alt.Chart(d[d["shots"] >= 200]).mark_line(point=True).encode(
            x=alt.X("ft:Q", title="Closest defender distance (ft)"),
            y=alt.Y("fg:Q", axis=FG_AXIS),
            color=alt.Color("shot:N", title=None),
            tooltip=["ft", "shot"] + TIP,
        ),
        
    )
    st.caption("Split by 2PT/3PT on purpose: close defenders are mostly near the rim, "
               "where shots go in more — mixing them hides the true effect of defense.")

    left, right = st.columns(2)
    with left:
        st.subheader("FG% by number of contesters")
        st.altair_chart(
            alt.Chart(fg_by(f, "num_contesters")).mark_bar().encode(
                x=alt.X("num_contesters:O", title="Contesters"),
                y=alt.Y("fg:Q", axis=FG_AXIS), tooltip=["num_contesters"] + TIP,
            ),
            
        )
    with right:
        st.subheader("FG% by defender closing speed")
        g = f.assign(closing=pd.cut(f["closing_speed"], [-np.inf, 0, 2, np.inf],
                                    labels=["Drifting away", "Moderate", "Fast close-out"]))
        st.altair_chart(
            alt.Chart(fg_by(g, "closing")).mark_bar().encode(
                x=alt.X("closing:N", sort=None, title=None),
                y=alt.Y("fg:Q", axis=FG_AXIS), tooltip=["closing"] + TIP,
            ),
            
        )
    st.caption("Closing speed = defender distance 4 frames before the shot minus distance at release.")

# ---------- Model ----------
with tab_model:
    m = out["metrics"]
    st.subheader("Validation performance")
    st.caption(f"Trained on {m['n_train']:,} shots, scored on {m['n_val']:,} held-out shots "
               "(stratified 80/20 split). Baseline = always predict the more common outcome.")
    c = st.columns(3)
    c[0].metric("ROC AUC", f"{m['auc']:.3f}", f"{m['auc'] - 0.5:+.3f} vs coin flip")
    c[1].metric("Accuracy", f"{m['accuracy']:.1%}",
                f"{m['accuracy'] - m['base_accuracy']:+.1%} vs baseline")
    c[2].metric("Log loss", f"{m['log_loss']:.3f}",
                f"{m['log_loss'] - m['base_log_loss']:+.3f} vs baseline", delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.subheader("What drives a make?")
        coefs = out["coefs"].assign(effect=lambda x: np.where(x.coefficient > 0, "Helps", "Hurts"))
        st.altair_chart(
            alt.Chart(coefs).mark_bar().encode(
                x=alt.X("coefficient:Q", title="Standardized coefficient (log-odds)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("effect:N", title=None,
                                scale=alt.Scale(domain=["Helps", "Hurts"], range=["#2a9d8f", "#e76f51"])),
                tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")],
            ),
            
        )
        st.caption("Features are standardized, so bar length compares effect per 1 std. dev. "
                   "Shot types are relative to jumpers; game states relative to half-court.")
    with right:
        st.subheader("Are the probabilities honest?")
        cal = out["calibration"].reset_index(drop=True)
        diag = pd.DataFrame({"pred": [0.2, 0.9], "actual": [0.2, 0.9]})
        st.altair_chart(
            alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="pred:Q", y="actual:Q")
            + alt.Chart(cal).mark_line(point=True).encode(
                x=alt.X("pred:Q", title="Predicted make %", axis=alt.Axis(format="%")),
                y=alt.Y("actual:Q", title="Actual make %", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("pred:Q", format=".1%"), alt.Tooltip("actual:Q", format=".1%")],
            ),
            
        )
        st.caption("Validation shots grouped into 10 buckets by predicted probability. "
                   "Points on the dashed line mean '40% predicted' really goes in ~40% of the time.")

# ---------- Simulator ----------
with tab_sim:
    st.subheader("Build a shot, get a make probability")
    a, b = st.columns(2)
    with a:
        shottype = st.selectbox("Shot type", sorted(df["shottype"].unique()),
                                index=sorted(df["shottype"].unique()).index("jumper"))
        gamestate = st.selectbox("Game state", sorted(df["gamestate"].unique()),
                                 index=sorted(df["gamestate"].unique()).index("HALFCOURT"))
        distance = st.slider("Shot distance (ft)", 0.0, 40.0, 18.0, 0.5)
        dribbles = st.slider("Dribbles before shot", 0, 20, 1)
        shotclock = st.slider("Shot clock (s)", 0.0, 24.0, 12.0, 0.5)
    with b:
        defdist = st.slider("Closest defender distance (ft)", 0.0, 20.0, 4.0, 0.5)
        closing = st.slider("Defender closing speed (ft over last 4 frames)", -5.0, 12.0, 2.0, 0.5)
        contesters = st.slider("Contesters", 0, 4, 1)
        speed = st.slider("Shooter speed", 0.0, 16.0, 3.0, 0.5)

    shot = dict(
        shottype=shottype, gamestate=gamestate, distance=distance,
        dribblesbefore=dribbles, shotclock=shotclock, closestdefdist=defdist,
        closestdefapproach=f"{{{defdist + closing},0,0,{defdist}}}",  # gives closing_speed
        num_contesters=contesters, contested=contesters > 0,  # contested == contesters > 0 in the data
        shooterspeed=speed, locationx=-M.HOOP_X, locationy=0.0, three=distance >= 23.75,
    )
    p = M.predict_one(out["model"], shot)
    pts = 3 if distance >= 23.75 else 2  # approx: corner threes start at 22 ft

    c = st.columns(3)
    c[0].metric("Make probability", f"{p:.1%}", f"{p - df['outcome'].mean():+.1%} vs league avg")
    c[1].metric("Expected points", f"{p * pts:.2f}", help="probability × shot value (≥23.75 ft = 3)")
    c[2].metric("Shot value", f"{pts} pts")
    st.caption("Try it: drag distance from 3 ft to 25 ft, or compare a catch-and-shoot (0 dribbles) "
               "with an open defender vs. a late-clock, heavily contested pull-up.")