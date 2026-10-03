"""NEXUS - Autonomous Energy Engineer
Reservoir + Production engineering workbench with ML surveillance and a Groq-powered AI copilot.
Run: streamlit run app.py
"""
import os, json
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor

st.set_page_config(page_title="NEXUS | Autonomous Energy Engineer", page_icon="⚡", layout="wide")
st.title("⚡ NEXUS - Autonomous Energy Engineer")
st.caption("Decline analysis • Nodal • PVT/MBAL • Well test • ML surveillance • AI copilot")
ctx = st.session_state.setdefault("ctx", {})

# ---------------- core engineering functions ----------------
def arps(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)

def eur(qi, di, b, qec):
    b = min(max(b, 1e-3), 0.999)
    if qec >= qi: return 0.0, 0.0
    t_ec = ((qi / qec) ** b - 1) / (b * di)
    np_ = qi ** b / ((1 - b) * di) * (qi ** (1 - b) - qec ** (1 - b))
    return t_ec, np_

def demo_data(n=150, seed=7):
    r = np.random.default_rng(seed); t = np.arange(n)
    q = arps(t, 1500, 0.03, 0.8) * (1 + r.normal(0, .03, n))
    i = r.choice(n, 6, replace=False); q[i] *= r.uniform(.35, .6, 6)
    return pd.DataFrame({"day": t, "rate": q, "whp": 900 - .8 * t + r.normal(0, 8, n),
                         "wc": np.clip(.05 + .003 * t + r.normal(0, .01, n), 0, 1)})

def vogel_qmax(qt, pwft, pr):
    x = pwft / pr; return qt / (1 - .2 * x - .8 * x ** 2)

def vogel_pwf(q, qmax, pr):
    return .125 * pr * (-1 + np.sqrt(np.clip(81 - 80 * q / qmax, 0, None)))

def vlp(q, pwh, tvd, d_in, api, wc, glr):
    so = 141.5 / (131.5 + api); rho = ((1 - wc) * so + wc * 1.03) * 1000
    rho *= max(.5, 1 - 0.00025 * glr)                       # gas-lift lightening (simplified)
    hyd = rho * 9.81 * tvd * .3048 / 6894.76
    D = d_in * .0254; Q = q * 1.8401e-6; v = Q / (np.pi * D ** 2 / 4)
    fric = 0.02 * (tvd * .3048 / D) * rho * v ** 2 / 2 / 6894.76
    return pwh + hyd + fric

def z_papay(p, T_F, sg):
    ppc = 756.8 - 131 * sg - 3.6 * sg ** 2; tpc = 169.2 + 349.5 * sg - 74 * sg ** 2
    pr, tr = p / ppc, (T_F + 460) / tpc
    return 1 - 3.52 * pr * np.exp(-2.26 * tr) + .274 * pr ** 2 * np.exp(-1.878 * tr)

def groq_chat(messages, key, model="llama-3.3-70b-versatile", temp=.2):
    from groq import Groq
    r = Groq(api_key=key).chat.completions.create(model=model, messages=messages, temperature=temp)
    return r.choices[0].message.content

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Setup")
    key = st.text_input("Groq API key", type="password",
                        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")) if hasattr(st, "secrets") else "")
    model = st.selectbox("LLM", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    up = st.file_uploader("Production CSV (day, rate, whp, wc)", type="csv")
    df = pd.read_csv(up) if up else demo_data()
    st.caption("Using uploaded data" if up else "Using synthetic demo well")

t1, t2, t3, t4, t5, t6 = st.tabs(["📉 Decline & EUR", "🛢️ Nodal (IPR/VLP)", "🧪 PVT & MBAL",
                                  "📈 Well Test", "🤖 ML Surveillance", "💬 Nexus Copilot"])

# ---------------- 1. DCA ----------------
with t1:
    c1, c2 = st.columns([1, 3])
    qec = c1.number_input("Economic limit (STB/d)", value=50.0)
    clean = c1.checkbox("Remove ML-flagged outliers before fit", True)
    d = df.copy()
    if clean:
        m = IsolationForest(contamination=.05, random_state=0).fit(d[["rate"]])
        d = d[m.predict(d[["rate"]]) == 1]
    try:
        p, _ = curve_fit(arps, d.day, d.rate, p0=[d.rate.max(), .02, .5],
                         bounds=([1, 1e-5, .01], [1e6, 1, .99]))
        qi, di, b = p; t_ec, np_ = eur(qi, di, b, qec)
        c1.metric("qi (STB/d)", f"{qi:,.0f}"); c1.metric("Di (1/d)", f"{di:.4f}")
        c1.metric("b-factor", f"{b:.2f}"); c1.metric("Remaining life (yr)", f"{t_ec/365:.1f}")
        c1.metric("EUR (MMSTB)", f"{np_/1e6:.2f}")
        tf = np.arange(0, max(t_ec, df.day.max()) * 1.0)
        f = go.Figure()
        f.add_scatter(x=df.day, y=df.rate, mode="markers", name="Actual")
        f.add_scatter(x=tf, y=arps(tf, *p), name="Arps fit")
        f.add_hline(y=qec, line_dash="dash", annotation_text="Economic limit")
        f.update_layout(yaxis_type="log", xaxis_title="Day", yaxis_title="Rate (STB/d)")
        c2.plotly_chart(f, use_container_width=True)
        ctx["DCA"] = dict(qi=round(qi), Di=round(di, 4), b=round(b, 2), EUR_STB=round(np_), life_yr=round(t_ec / 365, 1))
    except Exception as e:
        st.error(f"Fit failed: {e}")

# ---------------- 2. Nodal ----------------
with t2:
    c = st.columns(4)
    pr = c[0].number_input("Reservoir P (psi)", value=3500.0); qt = c[0].number_input("Test rate (STB/d)", value=800.0)
    pt = c[1].number_input("Test Pwf (psi)", value=2800.0); pwh = c[1].number_input("WHP (psi)", value=300.0)
    tvd = c[2].number_input("TVD (ft)", value=8000.0); tub = c[2].number_input("Tubing ID (in)", value=2.992)
    api = c[3].number_input("API", value=35.0); wc = c[3].slider("Water cut", 0.0, 1.0, .2)
    glr = st.slider("Gas-lift injection GLR (scf/bbl)", 0, 2000, 0, 100)
    qmax = vogel_qmax(qt, pt, pr); q = np.linspace(1, qmax * .999, 200)
    ipr = vogel_pwf(q, qmax, pr)
    fig = go.Figure(); fig.add_scatter(x=q, y=ipr, name="IPR (Vogel)")
    res = {}
    for g in sorted({0, glr}):
        v = vlp(q, pwh, tvd, tub, api, wc, g); fig.add_scatter(x=q, y=v, name=f"VLP GLR={g}")
        i = np.argmin(abs(ipr - v)); res[g] = (q[i], ipr[i])
    fig.update_layout(xaxis_title="Rate (STB/d)", yaxis_title="Pwf (psi)"); st.plotly_chart(fig, use_container_width=True)
    q_op, p_op = res[glr]; st.success(f"Operating point: **{q_op:,.0f} STB/d @ {p_op:,.0f} psi** | AOF ≈ {qmax:,.0f} STB/d")
    if glr: st.info(f"Gas-lift gain: {q_op - res[0][0]:+,.0f} STB/d vs natural flow")
    ctx["Nodal"] = dict(q_op=round(q_op), pwf_op=round(p_op), AOF=round(qmax), GLR=glr, WC=wc, tubing_in=tub)

# ---------------- 3. PVT & MBAL ----------------
with t3:
    a, b_ = st.columns(2)
    with a:
        st.subheader("Oil PVT (Standing)")
        Rs = st.number_input("Rs (scf/STB)", value=600.0); gg = st.number_input("Gas SG", value=.75)
        T = st.number_input("Temp (°F)", value=200.0); ap = st.number_input("Oil API", value=35.0, key="a2")
        go_ = 141.5 / (131.5 + ap)
        pb = 18.2 * ((Rs / gg) ** .83 * 10 ** (.00091 * T - .0125 * ap) - 1.4)
        bo = .9759 + .00012 * (Rs * (gg / go_) ** .5 + 1.25 * T) ** 1.2
        st.metric("Bubble point (psi)", f"{pb:,.0f}"); st.metric("Bo (rb/STB)", f"{bo:.3f}")
        pp = np.linspace(100, 6000, 60); fz = go.Figure(go.Scatter(x=pp, y=z_papay(pp, T, gg)))
        fz.update_layout(title="Gas Z-factor (Papay)", xaxis_title="P (psi)", yaxis_title="Z", height=280)
        st.plotly_chart(fz, use_container_width=True)
    with b_:
        st.subheader("Gas p/z Material Balance")
        mb = st.text_area("P(psi),Gp(MMscf)", "4000,0\n3700,1500\n3400,3100\n3100,4600\n2800,6200", height=130)
        try:
            M = np.array([[float(x) for x in l.split(",")] for l in mb.strip().splitlines()])
            pz = M[:, 0] / z_papay(M[:, 0], T, gg); s, i0 = np.polyfit(M[:, 1], pz, 1); ogip = -i0 / s
            fm = go.Figure(); fm.add_scatter(x=M[:, 1], y=pz, mode="markers", name="p/z")
            xs = np.linspace(0, ogip, 20); fm.add_scatter(x=xs, y=s * xs + i0, name="Fit")
            fm.update_layout(xaxis_title="Gp (MMscf)", yaxis_title="p/z", height=330); st.plotly_chart(fm, use_container_width=True)
            st.metric("OGIP (Bscf)", f"{ogip/1e3:,.1f}"); ctx["MBAL"] = dict(OGIP_Bscf=round(ogip / 1e3, 1), Pb=round(pb), Bo=round(bo, 3))
        except Exception as e:
            st.warning(f"Check input format: {e}")

# ---------------- 4. Well test ----------------
with t4:
    c = st.columns(6)
    q_ = c[0].number_input("q (STB/d)", value=500.0); B = c[1].number_input("Bo", value=1.2)
    mu = c[2].number_input("μ (cp)", value=1.0); h = c[3].number_input("h (ft)", value=30.0)
    phi = c[4].number_input("φ", value=.2); ct = c[5].number_input("ct (1/psi)", value=1e-5, format="%.1e")
    tp = st.number_input("Producing time tp (hr)", value=240.0); pwf0 = st.number_input("Pwf at shut-in (psi)", value=2800.0)
    rw = .354; rng = np.random.default_rng(1); dt = np.logspace(-1, 2, 30)
    mt = 162.6 * q_ * B * mu / (50 * h)                         # demo truth: k = 50 md
    x = np.log10((tp + dt) / dt); pws = 3500 - mt * x + rng.normal(0, 2, len(dt))
    cut = st.slider("Straight-line start (hr)", 1.0, 50.0, 10.0); k = dt >= cut
    sl, ic = np.polyfit(x[k], pws[k], 1); m = -sl
    perm = 162.6 * q_ * B * mu / (m * h); p1 = ic + sl * np.log10(tp + 1)
    skin = 1.151 * ((p1 - pwf0) / m - np.log10(perm / (phi * mu * ct * rw ** 2)) + 3.23)
    fh = go.Figure(); fh.add_scatter(x=x, y=pws, mode="markers", name="Pws")
    fh.add_scatter(x=x, y=ic + sl * x, name="Horner line"); fh.update_layout(xaxis_title="log[(tp+Δt)/Δt]", xaxis_autorange="reversed", yaxis_title="Pws")
    st.plotly_chart(fh, use_container_width=True)
    c = st.columns(4); c[0].metric("k (md)", f"{perm:.1f}"); c[1].metric("Skin", f"{skin:.2f}")
    c[2].metric("P* (psi)", f"{ic:,.0f}"); c[3].metric("ΔP skin (psi)", f"{.87*m*skin:,.0f}")
    ctx["WellTest"] = dict(k_md=round(perm, 1), skin=round(skin, 2), Pstar=round(ic))

# ---------------- 5. ML surveillance ----------------
with t5:
    feats = [c for c in ["rate", "whp", "wc"] if c in df]
    X = df[feats].copy(); X["d_rate"] = X.rate.pct_change().fillna(0)
    iso = IsolationForest(contamination=.05, random_state=0).fit(X); df["anom"] = iso.predict(X) == -1
    fa = go.Figure(); fa.add_scatter(x=df.day, y=df.rate, name="Rate")
    fa.add_scatter(x=df.day[df.anom], y=df.rate[df.anom], mode="markers", marker=dict(color="red", size=10), name="Anomaly")
    st.plotly_chart(fa, use_container_width=True)
    st.write(f"🚨 **{int(df.anom.sum())} anomalies** detected (trips, chokes, slugging, sensor drift candidates)")
    st.subheader("Gradient-boosting 30-day forecast")
    lr = np.log(df.rate.clip(lower=1)).values; r = np.diff(lr); L = 5
    Xs = np.array([r[i:i + L] for i in range(len(r) - L)]); ys = r[L:]
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2).fit(Xs, ys)
    hist = list(r[-L:]); lv = lr[-1]; fc = []
    for _ in range(30):
        n_ = gb.predict([hist[-L:]])[0]; lv += n_; hist.append(n_); fc.append(np.exp(lv))
    ff = go.Figure(); ff.add_scatter(x=df.day, y=df.rate, name="History")
    ff.add_scatter(x=df.day.max() + 1 + np.arange(30), y=fc, name="ML forecast"); st.plotly_chart(ff, use_container_width=True)
    ctx["ML"] = dict(anomalies=int(df.anom.sum()), rate_30d_ML=round(fc[-1]), last_rate=round(df.rate.iloc[-1]))

# ---------------- 6. Copilot ----------------
SYS = ("You are NEXUS, a senior reservoir & production engineer. Use the live analysis context given. "
       "Be quantitative, state assumptions, give ranked actions (workover, artificial lift, choke, stimulation, "
       "surveillance) with expected impact and risk. Never invent numbers not in context; flag uncertainty.")
with t6:
    st.json(ctx, expanded=False)
    m1, m2 = st.tabs(["Chat", "Daily report parser (NLP)"])
    with m1:
        hist = st.session_state.setdefault("chat", [])
        for h_ in hist: st.chat_message(h_["role"]).write(h_["content"])
        if st.button("🧠 Auto-diagnose well"):
            prompt = "Diagnose this well from the analysis context and give a prioritized action plan."
        else:
            prompt = st.chat_input("Ask Nexus (e.g. should we install gas lift?)")
        if prompt:
            if not key: st.error("Add your Groq API key in the sidebar.")
            else:
                hist.append({"role": "user", "content": prompt}); st.chat_message("user").write(prompt)
                msgs = [{"role": "system", "content": SYS + "\nCONTEXT: " + json.dumps(ctx)}] + hist
                with st.spinner("Thinking..."):
                    ans = groq_chat(msgs, key, model)
                hist.append({"role": "assistant", "content": ans}); st.chat_message("assistant").write(ans)
    with m2:
        rep = st.text_area("Paste field/morning report text", height=180)
        if st.button("Extract") and rep and key:
            out = groq_chat([{"role": "system", "content": "Return ONLY JSON: {well, issues[], rates{}, pressures{}, "
                              "risks[], recommended_actions[], severity(1-5)}"}, {"role": "user", "content": rep}], key, model, 0)
            try: st.json(json.loads(out[out.find("{"):out.rfind("}") + 1]))
            except Exception: st.write(out)
