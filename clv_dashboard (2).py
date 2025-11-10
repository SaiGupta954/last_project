import os
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# =========================
# THEME (colorful + LIGHT)
# =========================
st.set_page_config(page_title="🌈 Retail Insights Dashboard", layout="wide")

# Light theme + vivid palette
pio.templates.default = "plotly"  # <- light
px.defaults.template = "plotly"
px.defaults.color_discrete_sequence = [
    "#7C3AED", "#06B6D4", "#22C55E", "#F59E0B", "#EF4444", "#10B981", "#3B82F6", "#EC4899"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSETS_DIR = Path(__file__).parent / "assets"

# =========================
# COLORFUL / ANIMATED CSS
# =========================
st.markdown("""
<style>
:root{
  --accent:#7C3AED; --accent2:#06B6D4; --accent3:#22C55E;
  --card:#ffffffee; --border:#e6e9f2; --text:#0f172a;
}
@keyframes floaty {
  0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}
}
html, body, .stApp {
  background: linear-gradient(120deg,#e0f2fe,#f3e8ff,#dcfce7,#fff7ed);
  background-size: 300% 300%;
  animation: floaty 20s ease infinite;
}
.block-container{padding-top:1rem;}
.login-card, .hero-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(15,23,42,.08);
}
.login-card{padding:20px;}
.hero-card{position:relative; overflow:hidden;}
.hero-header{
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 50%, var(--accent3) 100%);
  color: #fff; font-weight:800; font-size:18px; padding:12px 16px;
}
.hero-body{padding:12px;}
.hero-body .stImage img{border-radius:12px; filter:saturate(1.2) contrast(1.05) brightness(1.02);}
.badge-live{
  position:absolute; top:12px; right:12px; color:#fff; font-weight:800; font-size:12px;
  padding:6px 10px; border-radius:999px; background:linear-gradient(90deg,#F97316,#EF4444);
  box-shadow:0 6px 18px rgba(239,68,68,.35);
}
.stTextInput input, .stSelectbox div[data-baseweb="select"]>div{
  background:#fff !important; color:var(--text) !important; border:1px solid var(--border) !important;
}
.stFileUploader{
  background:#fff; border:1px dashed #d8dbe6; border-radius:12px; padding:6px;
}
.stButton>button{
  background: linear-gradient(90deg, var(--accent), var(--accent2)); color:#fff; font-weight:800;
  border:none; border-radius:10px; padding:.6rem 1rem; box-shadow:0 10px 24px rgba(99,102,241,.25);
}
.stButton>button:hover{opacity:.95; transform: translateY(-1px);}
h1, h2, h3{color:#0f172a;}
.section-bar{
  height:6px; border-radius:999px;
  background:linear-gradient(90deg,#7C3AED,#06B6D4,#22C55E); margin:6px 0 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# AUTH (simple demo)
# =========================
if "user_db" not in st.session_state: st.session_state.user_db = {}
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "username" not in st.session_state: st.session_state.username = None

def _hash(p:str)->str: return hashlib.sha256(p.encode()).hexdigest()
def _ok(p,h)->bool: return _hash(p)==h

def login_signup_inline():
    tab_login, tab_signup = st.tabs(["🔐 Login", "🆕 Signup"])
    with tab_signup:
        u = st.text_input("Username", key="su_u")
        e = st.text_input("Email", key="su_e")
        p = st.text_input("Password", type="password", key="su_p")
        if st.button("Create Account", key="su_btn"):
            if u and p:
                if u in st.session_state.user_db: st.error("Username already exists.")
                else:
                    st.session_state.user_db[u] = {"email": e, "password": _hash(p)}
                    st.success("Signup successful. Please login.")
            else: st.error("Username and password required.")
    with tab_login:
        u = st.text_input("Username", key="li_u")
        p = st.text_input("Password", type="password", key="li_p")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                user = st.session_state.user_db.get(u)
                if user and _ok(p, user["password"]):
                    st.session_state.authenticated, st.session_state.username = True, u
                    st.rerun()
                else: st.error("Invalid credentials")
        with c2:
            if st.button("Login as Demo", use_container_width=True):
                demo = "demo"; st.session_state.user_db.setdefault(demo, {"email":"demo@x.com","password":_hash("demo")})
                st.session_state.authenticated, st.session_state.username = True, demo
                st.rerun()

# =========================
# HERO (login left + colorful hero right)
# =========================
left, right = st.columns([1,2], vertical_alignment="center")
with left:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.subheader("✨ Login to Continue")
    st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
    login_signup_inline()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-header">Retail Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-live">LIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-body">', unsafe_allow_html=True)
    hero_path = ASSETS_DIR / "hero_dashboard.gif"
    if hero_path.exists():
        st.image(str(hero_path), use_container_width=True)
    else:
        # colorful fallback image (safe)
        st.image("https://images.unsplash.com/photo-1551281044-8d8e81a0eb70?q=80&w=1600&auto=format&fit=crop", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# DATA LOADERS (local parquet/csv)
# =========================
def read_any(parquet_path, csv_path):
    if os.path.exists(parquet_path): return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):    return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner="Loading data…")
def load_data():
    tx = read_any(os.path.join(DATA_DIR,"Transactions.parquet"), os.path.join(DATA_DIR,"Transactions.csv"))
    hh = read_any(os.path.join(DATA_DIR,"Households.parquet"),   os.path.join(DATA_DIR,"Households.csv"))
    pr = read_any(os.path.join(DATA_DIR,"Products.parquet"),     os.path.join(DATA_DIR,"Products.csv"))
    for df in (tx,hh,pr):
        if not df.empty: df.columns = df.columns.str.strip()
    return tx,hh,pr

def _norm(s): return str(s).strip().lower().replace(" ","").replace("_","").replace("-","")
def _rename(df,target,cands):
    if df.empty: return df
    m={_norm(c):c for c in df.columns}
    for c in cands:
        if _norm(c) in m and target not in df.columns:
            df.rename(columns={m[_norm(c)]:target}, inplace=True); break
    return df

def normalize_cols(tx,hh,pr):
    tx=_rename(tx,"hshd_num",["HSHD_NUM","hshd","household","householdid"])
    tx=_rename(tx,"product_num",["PRODUCT_NUM","product","productid","prod_num"])
    hh=_rename(hh,"hshd_num",["HSHD_NUM","hshd","household","householdid"])
    pr=_rename(pr,"product_num",["PRODUCT_NUM","product","productid","prod_num"])
    return tx,hh,pr

def merge_full(tx,hh,pr):
    full=tx.copy()
    if not full.empty and not hh.empty and "hshd_num" in full.columns and "hshd_num" in hh.columns:
        full=full.merge(hh,on="hshd_num",how="left")
    if not full.empty and not pr.empty and "product_num" in full.columns and "product_num" in pr.columns:
        full=full.merge(pr,on="product_num",how="left")
    return full

# =========================
# DASHBOARD (colorful + animated)
# =========================
if st.session_state.authenticated:
    st.title("📊 Retail Customer Analytics — Color Edition")
    st.success(f"Welcome, {st.session_state.username}! Enjoy the colorful vibes 🌈")

    # Uploads (optional)
    tx, hh, pr = load_data()
    c1,c2,c3 = st.columns(3)
    with c1: up_tx = st.file_uploader("Upload Transactions (CSV)", type="csv")
    with c2: up_hh = st.file_uploader("Upload Households (CSV)", type="csv")
    with c3: up_pr = st.file_uploader("Upload Products (CSV)", type="csv")
    if up_tx is not None: tx = pd.read_csv(up_tx)
    if up_hh is not None: hh = pd.read_csv(up_hh)
    if up_pr is not None: pr = pd.read_csv(up_pr)

    tx, hh, pr = normalize_cols(tx,hh,pr)
    if tx.empty or hh.empty or pr.empty:
        st.warning("Please make sure **Transactions, Households, Products** are available in `/data` or upload them.")
        st.stop()

    # Date column
    if {"YEAR","WEEK_NUM"}.issubset(tx.columns):
        tx["date"]=pd.to_datetime(tx["YEAR"].astype(str)+tx["WEEK_NUM"].astype(str)+'0',format="%Y%U%w",errors="coerce")
    elif "PURCHASE_" in tx.columns:
        tx["date"]=pd.to_datetime(tx["PURCHASE_"],errors="coerce")
    else:
        tx["date"]=pd.NaT

    full = merge_full(tx,hh,pr)

    # ---- 1) Animated Engagement Over Time ----
    st.header("📈 Engagement Over Time (Animated)")
    if "SPEND" in tx.columns and "date" in tx.columns:
        t = tx.dropna(subset=["date"]).copy()
        if not t.empty:
            t["week"] = t["date"].dt.to_period("W").astype(str)
            wk = (t.groupby(["week"], as_index=False)["SPEND"].sum()
                    .sort_values("week"))
            # Cumulative for smoother animation
            wk["CUM_SPEND"] = wk["SPEND"].cumsum()
            fig = px.line(
                wk, x="week", y="CUM_SPEND", markers=True,
                title="Cumulative Spend by Week",
                labels={"week":"Week","CUM_SPEND":"Cumulative Spend"},
                animation_frame="week",  # frames play through weeks
                range_y=[0, max(1.05*wk["CUM_SPEND"].max(), 1)],
            )
            fig.update_traces(line=dict(width=4))
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid dates found.")
    else:
        st.info("Need SPEND + date fields.")

    # ---- 2) Bar-Race: Top Products over Time (Animated) ----
    st.header("🏁 Top Products Over Time (Bar-Race)")
    if {"BASKET_NUM","product_num","SPEND"}.issubset(tx.columns):
        tt = tx.dropna(subset=["date"]).copy()
        tt["week"] = tt["date"].dt.to_period("W").astype(str)
        prod_week = tt.groupby(["week","product_num"], as_index=False)["SPEND"].sum()
        # Take top N per week to keep frames small
        topN = prod_week.sort_values(["week","SPEND"], ascending=[True,False]).groupby("week").head(8)
        # Attach labels if available
        label_col = "COMMODITY" if "COMMODITY" in pr.columns else "product_num"
        topN = topN.merge(pr[["product_num", label_col]].drop_duplicates(), on="product_num", how="left")
        topN["label"] = topN[label_col].astype(str)
        fig = px.bar(
            topN, x="SPEND", y="label", orientation="h",
            animation_frame="week", range_x=[0, max(1.1*topN["SPEND"].max(), 1)],
            title="Top 8 Products by Week", labels={"label":"Product / Category","SPEND":"Spend"}
        )
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need BASKET_NUM, product_num, SPEND in transactions.")

    # ---- 3) Seasonal Patterns with Department Frames ----
    st.header("🌤️ Seasonal Patterns (Per Department)")
    if "date" in tx.columns and "SPEND" in tx.columns:
        s = tx.dropna(subset=["date"]).copy()
        if not s.empty:
            s["Month"] = s["date"].dt.month_name()
            if "DEPARTMENT" in pr.columns and "product_num" in s.columns:
                s = s.merge(pr[["product_num","DEPARTMENT"]], on="product_num", how="left")
                fig = px.area(
                    s.groupby(["DEPARTMENT","Month"], as_index=False)["SPEND"].sum(),
                    x="Month", y="SPEND", color="Month",
                    animation_frame="DEPARTMENT",
                    category_orders={"Month": ["January","February","March","April","May","June","July","August","September","October","November","December"]},
                    title="Monthly Spend by Department"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # fallback: no department available
                seasonal = s.groupby("Month", as_index=False)["SPEND"].sum()
                fig = px.area(seasonal, x="Month", y="SPEND", title="Monthly Spend")
                st.plotly_chart(fig, use_container_width=True)

    # ---- 4) Age Group Pie by Month (Animated) ----
    if {"AGE_RANGE","SPEND"}.issubset(full.columns) and "date" in tx.columns:
        st.header("🧩 Age-Group Share by Month (Animated)")
        a = full.copy()
        # derive month from tx and merge if needed
        if "date" not in a.columns and "hshd_num" in a.columns and "hshd_num" in tx.columns:
            tmonth = tx[["hshd_num","date","SPEND"]].dropna(subset=["date"]).copy()
            tmonth["Month"]=tmonth["date"].dt.month_name()
            a = tmonth.merge(a.drop(columns=["SPEND"], errors="ignore"), on="hshd_num", how="left")
        else:
            if "date" in a.columns:
                a["Month"]=pd.to_datetime(a["date"], errors="coerce").dt.month_name()
        if "Month" in a.columns:
            pie = a.groupby(["Month","AGE_RANGE"], as_index=False)["SPEND"].sum()
            fig = px.pie(pie, names="AGE_RANGE", values="SPEND",
                         animation_frame="Month", hole=.35, title="Age Distribution per Month")
            st.plotly_chart(fig, use_container_width=True)

    # ---- CLV & Top Customers (colorful static) ----
    st.header("💰 Customer Lifetime Value")
    if {"hshd_num","SPEND"}.issubset(tx.columns):
        clv = tx.groupby("hshd_num", as_index=False)["SPEND"].sum().sort_values("SPEND", ascending=False).head(10)
        fig = px.bar(clv, x="hshd_num", y="SPEND", text_auto=".2s", title="Top 10 Customers")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Quick Search ----
    st.header("🔎 Search by Household")
    query = st.text_input("Enter HSHD_NUM")
    if query:
        try:
            q = int(query)
            if "hshd_num" in tx.columns:
                out = tx[tx["hshd_num"]==q].copy()
                if not out.empty:
                    st.success(f"Found {len(out)} rows for {q}")
                    sort_cols=[c for c in ["BASKET_NUM","date","product_num"] if c in out.columns]
                    st.dataframe(out.sort_values(by=sort_cols) if sort_cols else out)
                else:
                    st.warning("No matching rows.")
            else: st.error("Missing hshd_num in transactions.")
        except ValueError:
            st.error("Please enter a numeric HSHD_NUM.")

    # ---- ML Tabs (unchanged visuals but colorful) ----
    tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧠 Basket Spend Regression"])

    with tab1:
        st.subheader("RFM-based Churn Model")
        if {"date","SPEND","BASKET_NUM","hshd_num"}.issubset(tx.columns):
            t = tx.dropna(subset=["date"]).copy()
            if not t.empty:
                now = t["date"].max()
                rfm = t.groupby("hshd_num").agg(
                    recency=("date", lambda x:(now-x.max()).days),
                    frequency=("BASKET_NUM","nunique"),
                    monetary=("SPEND","sum")
                )
                rfm["churn"]=(rfm["recency"]>84).astype(int)
                X=rfm[["recency","frequency","monetary"]]; y=rfm["churn"]
                Xtr,Xte,ytr,yte=train_test_split(X,y, stratify=y, test_size=.2, random_state=42)
                clf=RandomForestClassifier(n_estimators=120, random_state=42).fit(Xtr,ytr)
                yp=clf.predict(Xte)
                st.text(classification_report(yte, yp))
                cm = pd.DataFrame(confusion_matrix(yte, yp),
                                  columns=["Pred Not Churn","Pred Churn"],
                                  index=["Act Not Churn","Act Churn"])
                st.dataframe(cm)
                imp = pd.Series(clf.feature_importances_, index=["Recency","Frequency","Monetary"]).reset_index()
                imp.columns=["Feature","Importance"]
                fig = px.bar(imp, x="Feature", y="Importance", title="Feature Importances")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Predict Basket Spend")
        if {"BASKET_NUM","SPEND","product_num"}.issubset(tx.columns):
            bm = tx.merge(pr, on="product_num", how="left")
            cols = [c for c in ["DEPARTMENT","COMMODITY","BRAND_TY","NATURAL_ORGANIC_FLAG"] if c in bm.columns]
            if cols:
                feats = pd.get_dummies(bm[["BASKET_NUM"]+cols], drop_first=True)
                X = feats.groupby("BASKET_NUM").sum()
                y = bm.groupby("BASKET_NUM")["SPEND"].sum()
                Xtr,Xte,ytr,yte=train_test_split(X,y, test_size=.2, random_state=42)
                rf=RandomForestRegressor(n_estimators=150, random_state=42).fit(Xtr,ytr)
                yp=rf.predict(Xte)
                st.write(f"**R²:** {r2_score(yte, yp):.3f}   **MSE:** {mean_squared_error(yte, yp):.2f}")
                comp = pd.DataFrame({"Actual":yte.values, "Predicted":yp})
                fig = px.line(comp.reset_index(drop=True), y=["Actual","Predicted"], title="Predicted vs Actual")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(comp)
