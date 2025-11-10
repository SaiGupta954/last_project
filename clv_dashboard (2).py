import os
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix


# =========================
# Global Theme Setup
# =========================
st.set_page_config(page_title="🌈 Retail Insights Dashboard", layout="wide")
pio.templates.default = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#7C3AED", "#06B6D4", "#22C55E", "#F59E0B", "#EC4899", "#10B981", "#60A5FA"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSETS_DIR = Path(__file__).parent / "assets"

# =========================
# Custom Colorful Style
# =========================
st.markdown("""
<style>
:root{
  --accent:#7C3AED;
  --accent2:#06B6D4;
  --accent3:#22C55E;
  --bg1:#0b1026;
  --card:#0f1a42;
  --border: rgba(255,255,255,.10);
}

/* colorful background */
html, body, .stApp {
  background: radial-gradient(1200px 700px at 85% -10%, #0b5fff33, transparent 60%),
              radial-gradient(1000px 600px at -10% 80%, #22c55e22, transparent 60%),
              linear-gradient(160deg, #0b1026 0%, #0b1437 45%, #0a1a44 100%) !important;
}

.block-container { padding-top: 1.1rem; }

/* cards */
.hero-card, .login-card {
  background: linear-gradient(180deg, rgba(15,26,66,.92), rgba(15,20,55,.92));
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 18px 60px rgba(0,0,0,.35);
}

.login-card { padding: 22px 18px; }
.hero-card { position: relative; overflow: hidden; }

/* hero */
.hero-header {
  background: linear-gradient(90deg, var(--accent) 0%, #4F46E5 35%, var(--accent2) 100%);
  color: #fff; padding: 14px 16px; font-weight: 800; font-size: 18px;
}
.hero-body { padding: 12px; background: transparent; }
.hero-body .stImage img{
  width:100%; border-radius:14px; display:block;
  filter: brightness(1.3) saturate(1.25) contrast(1.06);
  box-shadow: 0 18px 50px rgba(0,0,0,.35);
}
.badge-live {
  position:absolute; top:14px; right:14px;
  background: linear-gradient(90deg, #fb7185, #ef4444);
  color:#fff; font-weight:800; font-size:12px;
  padding:6px 10px; border-radius:999px;
  box-shadow: 0 6px 18px rgba(239,68,68,.45);
}

/* tabs */
.stTabs [data-baseweb="tab"]{
  color:#cbd5e1; font-weight:700; border:none; background:transparent;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color:#fff;
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
  border-radius:999px;
}

/* inputs */
input, textarea, .stTextInput>div>div>input{
  background:#0b1430 !important; color:#e5e7eb !important;
  border:1px solid var(--border) !important;
}
.stSelectbox, .stTextInput, .stFileUploader{
  filter: drop-shadow(0 10px 24px rgba(0,0,0,.25));
}

/* buttons */
.stButton>button{
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
  border:none; color:#fff; font-weight:800;
  padding:.6rem 1rem; border-radius:12px;
  box-shadow: 0 12px 30px rgba(124,58,237,.35);
}
.stButton>button:hover{ transform: translateY(-1px); opacity:.95; }

/* headings */
h1, h2, h3 { color:#fff; }
</style>
""", unsafe_allow_html=True)


# =========================
# Auth Helpers
# =========================
if "user_db" not in st.session_state:
    st.session_state.user_db = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

def make_hashes(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password: str, hashed_text: str) -> bool:
    return make_hashes(password) == hashed_text

def login_signup_inline():
    tab_login, tab_signup = st.tabs(["🔐 Login", "🆕 Signup"])
    with tab_signup:
        new_user = st.text_input("Username", key="signup_user")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create Account", key="signup_btn"):
            if new_user and new_password:
                if new_user in st.session_state.user_db:
                    st.error("Username already exists.")
                else:
                    st.session_state.user_db[new_user] = {
                        "email": new_email,
                        "password": make_hashes(new_password),
                    }
                    st.success("Signup successful. Please login.")
    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", key="login_btn", use_container_width=True):
                user = st.session_state.user_db.get(username)
                if user and check_hashes(password, user["password"]):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with c2:
            if st.button("Login as Demo", key="demo_btn", use_container_width=True):
                demo_user = "demo_user"
                demo_pw_h = make_hashes("demo_pass")
                st.session_state.user_db.setdefault(demo_user, {"email": "demo@example.com", "password": demo_pw_h})
                st.session_state.authenticated = True
                st.session_state.username = demo_user
                st.rerun()


# =========================
# Hero Section
# =========================
left, right = st.columns([1, 2], vertical_alignment="center")

with left:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.subheader("💫 Login to Continue")
    st.markdown(
        "<div style='height:6px;border-radius:999px;background:linear-gradient(90deg,#7C3AED,#06B6D4,#22C55E);margin:6px 0 14px;'></div>",
        unsafe_allow_html=True
    )
    login_signup_inline()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    hero_path = ASSETS_DIR / "hero_dashboard.gif"
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-header">Retail Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-live">LIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-body">', unsafe_allow_html=True)
    if hero_path.exists():
        st.image(str(hero_path), use_container_width=True)
    else:
        st.info("Upload your animated GIF to assets/hero_dashboard.gif")
    st.markdown('</div></div>', unsafe_allow_html=True)


# =========================
# Dashboard Content
# =========================
if st.session_state.authenticated:
    st.title("📊 Retail Customer Analytics Dashboard")
    st.markdown("---")
    st.success(f"Welcome, {st.session_state.username}! Your dashboard is ready.")
    st.write("🎨 All your insights with a colorful touch — enjoy exploring your data!")




    # 1) Customer Engagement Over Time
    st.header("📈 Customer Engagement Over Time")
    if "SPEND" in df_transactions.columns:
        temp = df_transactions.dropna(subset=["date"]).copy()
        if not temp.empty:
            weekly = temp.groupby(temp["date"].dt.to_period("W"))["SPEND"].sum().reset_index()
            weekly["ds"] = weekly["date"].dt.start_time
            st.line_chart(weekly.set_index("ds")["SPEND"])
        else:
            st.info("No valid dates found to plot engagement over time.")
    else:
        st.info("SPEND column not found in transactions.")

    # 2) Demographics and Engagement
    st.header("👨👩👧 Demographics and Engagement")
    demo_options = [c for c in ["INCOME_RANGE", "AGE_RANGE", "CHILDREN"] if c in full_df.columns]
    if demo_options:
        selected_demo = st.selectbox("Segment by:", demo_options)
        demo_spending = full_df.groupby(selected_demo)["SPEND"].sum().reset_index()
        st.bar_chart(demo_spending.rename(columns={selected_demo: "index"}).set_index("index"))
    else:
        st.info("No demographic columns available to segment.")

    # 3) Customer Segmentation
    st.header("🔍 Customer Segmentation")
    if "SPEND" in full_df.columns and "hshd_num" in full_df.columns:
        seg = full_df.groupby("hshd_num").agg(SPEND=("SPEND", "sum"))
        if "INCOME_RANGE" in full_df.columns:
            seg["INCOME_RANGE"] = full_df.groupby("hshd_num")["INCOME_RANGE"].first()
        if "AGE_RANGE" in full_df.columns:
            seg["AGE_RANGE"] = full_df.groupby("hshd_num")["AGE_RANGE"].first()
        st.dataframe(seg.sort_values(by="SPEND", ascending=False).head(10))
    else:
        st.info("Required columns missing for segmentation.")

    # 4) Loyalty Program Effect
    st.header("🌟 Loyalty Program Effect")
    if "LOYALTY_FLAG" in full_df.columns and "SPEND" in full_df.columns:
        loyalty = full_df.groupby("LOYALTY_FLAG")["SPEND"].agg(["sum", "mean"]).reset_index()
        st.dataframe(loyalty)
    else:
        st.info("LOYALTY_FLAG or SPEND not available.")

    # 5) Basket Analysis
    st.header("🧺 Basket Analysis")
    if {"BASKET_NUM", "product_num", "SPEND"}.issubset(df_transactions.columns):
        basket = df_transactions.groupby(["BASKET_NUM", "product_num"])["SPEND"].sum().reset_index()
        top_products = basket.groupby("product_num")["SPEND"].sum().nlargest(10).reset_index()
        top_products = top_products.merge(df_products, on="product_num", how="left")
        if "COMMODITY" in top_products.columns:
            st.bar_chart(top_products.set_index("COMMODITY")["SPEND"])
            product_spending = top_products.groupby("COMMODITY")["SPEND"].sum().reset_index()
            fig = px.pie(product_spending, values="SPEND", names="COMMODITY", title="Spending Distribution by Product Category")
            st.plotly_chart(fig)
        else:
            st.dataframe(top_products)
    else:
        st.info("Basket columns not available.")

    # 6) Seasonal Spending Patterns
    st.header("📆 Seasonal Spending Patterns")
    if "date" in df_transactions.columns and "SPEND" in df_transactions.columns:
        temp = df_transactions.dropna(subset=["date"]).copy()
        if not temp.empty:
            temp["month"] = temp["date"].dt.month_name()
            seasonal = temp.groupby("month")["SPEND"].sum().reset_index()
            seasonal["month"] = pd.Categorical(
                seasonal["month"],
                categories=[
                    "January","February","March","April","May","June",
                    "July","August","September","October","November","December"
                ],
                ordered=True,
            )
            seasonal = seasonal.sort_values("month")
            st.bar_chart(seasonal.set_index("month"))
        else:
            st.info("No valid dates to compute seasonal patterns.")
    else:
        st.info("date/SPEND not available for seasonal analysis.")

    # 7) Customer Lifetime Value
    st.header("💰 Customer Lifetime Value")
    if {"hshd_num", "SPEND"}.issubset(df_transactions.columns):
        clv = df_transactions.groupby("hshd_num")["SPEND"].sum().reset_index().sort_values(by="SPEND", ascending=False)
        st.dataframe(clv.head(10))
    else:
        st.info("CLV requires hshd_num and SPEND.")

    # 8) Customer Spending by Product Category
    st.header("📊 Customer Spending by Product Category")
    if {"COMMODITY", "SPEND"}.issubset(full_df.columns):
        category_spending = full_df.groupby("COMMODITY")["SPEND"].sum().reset_index()
        st.bar_chart(category_spending.set_index("COMMODITY")["SPEND"])
    else:
        st.info("COMMODITY/SPEND not available.")

    # 9) Top 10 Customers
    st.header("🏆 Top 10 Customers by Spending")
    if {"hshd_num", "SPEND"}.issubset(full_df.columns):
        top_customers = full_df.groupby("hshd_num")["SPEND"].sum().reset_index().sort_values(by="SPEND", ascending=False)
        st.dataframe(top_customers.head(10))

    # 10) Age Group Spending
    st.header("📈 Trends in Age Group Spending")
    if {"AGE_RANGE", "SPEND"}.issubset(full_df.columns):
        age_group_spending = full_df.groupby("AGE_RANGE")["SPEND"].sum().reset_index()
        st.bar_chart(age_group_spending.set_index("AGE_RANGE")["SPEND"])
        fig = px.pie(age_group_spending, values="SPEND", names="AGE_RANGE", title="Spending Distribution by Age Group")
        st.plotly_chart(fig)

    # 11) Search Transactions by Household Number (in-memory search)
    st.header("🔎 Search Transactions by Household Number")
    hshd_num_input = st.text_input("Enter Household Number (HSHD_NUM) to search:")
    if hshd_num_input:
        try:
            hval = int(hshd_num_input)
            key = "hshd_num" if "hshd_num" in df_transactions.columns else None
            if key is None:
                st.error("No household key found in transactions (expected 'hshd_num').")
            else:
                out = df_transactions[df_transactions[key] == hval].copy()
                if not out.empty and not df_products.empty:
                    # Attach product info if present
                    attach_cols = [c for c in ["product_num", "DEPARTMENT", "COMMODITY"] if c in df_products.columns or c == "product_num"]
                    out = out.merge(df_products[[c for c in attach_cols if c in df_products.columns]],
                                    on="product_num", how="left")
                if not out.empty:
                    sort_cols = [c for c in ["BASKET_NUM", "date", "product_num"] if c in out.columns]
                    if sort_cols:
                        out = out.sort_values(by=sort_cols)
                    st.success(f"Found {len(out)} records for HSHD_NUM {hval}")
                    st.dataframe(out)
                else:
                    st.warning("No data found for the entered Household Number.")
        except ValueError:
            st.error("Please enter a valid numeric Household Number (HSHD_NUM).")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # =========================
    # ML Tabs
    # =========================
    tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧺 Basket Spend Regression"])

    with tab1:
        st.header("Churn Prediction: Customer Engagement Over Time")

        # Filters
        dept_opts = ["All"] + (sorted(df_products["DEPARTMENT"].dropna().unique()) if "DEPARTMENT" in df_products.columns else [])
        comm_opts = ["All"] + (sorted(df_products["COMMODITY"].dropna().unique()) if "COMMODITY" in df_products.columns else [])
        brand_opts = ["All"] + (sorted(df_products["BRAND_TY"].dropna().unique()) if "BRAND_TY" in df_products.columns else [])
        org_opts = ["All"] + (sorted(df_products["NATURAL_ORGANIC_FLAG"].dropna().unique()) if "NATURAL_ORGANIC_FLAG" in df_products.columns else [])

        col1, col2, col3, col4 = st.columns(4)
        department = col1.selectbox("Select Department:", dept_opts)
        commodity = col2.selectbox("Select Commodity:", comm_opts)
        brand_type = col3.selectbox("Select Brand Type:", brand_opts)
        organic_flag = col4.selectbox("Select Organic:", org_opts)

        if st.button("Apply Filters", key="churn_apply"):
            filtered = df_transactions.merge(df_products, on="product_num", how="left")
            if department != "All" and "DEPARTMENT" in filtered.columns:
                filtered = filtered[filtered["DEPARTMENT"] == department]
            if commodity != "All" and "COMMODITY" in filtered.columns:
                filtered = filtered[filtered["COMMODITY"] == commodity]
            if brand_type != "All" and "BRAND_TY" in filtered.columns:
                filtered = filtered[filtered["BRAND_TY"] == brand_type]
            if organic_flag != "All" and "NATURAL_ORGANIC_FLAG" in filtered.columns:
                filtered = filtered[filtered["NATURAL_ORGANIC_FLAG"] == organic_flag]

            if "date" in filtered.columns and "SPEND" in filtered.columns and not filtered.empty:
                churn_df = filtered.dropna(subset=["date"]).groupby("date")["SPEND"].sum().reset_index().sort_values("date")
                st.line_chart(churn_df.set_index("date")["SPEND"])
                with st.expander("📄 Raw Data"):
                    st.dataframe(churn_df)
            else:
                st.warning("No data found for selected filters.")

        st.subheader("ML Churn Prediction: At-Risk Customers")
        if "date" in df_transactions.columns and "SPEND" in df_transactions.columns and "BASKET_NUM" in df_transactions.columns and "hshd_num" in df_transactions.columns:
            tmp = df_transactions.dropna(subset=["date"]).copy()
            if not tmp.empty:
                now = tmp["date"].max()
                rfm = tmp.groupby("hshd_num").agg(
                    recency=("date", lambda x: (now - x.max()).days),
                    frequency=("BASKET_NUM", "nunique"),
                    monetary=("SPEND", "sum"),
                )
                # Label churn if inactive > 12 weeks (~84 days)
                rfm["churn"] = (rfm["recency"] > 84).astype(int)

                X = rfm[["recency", "frequency", "monetary"]]
                y = rfm["churn"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                st.write("**Classification Report:**")
                st.text(classification_report(y_test, y_pred))

                st.write("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(
                    confusion_matrix(y_test, y_pred),
                    columns=["Predicted Not Churn", "Predicted Churn"],
                    index=["Actual Not Churn", "Actual Churn"],
                ))

                feat_imp = pd.Series(clf.feature_importances_, index=["Recency", "Frequency", "Monetary"])
                st.write("**Feature Importances:**")
                st.bar_chart(feat_imp)
            else:
                st.info("No valid dates to compute churn ML.")
        else:
            st.info("Need columns: date, SPEND, BASKET_NUM, hshd_num for churn ML.")

    with tab2:
        st.header("Basket Analysis - Predicting Total Spend")
        if {"BASKET_NUM", "SPEND", "product_num"}.issubset(df_transactions.columns):
            basket_merged = df_transactions.merge(df_products, on="product_num", how="left")
            use_cols = [c for c in ["DEPARTMENT", "COMMODITY", "BRAND_TY", "NATURAL_ORGANIC_FLAG"] if c in basket_merged.columns]
            if not use_cols:
                st.info("No categorical product columns available for regression features.")
            else:
                feats = pd.get_dummies(basket_merged[["BASKET_NUM"] + use_cols], drop_first=True)
                X_basket = feats.groupby("BASKET_NUM").sum()
                y_basket = basket_merged.groupby("BASKET_NUM")["SPEND"].sum()

                X_train, X_test, y_train, y_test = train_test_split(X_basket, y_basket, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.write(f"**R² Score:** {r2:.3f}")
                st.write(f"**MSE:** {mse:.2f}")

                chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
                st.line_chart(chart_df.reset_index(drop=True))
                st.subheader("📄 Actual vs. Predicted Spend Table")
                st.dataframe(chart_df)

                csv = chart_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="predicted_vs_actual_basket_spend.csv",
                    mime="text/csv",
                )
        else:
            st.info("BASKET_NUM, product_num, and SPEND are required for basket regression.")
