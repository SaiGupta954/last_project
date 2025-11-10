# =========================
# Retail Insights Dashboard 
# =========================
import os
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix


# =========================
# Basic App Setup
# =========================
st.set_page_config(page_title="🍭️ Retail Insights Dashboard", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSETS_DIR = Path(__file__).parent / "assets"

# Session boot
if "user_db" not in st.session_state:
    st.session_state.user_db = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None


# =========================
# Auth helpers
# =========================
def make_hashes(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password: str, hashed_text: str) -> bool:
    return make_hashes(password) == hashed_text

def login_signup_inline():
    """
    Inline (non-sidebar) login/signup box that uses st.session_state.user_db.
    Sets st.session_state.authenticated and st.session_state.username when login succeeds.
    """
    tab_login, tab_signup = st.tabs(["🔐 Login", "🆕 Signup"])

    with tab_signup:
        st.subheader("Create Account")
        new_user = st.text_input("Username", key="signup_user")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create account", key="signup_btn"):
            if new_user and new_password:
                if new_user in st.session_state.user_db:
                    st.error("Username already exists.")
                else:
                    st.session_state.user_db[new_user] = {
                        "email": new_email,
                        "password": make_hashes(new_password),
                    }
                    st.success("Signup successful. Please switch to Login.")
            else:
                st.error("Username and password cannot be empty.")

    with tab_login:
        st.subheader("Welcome back")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        col_a, col_b = st.columns([1,1])
        with col_a:
            if st.button("Login", key="login_btn", use_container_width=True):
                user = st.session_state.user_db.get(username)
                if user and check_hashes(password, user["password"]):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with col_b:
            if st.button("Login as Demo", key="demo_btn", use_container_width=True):
                # Create a demo user on the fly
                demo_user = "demo_user"
                demo_pw_h = make_hashes("demo_pass")
                st.session_state.user_db.setdefault(demo_user, {"email": "demo@example.com", "password": demo_pw_h})
                st.session_state.authenticated = True
                st.session_state.username = demo_user
                st.rerun()


# =========================
# Data Loading (FREE: local files or uploads)
# =========================
def read_any(path_parquet: str, path_csv: str) -> pd.DataFrame:
    """Read Parquet if present, else CSV; return empty DataFrame if neither exists."""
    if os.path.exists(path_parquet):
        return pd.read_parquet(path_parquet)
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    return pd.DataFrame()

@st.cache_data(show_spinner="Loading data…", ttl=600)
def load_local_data():
    """Load data from ./data (Parquet or CSV)."""
    df_transactions = read_any(
        os.path.join(DATA_DIR, "Transactions.parquet"),
        os.path.join(DATA_DIR, "Transactions.csv"),
    )
    df_households = read_any(
        os.path.join(DATA_DIR, "Households.parquet"),
        os.path.join(DATA_DIR, "Households.csv"),
    )
    df_products = read_any(
        os.path.join(DATA_DIR, "Products.parquet"),
        os.path.join(DATA_DIR, "Products.csv"),
    )

    for df in (df_transactions, df_households, df_products):
        if not df.empty:
            df.columns = df.columns.str.strip()

    return df_transactions, df_households, df_products

# ---- robust column normalization (prevents KeyErrors) ----
def _norm(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

def _rename_if_present(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    norm_cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        nc = _norm(cand)
        if nc in norm_cols:
            real_col = norm_cols[nc]
            if real_col != target and target not in df.columns:
                df.rename(columns={real_col: target}, inplace=True)
            break
    return df

def normalize_columns(df_transactions, df_households, df_products):
    # common variants that might appear in files
    hshd_candidates = [
        "hshd_num", "HSHD_NUM", "hshd", "household", "household_num",
        "householdnumber", "hshdnumber", "householdid", "hshdid"
    ]
    prod_candidates = [
        "product_num", "PRODUCT_NUM", "product", "productnumber",
        "prod_num", "prodid", "productid"
    ]

    # transactions
    df_transactions = _rename_if_present(df_transactions, "hshd_num", hshd_candidates)
    df_transactions = _rename_if_present(df_transactions, "product_num", prod_candidates)

    # households
    df_households = _rename_if_present(df_households, "hshd_num", hshd_candidates)

    # products
    df_products = _rename_if_present(df_products, "product_num", prod_candidates)

    return df_transactions, df_households, df_products

def build_full_df(df_transactions, df_households, df_products):
    """Merge safely only if join keys exist on both sides."""
    full = df_transactions.copy()

    if not full.empty and not df_households.empty:
        if "hshd_num" in full.columns and "hshd_num" in df_households.columns:
            full = full.merge(df_households, on="hshd_num", how="left")
        else:
            st.warning("Household key 'hshd_num' missing in one of the tables; skipping households merge.")

    if not full.empty and not df_products.empty:
        if "product_num" in full.columns and "product_num" in df_products.columns:
            full = full.merge(df_products, on="product_num", how="left")
        else:
            st.warning("Product key 'product_num' missing in one of the tables; skipping products merge.")

    return full


# =========================
# HERO + LOGIN (always visible on app start)
# =========================
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; }
.hero-card, .login-card {
  background: #0f1a42;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  box-shadow: 0 16px 40px rgba(0,0,0,.25);
}
.login-card { padding: 20px 18px; }
.hero-card { position: relative; overflow: hidden; }
.hero-header {
  background: linear-gradient(135deg, #0b5fff 0%, #0a2a6a 100%);
  color: #fff; padding: 14px 16px; font-weight: 800; font-size: 18px;
}
.hero-body { padding: 10px; background:#0b1437; }
.hero-body img { width: 100%; border-radius: 12px; display:block; }
.badge-live {
  position:absolute; top:14px; right:14px; background:#ff2d55;
  color:#fff; font-weight:700; font-size:12px; padding:6px 10px; border-radius:999px;
  letter-spacing:.3px;
}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2], vertical_alignment="center")

# LEFT: Login/Signup panel (inline)
with left:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.subheader("🔐 Login to Continue")
    login_signup_inline()
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT: Hero animated dashboard image (LIVE)
with right:
    hero_path = ASSETS_DIR / "hero_dashboard.gif"
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-header">Retail Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-live">LIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-body">', unsafe_allow_html=True)
    if hero_path.exists():
        # Optional: subtle refresh so GIF feels alive if replaced server-side
        st.markdown("<meta http-equiv='refresh' content='12'>", unsafe_allow_html=True)
        st.image(str(hero_path), use_container_width=True)
    else:
        st.info("Upload an animated image to assets/hero_dashboard.gif")
    st.markdown('</div></div>', unsafe_allow_html=True)


# =========================
# Main (only for authenticated users)
# =========================
if st.session_state.authenticated:
    st.title("📂 Dataset Loader")

    # Optional uploads – override repo data at runtime
    uploaded_transactions = st.file_uploader("Upload Transactions (CSV)", type="csv")
    uploaded_households = st.file_uploader("Upload Households (CSV)", type="csv")
    uploaded_products = st.file_uploader("Upload Products (CSV)", type="csv")

    # Load from repo first
    df_transactions, df_households, df_products = load_local_data()

    # Override with uploads if provided
    if uploaded_transactions is not None:
        df_transactions = pd.read_csv(uploaded_transactions)
    if uploaded_households is not None:
        df_households = pd.read_csv(uploaded_households)
    if uploaded_products is not None:
        df_products = pd.read_csv(uploaded_products)

    # Button to reload repo files (UX nicety)
    if st.button("📥 Load Latest Data from Repository"):
        df_transactions, df_households, df_products = load_local_data()
        st.success("Loaded data from local repository files.")

    # Show samples
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.caption("Sample Transactions")
        st.dataframe(df_transactions.head(5))
    with col_b:
        st.caption("Sample Households")
        st.dataframe(df_households.head(5))
    with col_c:
        st.caption("Sample Products")
        st.dataframe(df_products.head(5))

    # Normalize cols used later
    df_transactions, df_households, df_products = normalize_columns(df_transactions, df_households, df_products)

    # Guard if data missing
    if df_transactions.empty or df_households.empty or df_products.empty:
        st.warning("Please ensure all three datasets (Transactions, Households, Products) exist (in /data or via upload).")
        st.stop()

    # Derive a date column
    if {"YEAR", "WEEK_NUM"}.issubset(df_transactions.columns):
        try:
            df_transactions["date"] = pd.to_datetime(
                df_transactions["YEAR"].astype(str) + df_transactions["WEEK_NUM"].astype(str) + "0",
                format="%Y%U%w",
                errors="coerce",
            )
        except Exception:
            df_transactions["date"] = pd.NaT
    elif "PURCHASE_" in df_transactions.columns:
        df_transactions["date"] = pd.to_datetime(df_transactions["PURCHASE_"], errors="coerce")
    else:
        df_transactions["date"] = pd.NaT

    # Build merged frame
    full_df = build_full_df(df_transactions, df_households, df_products)

    # =========================
    # Dashboard
    # =========================
    st.title("📊 Retail Customer Analytics Dashboard")

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
