# --- 1. Imports ---
import pandas as pd
import streamlit as st
import pyodbc
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.express as px

# --- 2. Database connection and data loading functions ---

server = 'newretailserver123.database.windows.net'
database = 'RetailDB'
username = 'azureuser'
password = 'YourStrongP@ssw0rd'
driver = '{ODBC Driver 17 for SQL Server}'

def get_connection():
    return pyodbc.connect(
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
    )

@st.cache_data(ttl=600)
def load_data_from_db():
    conn = get_connection()
    df_transactions = pd.read_sql("SELECT * FROM Transactions", conn)
    df_households = pd.read_sql("SELECT * FROM Households", conn)
    df_products = pd.read_sql("SELECT * FROM Products", conn)
    conn.close()
    df_transactions.columns = df_transactions.columns.str.strip()
    df_households.columns = df_households.columns.str.strip()
    df_products.columns = df_products.columns.str.strip()
    return df_transactions, df_households, df_products

# --- 3. Password hashing helper functions ---

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# --- 4. Streamlit page config ---
st.set_page_config(page_title="🍭 Retail Insights Dashboard", layout="wide")

# --- 5. Session initialization ---
if 'user_db' not in st.session_state:
    st.session_state.user_db = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 6. Login and Signup functions ---

def login_signup():
    if not st.session_state.authenticated:
        auth_option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])
        if auth_option == "Signup":
            st.sidebar.subheader("Create Account")
            new_user = st.sidebar.text_input("Username")
            new_email = st.sidebar.text_input("Email")
            new_password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Signup"):
                if new_user and new_password:
                    if new_user in st.session_state.user_db:
                        st.sidebar.error("Username already exists.")
                    else:
                        hashed_pw = make_hashes(new_password)
                        st.session_state.user_db[new_user] = {"email": new_email, "password": hashed_pw}
                        st.sidebar.success("Signup successful. Please login.")
                else:
                    st.sidebar.error("Username and password cannot be empty.")
        elif auth_option == "Login":
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Login"):
                user = st.session_state.user_db.get(username)
                if user and check_hashes(password, user['password']):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")

# --- 7. Run Login/Signup ---
login_signup()

# --- 8. After Login: Upload files or Load from Database ---
if st.session_state.authenticated:
    st.title("📂 Dataset Loader")

    uploaded_transactions = st.file_uploader("Upload Transactions Dataset (CSV)", type="csv")
    uploaded_households = st.file_uploader("Upload Households Dataset (CSV)", type="csv")
    uploaded_products = st.file_uploader("Upload Products Dataset (CSV)", type="csv")

    if uploaded_transactions is not None:
        st.session_state['transactions_df'] = pd.read_csv(uploaded_transactions)
    if uploaded_households is not None:
        st.session_state['households_df'] = pd.read_csv(uploaded_households)
    if uploaded_products is not None:
        st.session_state['products_df'] = pd.read_csv(uploaded_products)

    if 'transactions_df' in st.session_state:
        st.write("Sample Transactions Data", st.session_state['transactions_df'].head())
    if 'households_df' in st.session_state:
        st.write("Sample Households Data", st.session_state['households_df'].head())
    if 'products_df' in st.session_state:
        st.write("Sample Products Data", st.session_state['products_df'].head())

    if (('transactions_df' not in st.session_state) or
        ('households_df' not in st.session_state) or
        ('products_df' not in st.session_state)):
        if st.button("📥 Load Latest Data from Database"):
            tdf, hdf, pdf = load_data_from_db()
            st.session_state['transactions_df'] = tdf
            st.session_state['households_df'] = hdf
            st.session_state['products_df'] = pdf
            st.success("Loaded data successfully from Database.")

# --- 9. Now Continue with Dashboard after Upload or Load ---

    if all(x in st.session_state for x in ['transactions_df', 'households_df', 'products_df']):
        df_transactions = st.session_state['transactions_df']
        df_households = st.session_state['households_df']
        df_products = st.session_state['products_df']

        # Rename and merge
        df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
        df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
        df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

        full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
        full_df = full_df.merge(df_products, on='product_num', how='left')
        df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')

                # 🎯 Start Dashboard Analytics

        st.title("📊 Retail Customer Analytics Dashboard")

        st.header("📈 Customer Engagement Over Time")
        weekly_engagement = df_transactions.groupby(df_transactions['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
        weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
        st.line_chart(weekly_engagement.set_index('ds')['SPEND'])

        st.header("👨‍👩‍👧 Demographics and Engagement")
        selected_demo = st.selectbox("Segment by:", ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'])
        demo_spending = full_df.groupby(selected_demo)['SPEND'].sum().reset_index()
        st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))

        st.header("🔍 Customer Segmentation (Top 10 Households)")
        segmentation = full_df.groupby(['hshd_num']).agg({'SPEND': 'sum', 'INCOME_RANGE': 'first', 'AGE_RANGE': 'first'})
        st.dataframe(segmentation.sort_values(by='SPEND', ascending=False).head(10))

        st.header("🌟 Loyalty Program Effect")
        if 'LOYALTY_FLAG' in df_households.columns:
            loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
            st.dataframe(loyalty)

        st.header("🧺 Basket Analysis")
        basket = df_transactions.groupby(['BASKET_NUM', 'product_num'])['SPEND'].sum().reset_index()
        top_products = basket.groupby('product_num')['SPEND'].sum().nlargest(10).reset_index()
        top_products = top_products.merge(df_products, on='product_num', how='left')
        if 'COMMODITY' in top_products.columns:
            st.bar_chart(top_products.set_index('COMMODITY')['SPEND'])
            product_spending = top_products.groupby('COMMODITY')['SPEND'].sum().reset_index()
            fig = px.pie(product_spending, values='SPEND', names='COMMODITY', title='Spending Distribution by Product Category')
            st.plotly_chart(fig)
        else:
            st.dataframe(top_products)

        st.header("📆 Seasonal Spending Patterns")
        df_transactions['month'] = df_transactions['date'].dt.month_name()
        seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
        seasonal['month'] = pd.Categorical(seasonal['month'], categories=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], ordered=True)
        seasonal = seasonal.sort_values('month')
        st.bar_chart(seasonal.set_index('month'))

        st.header("💰 Customer Lifetime Value (CLV)")
        clv = df_transactions.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
        st.dataframe(clv.head(10))

        st.header("📊 Customer Spending by Product Category")
        category_spending = full_df.groupby('COMMODITY')['SPEND'].sum().reset_index()
        st.bar_chart(category_spending.set_index('COMMODITY')['SPEND'])

        st.header("🏆 Top 10 Customers by Spending")
        top_customers = full_df.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
        st.dataframe(top_customers.head(10))

        st.header("📈 Trends in Age Group Spending")
        age_group_spending = full_df.groupby('AGE_RANGE')['SPEND'].sum().reset_index()
        st.bar_chart(age_group_spending.set_index('AGE_RANGE')['SPEND'])
        fig = px.pie(age_group_spending, values='SPEND', names='AGE_RANGE', title='Spending Distribution by Age Group')
        st.plotly_chart(fig)

        # --- 🔎 Search Household Number Section ---

        st.header("🔎 Search Transactions by Household Number (HSHD_NUM)")

        def fetch_data_by_hshd(hshd_num):
            conn = get_connection()
            query = """
            SELECT 
                H.HSHD_NUM, 
                T.BASKET_NUM, 
                T.PURCHASE_ AS Date,  
                P.PRODUCT_NUM, 
                P.DEPARTMENT, 
                P.COMMODITY 
            FROM dbo.households H
            JOIN dbo.transactions T ON H.HSHD_NUM = T.HSHD_NUM
            LEFT JOIN dbo.products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
            WHERE H.HSHD_NUM = ?
            ORDER BY H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_, P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY;
            """
            df = pd.read_sql(query, conn, params=[hshd_num])
            conn.close()
            return df

        hshd_num_input = st.text_input("Enter Household Number (HSHD_NUM) to search:")

        if hshd_num_input:
            try:
                hshd_num_value = int(hshd_num_input)
                with st.spinner('Fetching data...'):
                    search_result = fetch_data_by_hshd(hshd_num_value)

                if not search_result.empty:
                    st.success(f"Found {len(search_result)} records for HSHD_NUM {hshd_num_value}")
                    st.dataframe(search_result)
                else:
                    st.warning("No data found for the entered Household Number.")

            except ValueError:
                st.error("Please enter a valid numeric Household Number (HSHD_NUM).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

