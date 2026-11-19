import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Vercel Python runtime compatibility export
def handler(*args, **kwargs):
    return {"statusCode": 200, "body": "Retail Crash-Test Engine (Streamlit)"}
app = handler
application = handler

st.set_page_config(
    page_title="Retail Crash-Test Engine",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* KPI metric cards */
    .kpi-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .kpi-card-red   { border-left: 4px solid #ff4b4b; }
    .kpi-card-teal  { border-left: 4px solid #00d4aa; }
    .kpi-card-gold  { border-left: 4px solid #ffd700; }
    .kpi-card-purple{ border-left: 4px solid #667eea; }

    .kpi-label { font-size: 0.78rem; color: #9aa0b0; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #ffffff; line-height: 1.1; }
    .kpi-badge { font-size: 0.75rem; margin-top: 0.3rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 0.6rem 1rem; }

    /* Remove extra top padding */
    header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
PERSONA_NAMES = [
    "Core VIP Pillars",
    "Price-Sensitive Bargain Hunters",
    "High-Ticket Occasional Buyers",
    "At-Risk / Low-Frequency"
]
CATEGORY_COLORS = {
    'Electronics': '#667eea',
    'Clothing': '#f093fb',
    'Beauty': '#f5576c',
    'Home & Kitchen': '#4facfe'
}
SAMPLE_DATA_PATH = "sample_retail_data.csv"

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


# ── Helper functions ─────────────────────────────────────────────────────────

def load_sample_data() -> pd.DataFrame:
    """Load sample CSV if it exists, otherwise generate synthetic data."""
    if os.path.exists(SAMPLE_DATA_PATH):
        try:
            df = pd.read_csv(SAMPLE_DATA_PATH)
            if len(df) > 0:
                return df
        except Exception:
            pass

    # Generate 1 500-row synthetic dataset
    rng = np.random.default_rng(42)
    n = 1500

    categories = ['Electronics', 'Clothing', 'Beauty', 'Home & Kitchen']
    cat_weights = [0.25, 0.30, 0.25, 0.20]
    price_ranges = {
        'Electronics':    (150, 1200),
        'Clothing':       (20,  150),
        'Beauty':         (15,  80),
        'Home & Kitchen': (30,  250),
    }
    # Seasonality: month weights per category
    season_weights = {
        'Electronics':    [0.05, 0.05, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10, 0.14, 0.14],
        'Clothing':       [0.07, 0.07, 0.10, 0.10, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07],
        'Beauty':         [0.07, 0.08, 0.11, 0.11, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07, 0.05],
        'Home & Kitchen': [0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.09, 0.08, 0.07],
    }

    product_category = rng.choice(categories, size=n, p=cat_weights)
    quantity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.40, 0.30, 0.15, 0.10, 0.05])

    prices = np.array([
        round(float(rng.uniform(*price_ranges[cat])), 2)
        for cat in product_category
    ])
    total_amount = np.round(quantity * prices, 2)

    # Dates with seasonality
    dates = []
    base_year = 2023
    for cat in product_category:
        w = np.array(season_weights[cat])
        w = w / w.sum()
        month = rng.choice(range(1, 13), p=w)
        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31][month - 1]
        day = rng.integers(1, days_in_month + 1)
        dates.append(f"{base_year}-{month:02d}-{day:02d}")

    ages = np.clip(rng.normal(35, 10, n).astype(int), 18, 65)
    genders = rng.choice(['Male', 'Female'], size=n, p=[0.50, 0.50])
    customer_ids = [f"C{rng.integers(1, 201):03d}" for _ in range(n)]
    transaction_ids = [f"T{i+1:04d}" for i in range(n)]

    df = pd.DataFrame({
        'Transaction ID': transaction_ids,
        'Date': dates,
        'Customer ID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Product Category': product_category,
        'Quantity': quantity,
        'Price per Unit': prices,
        'Total Amount': total_amount,
    })
    return df


def smart_column_mapper(df: pd.DataFrame) -> dict:
    """Map standard column names to actual DataFrame column names."""
    aliases = {
        'Date': ['date', 'transaction date', 'order date', 'purchase date', 'trans date'],
        'Product Category': ['product category', 'category', 'item category', 'product type', 'dept'],
        'Age': ['age', 'customer age', 'buyer age'],
        'Gender': ['gender', 'sex', 'customer gender'],
        'Quantity': ['quantity', 'qty', 'units', 'unit qty', 'items'],
        'Price per Unit': ['price per unit', 'price', 'unit price', 'price/unit', 'unit cost'],
        'Total Amount': ['total amount', 'total', 'revenue', 'sales', 'amount', 'total revenue', 'total sales'],
    }
    col_lower = {c.strip().lower(): c for c in df.columns}
    mapping = {}
    for std_name, alias_list in aliases.items():
        for alias in alias_list:
            if alias in col_lower:
                mapping[std_name] = col_lower[alias]
                break
    return mapping


def clean_and_prepare(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename, coerce, derive time features."""
    df = df.copy()
    rename = {v: k for k, v in col_map.items()}
    df.rename(columns=rename, inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
    else:
        df['Date'] = pd.NaT

    for col in ['Age', 'Quantity', 'Price per Unit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Total Amount: coerce or compute
    if 'Total Amount' in df.columns:
        df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')
        mask = df['Total Amount'].isna()
        if mask.any() and 'Quantity' in df.columns and 'Price per Unit' in df.columns:
            df.loc[mask, 'Total Amount'] = df.loc[mask, 'Quantity'] * df.loc[mask, 'Price per Unit']
    elif 'Quantity' in df.columns and 'Price per Unit' in df.columns:
        df['Total Amount'] = df['Quantity'] * df['Price per Unit']

    key_cols = [c for c in ['Age', 'Quantity', 'Price per Unit', 'Total Amount'] if c in df.columns]
    df.dropna(subset=key_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'Date' in df.columns and not df['Date'].isna().all():
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
    else:
        df['Month'] = 1
        df['DayOfWeek'] = 0
        df['Quarter'] = 1

    return df


def run_clustering(df: pd.DataFrame, n_clusters: int = 4):
    """Scale + KMeans; return (df_with_cluster, scaler, kmeans, silhouette)."""
    features = ['Age', 'Quantity', 'Price per Unit', 'Month', 'DayOfWeek']
    available = [f for f in features if f in df.columns]
    X = df[available].copy().fillna(df[available].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df['Cluster'] = labels

    sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0
    return df, scaler, kmeans, sil


def assign_persona_labels(kmeans, scaler, n_clusters: int) -> dict:
    """Heuristic persona assignment based on cluster centroids."""
    features = ['Age', 'Quantity', 'Price per Unit', 'Month', 'DayOfWeek']
    centroids_scaled = kmeans.cluster_centers_

    n_feat = centroids_scaled.shape[1]
    feat_names = features[:n_feat]

    try:
        centroids = scaler.inverse_transform(centroids_scaled)
    except Exception:
        centroids = centroids_scaled

    centroid_df = pd.DataFrame(centroids, columns=feat_names)
    centroid_df['cluster'] = range(n_clusters)

    persona_map = {}
    assigned = set()

    if 'Price per Unit' in centroid_df.columns:
        # High-Ticket: highest Price per Unit
        avail = centroid_df[~centroid_df['cluster'].isin(assigned)]
        if not avail.empty:
            ht_idx = avail['Price per Unit'].idxmax()
            c_id = int(centroid_df.loc[ht_idx, 'cluster'])
            persona_map[c_id] = "High-Ticket Occasional Buyers"
            assigned.add(c_id)

    if 'Price per Unit' in centroid_df.columns and 'Quantity' in centroid_df.columns:
        avail = centroid_df[~centroid_df['cluster'].isin(assigned)]
        if not avail.empty:
            # Bargain: lowest Price + highest Quantity (score = price_rank + inverse_qty_rank)
            pr = avail['Price per Unit'].rank()
            qr = avail['Quantity'].rank(ascending=False)
            combined = pr + qr
            bh_idx = combined.idxmin()
            c_id = int(centroid_df.loc[bh_idx, 'cluster'])
            persona_map[c_id] = "Price-Sensitive Bargain Hunters"
            assigned.add(c_id)

    if 'Quantity' in centroid_df.columns:
        avail = centroid_df[~centroid_df['cluster'].isin(assigned)]
        if not avail.empty:
            vip_idx = avail['Quantity'].idxmax()
            c_id = int(centroid_df.loc[vip_idx, 'cluster'])
            persona_map[c_id] = "Core VIP Pillars"
            assigned.add(c_id)

    generic_idx = 0
    for i in centroid_df.index:
        c_id = int(centroid_df.loc[i, 'cluster'])
        if c_id not in assigned:
            if n_clusters == 4 and "At-Risk / Low-Frequency" not in persona_map.values():
                persona_map[c_id] = "At-Risk / Low-Frequency"
            else:
                persona_map[c_id] = f"Segment {generic_idx + 1}"
            assigned.add(c_id)
            generic_idx += 1

    return persona_map


def compute_fragility_score(df: pd.DataFrame):
    """Return (score, components_dict)."""
    total_revenue = df['Total Amount'].sum()

    # Category concentration
    cat_rev = df.groupby('Product Category')['Total Amount'].sum()
    top_cat_rev = cat_rev.max() if not cat_rev.empty else 0
    category_concentration = (top_cat_rev / total_revenue * 100) if total_revenue > 0 else 0

    # Cluster dependency
    if 'Cluster' in df.columns:
        cluster_rev = df.groupby('Cluster')['Total Amount'].sum()
        top_cluster_rev = cluster_rev.max()
        cluster_dependency = (top_cluster_rev / total_revenue * 100) if total_revenue > 0 else 0
    else:
        cluster_dependency = 0.0

    # Monthly volatility
    monthly_rev = df.groupby('Month')['Total Amount'].sum()
    if len(monthly_rev) > 1 and monthly_rev.mean() > 0:
        monthly_volatility = (monthly_rev.std() / monthly_rev.mean()) * 100
    else:
        monthly_volatility = 0.0

    # Normalize
    cc_norm = min(category_concentration, 100)
    cd_norm = min(cluster_dependency, 100)
    mv_norm = min(monthly_volatility * 2, 100)

    score = 0.40 * cc_norm + 0.35 * cd_norm + 0.25 * mv_norm
    components = {
        'Category Concentration': round(category_concentration, 1),
        'Cluster Dependency': round(cluster_dependency, 1),
        'Monthly Volatility': round(monthly_volatility, 1),
    }
    return round(score, 1), components


def generate_ceo_report(df: pd.DataFrame, fragility_score: float,
                        components: dict, persona_map: dict, silhouette_val: float) -> str:
    """Build markdown CEO report string."""
    total_revenue = df['Total Amount'].sum()
    total_txn = len(df)
    aov = total_revenue / total_txn if total_txn > 0 else 0

    if fragility_score >= 70:
        risk_badge = "🔴 HIGH RISK"
    elif fragility_score >= 40:
        risk_badge = "🟡 MEDIUM RISK"
    else:
        risk_badge = "🟢 LOW RISK"

    cc = components.get('Category Concentration', 0)
    cd = components.get('Cluster Dependency', 0)
    mv = components.get('Monthly Volatility', 0)

    # Build ML cluster table
    if 'Cluster' in df.columns:
        cluster_rows = []
        for cid, pname in sorted(persona_map.items()):
            cdf = df[df['Cluster'] == cid]
            cnt = len(cdf)
            rev = cdf['Total Amount'].sum()
            cluster_rows.append(f"| {pname} | {cnt:,} | ${rev:,.0f} |")
        cluster_table = "\n".join(cluster_rows)
    else:
        cluster_table = "| N/A | N/A | N/A |"

    # Peak month
    monthly_rev = df.groupby('Month')['Total Amount'].sum()
    peak_month_num = int(monthly_rev.idxmax()) if not monthly_rev.empty else 1
    peak_month_name = MONTH_NAMES.get(peak_month_num, 'N/A')
    min_month_rev = monthly_rev.min() if not monthly_rev.empty else 0
    max_month_rev = monthly_rev.max() if not monthly_rev.empty else 1
    volatility_drop = ((max_month_rev - min_month_rev) / max_month_rev * 100) if max_month_rev > 0 else 0

    top_cat = df['Product Category'].value_counts().index[0] if 'Product Category' in df.columns else "N/A"
    all_cats = list(df['Product Category'].unique()) if 'Product Category' in df.columns else []
    alt_cats = [c for c in all_cats if c != top_cat]
    alt_cat = alt_cats[0] if alt_cats else "other categories"

    # Vulnerable cluster (highest rev)
    if 'Cluster' in df.columns:
        cluster_rev = df.groupby('Cluster')['Total Amount'].sum()
        top_cluster_id = cluster_rev.idxmax()
        top_persona = persona_map.get(top_cluster_id, f"Cluster {top_cluster_id}")
        top_cluster_customers = df[df['Cluster'] == top_cluster_id]['Customer ID'].nunique() if 'Customer ID' in df.columns else len(df[df['Cluster'] == top_cluster_id])
        top_cluster_rev_pct = cluster_rev[top_cluster_id] / total_revenue * 100
        top_cluster_pct_cust = top_cluster_customers / df['Customer ID'].nunique() * 100 if 'Customer ID' in df.columns else 0
    else:
        top_persona = "N/A"
        top_cluster_customers = 0
        top_cluster_rev_pct = 0
        top_cluster_pct_cust = 0

    # High-ticket segment stats
    ht_cluster = [k for k, v in persona_map.items() if 'High-Ticket' in v]
    if ht_cluster and 'Cluster' in df.columns:
        ht_df = df[df['Cluster'].isin(ht_cluster)]
        ht_customers = ht_df['Customer ID'].nunique() if 'Customer ID' in df.columns else len(ht_df)
        ht_avg_txn = len(ht_df) / ht_customers if ht_customers > 0 else 0
        ht_avg_value = ht_df['Total Amount'].mean() if not ht_df.empty else 0
        upsell_opportunity = ht_customers * ht_avg_value * 0.2
    else:
        ht_customers = 0
        ht_avg_txn = 0
        upsell_opportunity = 0

    report = f"""# 🚨 CEO Crash-Test & Survival Report
**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}
**Powered by:** Retail Crash-Test Engine

---

## 📊 Executive Summary
- **Total Revenue:** ${total_revenue:,.0f}
- **Total Transactions:** {total_txn:,}
- **Avg Order Value:** ${aov:.2f}
- **Business Fragility Score:** {fragility_score}/100 — {risk_badge}

---

## ⚠️ Risk Drivers
| Risk Factor | Value | Weight |
|---|---|---|
| Category Concentration | {cc:.1f}% | 40% |
| Cluster Dependency | {cd:.1f}% | 35% |
| Monthly Volatility | {mv:.1f}% | 25% |

---

## 🛡️ Survival Shield — Strategic Actions

1. **Fragility Alert:** Your Business Fragility Score of **{fragility_score}/100** ({risk_badge}) signals {'immediate intervention required' if fragility_score >= 70 else 'moderate strategic attention' if fragility_score >= 40 else 'healthy resilience — maintain diversification'}.

2. **Category Concentration Risk:** `{top_cat}` drives **{cc:.1f}%** of total revenue. Diversify into `{alt_cat}` through targeted promotions and bundled offers to reduce single-category dependency.

3. **Segment Vulnerability:** Your `{top_persona}` segment represents **{top_cluster_pct_cust:.1f}%** of customers but generates **{top_cluster_rev_pct:.1f}%** of revenue. Launch a dedicated loyalty programme with early access and exclusive discounts to cement retention.

4. **Seasonality Management:** Peak revenue month is **{peak_month_name}**. Off-peak months see revenue drop **{volatility_drop:.1f}%** vs. peak. Introduce subscription boxes or pre-order incentives to smooth monthly cash flow.

5. **High-Ticket Upsell:** Your High-Ticket Occasional Buyers (**{ht_customers}** customers) average **{ht_avg_txn:.1f}** transactions per year. A 20% upsell / add-on conversion programme could unlock **${upsell_opportunity:,.0f}** in incremental revenue.

6. **Competitor Resilience:** Build a price-match guarantee or exclusive bundle for Price-Sensitive Bargain Hunters to reduce vulnerability to competitor undercutting.

7. **Supply Chain Buffer:** Top category `{top_cat}` concentration means any supply disruption directly hits {cc:.1f}% of revenue. Maintain a 90-day stock buffer and qualify two alternate suppliers.

---

## 🤖 ML Cluster Insights
| Persona | Transactions | Revenue |
|---|---|---|
{cluster_table}

**Silhouette Score:** {silhouette_val:.3f} (range −1 to +1; higher = better-defined clusters)

---

## 🔬 Methodology Note
K-Means Clustering with StandardScaler. Data leakage prevented by excluding Total Amount from features.
Features used: Age, Quantity, Price per Unit, Month, DayOfWeek.
Silhouette Score used for cluster quality evaluation.
"""
    return report


# ── KPI card helper ──────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, accent: str, badge: str = ""):
    color_map = {
        'red': 'kpi-card-red',
        'teal': 'kpi-card-teal',
        'gold': 'kpi-card-gold',
        'purple': 'kpi-card-purple',
    }
    css_class = color_map.get(accent, 'kpi-card-teal')
    badge_html = f'<div class="kpi-badge">{badge}</div>' if badge else ''
    st.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Retail Crash-Test Engine")
    st.markdown("**Upload your sales CSV or load sample data**")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    load_sample_btn = st.button("⚡ Load Sample Retail Data", type="primary", use_container_width=True)

    st.divider()
    n_clusters = st.slider("K-Means Clusters (k)", 3, 5, 4)

    # Column mapping for uploaded files
    if uploaded_file is not None and 'col_map' in st.session_state and st.session_state.get('data_source') == 'uploaded':
        st.subheader("📋 Column Mapping")
        col_map = st.session_state['col_map']
        df_raw = st.session_state.get('df_raw')
        if df_raw is not None:
            std_cols = ['Date', 'Product Category', 'Age', 'Gender', 'Quantity', 'Price per Unit', 'Total Amount']
            raw_cols = ['(skip)'] + list(df_raw.columns)
            updated_map = {}
            for std in std_cols:
                current = col_map.get(std, None)
                default_idx = raw_cols.index(current) if current in raw_cols else 0
                chosen = st.selectbox(f"{std}", raw_cols, index=default_idx, key=f"colmap_{std}")
                if chosen != '(skip)':
                    updated_map[std] = chosen
            st.session_state['col_map'] = updated_map

    st.divider()
    st.markdown("**About**")
    st.caption("Retail Crash-Test Engine v1.0  \nBuilt with Streamlit + scikit-learn + Plotly")


# ── Data ingestion ────────────────────────────────────────────────────────────
if load_sample_btn:
    with st.spinner("Loading sample retail data..."):
        try:
            df_sample = load_sample_data()
            st.session_state['df_raw'] = df_sample
            st.session_state['data_source'] = 'sample'
            st.session_state.pop('col_map', None)
            st.session_state.pop('df_clean', None)
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")

if uploaded_file is not None and st.session_state.get('data_source') != 'uploaded':
    try:
        df_up = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df_up
        st.session_state['data_source'] = 'uploaded'
        st.session_state['col_map'] = smart_column_mapper(df_up)
        st.session_state.pop('df_clean', None)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

# ── Process data if available ─────────────────────────────────────────────────
if 'df_raw' in st.session_state:
    try:
        df_raw = st.session_state['df_raw']

        # Column mapping
        if 'col_map' not in st.session_state:
            col_map = smart_column_mapper(df_raw)
            st.session_state['col_map'] = col_map
        else:
            col_map = st.session_state['col_map']

        # Clean
        df_clean = clean_and_prepare(df_raw, col_map)

        if len(df_clean) < n_clusters:
            st.error(f"Not enough rows ({len(df_clean)}) to run {n_clusters} clusters. Please upload more data.")
            st.stop()

        # Clustering
        prev_k = st.session_state.get('n_clusters_used', None)
        prev_source = st.session_state.get('data_source_cached', None)
        cur_source = st.session_state.get('data_source', 'sample')

        if (prev_k != n_clusters or prev_source != cur_source or 'df_clean' not in st.session_state):
            df_clustered, scaler, kmeans_model, silhouette_val = run_clustering(df_clean, n_clusters)
            persona_map = assign_persona_labels(kmeans_model, scaler, n_clusters)
            fragility_score, fragility_components = compute_fragility_score(df_clustered)

            st.session_state['df_clean'] = df_clustered
            st.session_state['scaler'] = scaler
            st.session_state['kmeans_model'] = kmeans_model
            st.session_state['silhouette_val'] = silhouette_val
            st.session_state['persona_map'] = persona_map
            st.session_state['fragility_score'] = fragility_score
            st.session_state['fragility_components'] = fragility_components
            st.session_state['n_clusters_used'] = n_clusters
            st.session_state['data_source_cached'] = cur_source
        else:
            df_clustered = st.session_state['df_clean']
            scaler = st.session_state['scaler']
            kmeans_model = st.session_state['kmeans_model']
            silhouette_val = st.session_state['silhouette_val']
            persona_map = st.session_state['persona_map']
            fragility_score = st.session_state['fragility_score']
            fragility_components = st.session_state['fragility_components']

    except Exception as e:
        st.error(f"⚠️ Data processing error: {e}")
        st.stop()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Crash-Test Overview",
        "💥 Live Disaster Simulator",
        "🛡️ Survival Shield & CEO Report",
        "🔬 ML Transparency"
    ])

    # ── TAB 1: Crash-Test Overview ────────────────────────────────────────────
    with tab1:
        st.markdown("## 🎯 Crash-Test Overview")

        df = df_clustered
        total_revenue = df['Total Amount'].sum()
        total_txn = len(df)
        aov = total_revenue / total_txn if total_txn > 0 else 0

        # Risk badge
        if fragility_score >= 70:
            risk_badge = "🔴 HIGH RISK"
            gauge_color = "#ff4b4b"
        elif fragility_score >= 40:
            risk_badge = "🟡 MEDIUM RISK"
            gauge_color = "#ffd700"
        else:
            risk_badge = "🟢 LOW RISK"
            gauge_color = "#00d4aa"

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Total Revenue", f"${total_revenue:,.0f}", "teal")
        with col2:
            kpi_card("Total Transactions", f"{total_txn:,}", "purple")
        with col3:
            kpi_card("Avg Order Value", f"${aov:.2f}", "gold")
        with col4:
            kpi_card("Fragility Score", f"{fragility_score} / 100", "red", risk_badge)

        st.divider()

        # Gauge + risk drivers
        col_gauge, col_charts = st.columns([1, 1])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fragility_score,
                title={'text': "Business Fragility Score", 'font': {'color': 'white', 'size': 16}},
                number={'font': {'color': 'white', 'size': 48}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 40], 'color': '#1e2130'},
                        {'range': [40, 70], 'color': '#2a2d3e'},
                        {'range': [70, 100], 'color': '#3a2030'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,17,23,0.5)',
                font_color='white',
                height=350,
                margin=dict(t=50, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_charts:
            # Risk component bar chart
            comp_names = [
                'Category Concentration',
                'Top Segment Dependency',
                'Seasonal Volatility'
            ]
            comp_vals = [
                fragility_components.get('Category Concentration', 0),
                fragility_components.get('Cluster Dependency', 0),
                fragility_components.get('Monthly Volatility', 0)
            ]
            comp_colors = ['#ff4b4b', '#ffd700', '#667eea']

            fig_bar = go.Figure(go.Bar(
                x=comp_vals,
                y=comp_names,
                orientation='h',
                marker_color=comp_colors,
                text=[f"{v:.1f}%" for v in comp_vals],
                textposition='outside',
            ))
            fig_bar.update_layout(
                title="Risk Driver Breakdown",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,17,23,0.5)',
                font_color='white',
                height=170,
                margin=dict(t=40, b=10, l=10, r=60),
                xaxis=dict(range=[0, 110], gridcolor='#2a2d3e'),
                yaxis=dict(gridcolor='#2a2d3e'),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Category revenue donut
            if 'Product Category' in df.columns:
                cat_rev = df.groupby('Product Category')['Total Amount'].sum().reset_index()
                cat_rev.columns = ['Category', 'Revenue']
                cat_rev_colors = [CATEGORY_COLORS.get(c, '#888') for c in cat_rev['Category']]
                fig_donut = go.Figure(go.Pie(
                    labels=cat_rev['Category'],
                    values=cat_rev['Revenue'],
                    hole=0.55,
                    marker_colors=cat_rev_colors,
                    textfont_size=12,
                ))
                fig_donut.update_layout(
                    title="Revenue by Category",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(14,17,23,0.5)',
                    font_color='white',
                    height=175,
                    margin=dict(t=40, b=0, l=0, r=0),
                    legend=dict(font=dict(color='white')),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        # Monthly revenue trend
        if 'Month' in df.columns:
            monthly = df.groupby('Month')['Total Amount'].sum().reset_index()
            monthly['Month Name'] = monthly['Month'].map(MONTH_NAMES)
            fig_trend = go.Figure(go.Scatter(
                x=monthly['Month Name'],
                y=monthly['Total Amount'],
                mode='lines+markers',
                line=dict(color='#00d4aa', width=2.5),
                marker=dict(size=7, color='#00d4aa'),
                fill='tozeroy',
                fillcolor='rgba(0,212,170,0.12)',
            ))
            fig_trend.update_layout(
                title="Monthly Revenue Trend",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,17,23,0.5)',
                font_color='white',
                height=280,
                margin=dict(t=40, b=40, l=60, r=20),
                xaxis=dict(gridcolor='#2a2d3e'),
                yaxis=dict(gridcolor='#2a2d3e', tickprefix='$', tickformat=',.0f'),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ── TAB 2: Live Disaster Simulator ────────────────────────────────────────
    with tab2:
        st.markdown("## 💥 Live Disaster Simulator")
        st.markdown("Adjust the shock parameters to simulate market disruptions and see real-time revenue impact.")

        # Preset buttons
        preset_col1, preset_col2, preset_col3, _ = st.columns([1, 1, 1, 2])
        with preset_col1:
            if st.button("🔥 Mild Shock", use_container_width=True):
                st.session_state['slider_competitor'] = 10
                st.session_state['slider_supply'] = 20
                st.session_state['slider_inflation'] = 5
                st.rerun()
        with preset_col2:
            if st.button("⚡ Major Crisis", use_container_width=True):
                st.session_state['slider_competitor'] = 40
                st.session_state['slider_supply'] = 60
                st.session_state['slider_inflation'] = 20
                st.rerun()
        with preset_col3:
            if st.button("💀 Total Collapse", use_container_width=True):
                st.session_state['slider_competitor'] = 60
                st.session_state['slider_supply'] = 100
                st.session_state['slider_inflation'] = 40
                st.rerun()

        st.divider()

        # Initialize sliders in session state
        if 'slider_competitor' not in st.session_state:
            st.session_state['slider_competitor'] = 20
        if 'slider_supply' not in st.session_state:
            st.session_state['slider_supply'] = 30
        if 'slider_inflation' not in st.session_state:
            st.session_state['slider_inflation'] = 10

        # Sliders
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            competitor_loss = st.slider(
                "🏪 Competitor Attack: Lose % of Price-Sensitive Segment",
                0, 60,
                key='slider_competitor'
            )
        with sim_col2:
            supply_shock_pct = st.slider(
                "📦 Supply Chain Shock: Top Category Stock Loss (%)",
                0, 100,
                key='slider_supply'
            )
        with sim_col3:
            inflation_drop = st.slider(
                "💸 Inflation Shock: Overall Spend Drop (%)",
                0, 40,
                key='slider_inflation'
            )

        # Compute shock
        df = df_clustered
        bargain_cluster_id = [k for k, v in persona_map.items() if 'Bargain' in v or 'Price-Sensitive' in v]
        top_category = df['Product Category'].value_counts().index[0] if 'Product Category' in df.columns else None

        baseline_revenue = df['Total Amount'].sum()

        bargain_revenue = df[df['Cluster'].isin(bargain_cluster_id)]['Total Amount'].sum() if bargain_cluster_id else 0
        competitor_loss_amount = bargain_revenue * (competitor_loss / 100)

        if top_category:
            top_cat_revenue = df[df['Product Category'] == top_category]['Total Amount'].sum()
        else:
            top_cat_revenue = 0
        supply_loss_amount = top_cat_revenue * (supply_shock_pct / 100) * 0.25

        inflation_loss_amount = baseline_revenue * (inflation_drop / 100)

        post_shock_revenue = baseline_revenue - competitor_loss_amount - supply_loss_amount - inflation_loss_amount
        total_loss = baseline_revenue - post_shock_revenue
        survival_rate = (post_shock_revenue / baseline_revenue) * 100 if baseline_revenue > 0 else 0

        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Baseline Revenue", f"${baseline_revenue:,.0f}", "teal")
        with m2:
            kpi_card("Post-Crash Revenue", f"${post_shock_revenue:,.0f}", "gold" if survival_rate > 60 else "red")
        with m3:
            kpi_card("Estimated Loss", f"${total_loss:,.0f}", "red")
        with m4:
            kpi_card("Survival Rate", f"{survival_rate:.1f}%", "teal" if survival_rate > 70 else "gold" if survival_rate > 50 else "red")

        st.divider()

        # Before/after bar chart by cluster
        if 'Cluster' in df.columns:
            cluster_revs = df.groupby('Cluster')['Total Amount'].sum()
            before_vals = []
            after_vals = []
            cluster_labels = []

            for cid in sorted(persona_map.keys()):
                pname = persona_map.get(cid, f"Cluster {cid}")
                cluster_labels.append(pname[:25] + '…' if len(pname) > 25 else pname)
                orig = cluster_revs.get(cid, 0)
                before_vals.append(orig)

                adj = orig
                # Competitor shock hits bargain hunters
                if cid in bargain_cluster_id:
                    adj -= orig * (competitor_loss / 100)
                # Inflation hits everyone
                adj -= orig * (inflation_drop / 100)
                # Supply chain shock hits proportionally based on top category share in cluster
                if top_category:
                    cdf = df[df['Cluster'] == cid]
                    tc_share = cdf[cdf['Product Category'] == top_category]['Total Amount'].sum() / orig if orig > 0 else 0
                    adj -= orig * tc_share * (supply_shock_pct / 100) * 0.25

                after_vals.append(max(adj, 0))

            fig_shock = go.Figure()
            fig_shock.add_trace(go.Bar(
                name='Before Shock',
                x=cluster_labels,
                y=before_vals,
                marker_color='#00d4aa',
            ))
            fig_shock.add_trace(go.Bar(
                name='After Shock',
                x=cluster_labels,
                y=after_vals,
                marker_color='#ff4b4b',
            ))
            fig_shock.update_layout(
                title="Cluster Revenue: Before vs After Shock",
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,17,23,0.5)',
                font_color='white',
                height=380,
                margin=dict(t=50, b=80, l=60, r=20),
                xaxis=dict(gridcolor='#2a2d3e', tickangle=-20),
                yaxis=dict(gridcolor='#2a2d3e', tickprefix='$', tickformat=',.0f'),
                legend=dict(font=dict(color='white')),
            )
            st.plotly_chart(fig_shock, use_container_width=True)

        # Shock breakdown
        st.markdown("### 📊 Shock Breakdown")
        shock_data = pd.DataFrame({
            'Shock Type': ['Competitor Attack', 'Supply Chain Disruption', 'Inflation / Spend Drop'],
            'Revenue Lost ($)': [competitor_loss_amount, supply_loss_amount, inflation_loss_amount],
            '% of Baseline': [
                competitor_loss_amount / baseline_revenue * 100 if baseline_revenue > 0 else 0,
                supply_loss_amount / baseline_revenue * 100 if baseline_revenue > 0 else 0,
                inflation_loss_amount / baseline_revenue * 100 if baseline_revenue > 0 else 0,
            ]
        })
        shock_data['Revenue Lost ($)'] = shock_data['Revenue Lost ($)'].map('${:,.0f}'.format)
        shock_data['% of Baseline'] = shock_data['% of Baseline'].map('{:.1f}%'.format)
        st.dataframe(shock_data, use_container_width=True, hide_index=True)

    # ── TAB 3: Survival Shield & CEO Report ────────────────────────────────────
    with tab3:
        st.markdown("## 🛡️ Survival Shield & CEO Report")

        df = df_clustered
        total_revenue = df['Total Amount'].sum()
        total_txn = len(df)
        aov = total_revenue / total_txn if total_txn > 0 else 0

        cc = fragility_components.get('Category Concentration', 0)
        cd = fragility_components.get('Cluster Dependency', 0)
        mv = fragility_components.get('Monthly Volatility', 0)

        top_cat = df['Product Category'].value_counts().index[0] if 'Product Category' in df.columns else "N/A"
        all_cats = list(df['Product Category'].unique()) if 'Product Category' in df.columns else []
        alt_cats = [c for c in all_cats if c != top_cat]
        alt_cat = alt_cats[0] if alt_cats else "other categories"

        monthly_rev = df.groupby('Month')['Total Amount'].sum()
        peak_month_num = int(monthly_rev.idxmax()) if not monthly_rev.empty else 1
        peak_month_name = MONTH_NAMES.get(peak_month_num, 'N/A')
        min_mrev = monthly_rev.min() if not monthly_rev.empty else 0
        max_mrev = monthly_rev.max() if not monthly_rev.empty else 1
        vol_drop = ((max_mrev - min_mrev) / max_mrev * 100) if max_mrev > 0 else 0

        # Top revenue cluster / persona
        if 'Cluster' in df.columns:
            cluster_rev = df.groupby('Cluster')['Total Amount'].sum()
            top_cid = cluster_rev.idxmax()
            top_persona = persona_map.get(top_cid, f"Cluster {top_cid}")
            top_cust_count = df[df['Cluster'] == top_cid]['Customer ID'].nunique() if 'Customer ID' in df.columns else 0
            top_cust_pct = top_cust_count / df['Customer ID'].nunique() * 100 if 'Customer ID' in df.columns and df['Customer ID'].nunique() > 0 else 0
            top_rev_pct = cluster_rev[top_cid] / total_revenue * 100 if total_revenue > 0 else 0
        else:
            top_persona = "N/A"
            top_cust_pct = 0
            top_rev_pct = 0

        # High-ticket
        ht_cluster = [k for k, v in persona_map.items() if 'High-Ticket' in v]
        if ht_cluster and 'Cluster' in df.columns:
            ht_df = df[df['Cluster'].isin(ht_cluster)]
            ht_customers = ht_df['Customer ID'].nunique() if 'Customer ID' in df.columns else len(ht_df)
            ht_txn = len(ht_df)
            ht_avg_txn = ht_txn / ht_customers if ht_customers > 0 else 0
            ht_avg_val = ht_df['Total Amount'].mean() if not ht_df.empty else 0
            upsell_opp = ht_customers * ht_avg_val * 0.2
        else:
            ht_customers, ht_avg_txn, upsell_opp = 0, 0, 0

        # Survival shield alerts
        if fragility_score >= 70:
            st.error(f"🚨 **CRITICAL ALERT:** Your Business Fragility Score of **{fragility_score}/100** indicates HIGH RISK. Immediate strategic intervention required across all three risk drivers.")
        elif fragility_score >= 40:
            st.warning(f"⚠️ **MODERATE RISK:** Your Business Fragility Score is **{fragility_score}/100**. Take proactive steps to diversify revenue streams before a market shock occurs.")
        else:
            st.success(f"✅ **HEALTHY BUSINESS:** Your Business Fragility Score of **{fragility_score}/100** indicates solid resilience. Maintain current diversification strategy.")

        st.markdown("### 🛡️ Strategic Recommendations")

        if cc > 40:
            st.error(f"**📦 Category Concentration Risk:** `{top_cat}` drives **{cc:.1f}%** of revenue. Diversify into `{alt_cat}` through promotional bundles. Target: bring top-category dependency below 35%.")
        else:
            st.info(f"**📦 Category Health:** `{top_cat}` is your top category at **{cc:.1f}%** of revenue. Healthy spread — continue building `{alt_cat}` to maintain balance.")

        if top_persona != "N/A":
            st.warning(f"**👥 Segment Concentration:** Your `{top_persona}` segment ({top_cust_pct:.1f}% of customers) generates **{top_rev_pct:.1f}%** of revenue. Launch a targeted loyalty programme with VIP early-access events and personalised rewards to cement retention.")

        if mv > 20:
            st.warning(f"**📅 Seasonality Risk:** Peak month is **{peak_month_name}** and revenue drops **{vol_drop:.1f}%** in off-peak months. Introduce subscription boxes, pre-orders, or off-season promotions to smooth monthly cash flow.")
        else:
            st.info(f"**📅 Seasonality:** Revenue volatility is moderate ({mv:.1f}% CV). Peak month **{peak_month_name}** — maintain seasonal campaigns to protect peak-period performance.")

        if ht_customers > 0:
            st.info(f"**💎 High-Ticket Upsell Opportunity:** Your High-Ticket Occasional Buyers (**{ht_customers}** customers) average **{ht_avg_txn:.1f}** purchases per year. A 20% add-on conversion programme could unlock an additional **${upsell_opp:,.0f}** in annual revenue.")

        st.info("**🔗 Supply Chain Resilience:** Identify your two most revenue-critical product categories and qualify at least one backup supplier for each. Maintain a 60–90 day buffer stock during peak seasons.")
        st.info("**💡 Competitive Moat:** Offer an exclusive loyalty programme or price-match guarantee specifically targeting Price-Sensitive Bargain Hunters to reduce vulnerability to competitor undercutting.")

        st.divider()
        st.markdown("### 📥 Download CEO Report")

        try:
            report_md = generate_ceo_report(
                df_clustered, fragility_score, fragility_components,
                persona_map, silhouette_val
            )
            st.download_button(
                label="📥 Download CEO Crash-Test Report (.md)",
                data=report_md,
                file_name=f"CEO_CrashTest_Report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not generate report: {e}")

    # ── TAB 4: ML Transparency ────────────────────────────────────────────────
    with tab4:
        st.markdown("## 🔬 ML Transparency")

        st.info("""
**Algorithm:** K-Means Clustering (unsupervised) with scikit-learn.
Customers are grouped based on purchasing behaviour patterns. The algorithm finds natural clusters by minimising within-cluster variance.
Cluster quality is measured using the **Silhouette Score** (range −1 to +1; higher = more distinct, well-separated clusters).
        """)

        # Features table
        st.markdown("### 📋 Features Used in Clustering")
        features_df = pd.DataFrame({
            'Feature': ['Age', 'Quantity', 'Price per Unit', 'Month', 'DayOfWeek'],
            'Type': ['Demographic', 'Behavioural', 'Behavioural', 'Temporal', 'Temporal'],
            'Description': [
                'Customer age (18–65)',
                'Units purchased per transaction (1–5)',
                'Price paid per item ($)',
                'Month of transaction (1–12, captures seasonality)',
                'Day of week (0=Mon … 6=Sun, captures weekly patterns)'
            ]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

        # Silhouette score
        col_sil, col_k = st.columns(2)
        with col_sil:
            st.metric(
                "Silhouette Score",
                f"{silhouette_val:.3f}",
                help="Range −1 to +1. Scores > 0.5 indicate well-separated clusters."
            )
        with col_k:
            st.metric("Clusters (k)", n_clusters)

        st.divider()
        st.markdown("### 🧑‍🤝‍🧑 Cluster Persona Summary")

        if 'Cluster' in df_clustered.columns:
            summary_rows = []
            for cid, pname in sorted(persona_map.items()):
                cdf = df_clustered[df_clustered['Cluster'] == cid]
                cnt = len(cdf)
                rev = cdf['Total Amount'].sum()
                avg_price = cdf['Price per Unit'].mean() if 'Price per Unit' in cdf.columns else 0
                avg_qty = cdf['Quantity'].mean() if 'Quantity' in cdf.columns else 0
                avg_age = cdf['Age'].mean() if 'Age' in cdf.columns else 0
                summary_rows.append({
                    'Cluster': cid,
                    'Persona': pname,
                    'Transactions': cnt,
                    'Revenue': f"${rev:,.0f}",
                    'Avg Price/Unit': f"${avg_price:.2f}",
                    'Avg Qty': f"{avg_qty:.1f}",
                    'Avg Age': f"{avg_age:.0f}",
                })
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Cluster data not available.")

        st.divider()
        st.warning("""
🚫 **Data Leakage Prevention:** `Total Amount` was **EXCLUDED** from K-Means features because:
> `Total Amount = Quantity × Price per Unit`

Including it would create perfect linear dependence with two existing features, inflating cluster separation artificially and producing misleadingly high silhouette scores.
        """)

        st.markdown("""
### 📋 Dataset Assumptions & Limitations
- Data assumes a single currency and single-country market.
- K-Means assumes spherical clusters — non-spherical distributions may be better served by DBSCAN or Gaussian Mixture Models.
- Persona labels are heuristic (rule-based on centroids), not ground truth.
- Model is retrained fresh on each run; production use should version and persist trained models.
- Missing demographic data (e.g. Gender) does not affect clustering (not used as a feature).
        """)

# ── WELCOME SCREEN (no data loaded) ──────────────────────────────────────────
else:
    st.markdown("# 🚨 Retail Crash-Test Engine")
    st.markdown("### Don't just analyze your sales — stress-test your survival.")
    st.info("👈 **Upload a CSV in the sidebar or click the button below to instantly load sample data.**")

    hero_btn_col, _ = st.columns([1.5, 3])
    with hero_btn_col:
        if st.button("⚡ Quick Start: Load Sample Retail Data", type="primary", use_container_width=True, key="hero_sample_btn"):
            with st.spinner("Loading sample retail data..."):
                st.session_state['df_raw'] = load_sample_data()
                st.session_state['data_source'] = 'sample'
                st.session_state.pop('col_map', None)
                st.session_state.pop('df_clean', None)
                st.rerun()

    st.divider()

    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("""
<div class="kpi-card kpi-card-red">
    <div class="kpi-label">Feature 1</div>
    <div class="kpi-value" style="font-size:1.3rem;">📊 Business Fragility Score</div>
    <div style="color:#9aa0b0;font-size:0.85rem;margin-top:0.5rem;">
    A composite 0–100 risk index built from category concentration, cluster dependency, and monthly revenue volatility.
    Know how resilient your business is before a shock hits.
    </div>
</div>
        """, unsafe_allow_html=True)
    with feat_col2:
        st.markdown("""
<div class="kpi-card kpi-card-gold">
    <div class="kpi-label">Feature 2</div>
    <div class="kpi-value" style="font-size:1.3rem;">💥 Live Disaster Simulator</div>
    <div style="color:#9aa0b0;font-size:0.85rem;margin-top:0.5rem;">
    Simulate competitor attacks, supply-chain shocks, and inflation drops in real time.
    Watch how each scenario impacts your revenue cluster-by-cluster with interactive sliders.
    </div>
</div>
        """, unsafe_allow_html=True)
    with feat_col3:
        st.markdown("""
<div class="kpi-card kpi-card-purple">
    <div class="kpi-label">Feature 3</div>
    <div class="kpi-value" style="font-size:1.3rem;">🛡️ Auto-Generated Survival Shield</div>
    <div style="color:#9aa0b0;font-size:0.85rem;margin-top:0.5rem;">
    Rule-based strategic recommendations drawn directly from your data — no LLM API required.
    Download an executive CEO Crash-Test Report with one click.
    </div>
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 Powered by ML")
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown("""
**Algorithm:** K-Means Clustering (k=3–5)
**Scaling:** StandardScaler
**Evaluation:** Silhouette Score
**Leakage prevention:** Total Amount excluded from features
        """)
    with ml_col2:
        st.markdown("""
**Customer Personas discovered:**
- 🏆 Core VIP Pillars
- 🛒 Price-Sensitive Bargain Hunters
- 💎 High-Ticket Occasional Buyers
- ⚠️ At-Risk / Low-Frequency
        """)
