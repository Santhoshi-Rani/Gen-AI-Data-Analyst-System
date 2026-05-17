import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import skew, zscore
from scipy.stats import gaussian_kde
from statistics import mode as get_mode
import warnings
warnings.filterwarnings('ignore')

# ---------- Page config (must be first) ----------
st.set_page_config(
    page_title="Gen AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS: Black background + Times New Roman + Dim Blue ----------
st.markdown("""
<style>
    *{
        font-family: 'Times New Roman', Times, serif !important;
    }
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif;
        background-color: #000000;
        color: #c0d4f0;
    }
    .stApp {
        background-color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0f1a;
        font-family: 'Times New Roman', Times, serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #7cb9e8;
        font-family: 'Times New Roman', Times, serif;
        font-weight: 600;
    }
    .stMarkdown h1 {
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 0.3rem;
    }
    [data-testid="stMetricValue"] {
        color: #7cb9e8;
        font-size: 2rem;
    }
    .stButton button {
        background-color: #1e3a5f;
        color: #d0e4ff;
        border-radius: 8px;
        border: none;
        font-family: 'Times New Roman', Times, serif;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #2c5282;
        color: white;
        transform: scale(1.02);
    }
    .dataframe {
        background-color: #0a0f1a !important;
        color: #c0d4f0 !important;
        font-family: 'Times New Roman', Times, serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0a0f1a;
        color: #b8d0ff;
        border-radius: 8px;
        padding: 8px 20px;
        font-family: 'Times New Roman', Times, serif;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7a6a2f;
        color: white;
    }
    .stAlert {
        background-color: #1a2a3a;
        color: #c0d4f0;
    }
    .streamlit-expanderHeader {
        background-color: #0a0f1a;
        color: #7cb9e8;
        font-family: 'Times New Roman', Times, serif;
    }
    .stSelectbox div, .stMultiSelect div, .stTextArea div {
        background-color: #0a0f1a;
        color: #c0d4f0;
    }
    .pipeline-title {
        color: #7cb342 !important;
        font-size: 42px !important;
        font-weight: 700 !important;
        margin: 0 0 8px 0 !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .refine-title, .outlier-title, .select-title {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 0 0 6px 0 !important;
        font-family: 'Times New Roman', Times, serif !important;
    }  
    [data-baseweb="tag"] {
        background-color: #2e4d66 !important;
        color: #e8f2ff !important;
        border: 1px solid #5f7f99 !important;
    }
    [data-baseweb="tag"] span {
        color: #e8f2ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
@st.cache_data(ttl=3600)
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    return df

def auto_detect_datetime(df):
    """
    Advanced automatic datetime detection and conversion.
    Handles:
    - mixed date formats
    - stock market dates
    - slash/dash formats
    - invalid parsing safely
    """
    datetime_report = []

    date_keywords = [
        'date', 'time', 'timestamp',
        'year', 'month', 'day'
    ]

    for col in df.columns:
        original_dtype = df[col].dtype

        # check object columns OR columns with date keywords
        if(original_dtype == 'object' or any(keyword in col.lower() for keyword in date_keywords)):

            # Clean values
            cleaned_col = (df[col].astype(str).str.strip().replace(['nan', 'None', ''], np.nan))

            # ---------- TRY MULTIPLE DATE PARSING STRATEGIES ----------

            converted_col = pd.to_datetime(
                cleaned_col,
                errors='coerce',
                dayfirst=True
            )


            success_ratio = converted_col.notnull().mean()
            failed_count = converted_col.isnull().sum()

            # ---------- SECOND ATTEMPT ----------
            if success_ratio < 0.80:

                converted_col_alt = pd.to_datetime(
                    cleaned_col,
                    errors='coerce',
                    dayfirst=False
                )

                alt_ratio = converted_col_alt.notnull().mean()

                # Use better conversion
                if alt_ratio > success_ratio:
                    converted_col = converted_col_alt
                    success_ratio = alt_ratio
                    failed_count = converted_col.isnull().sum()

            # ---------- FINAL DECISION ----------

            if success_ratio >= 0.70:

                df[col] = converted_col

                detection_status = "Converted to Datetime"

            else:

                detection_status = "Kept as Object"

            datetime_report.append({
                "Column": col,
                "Conversion Success %": round(success_ratio * 100, 2),
                "Failed Rows": failed_count,
                "Status": detection_status
            })

    report_df = pd.DataFrame(datetime_report)

    return df, report_df





def detect_target_and_problem(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col.lower() in ['target', 'y', 'label', 'class']:
            if col in numeric_cols:
                if df[col].nunique() <= 10:
                    return col, 'classification'
                else:
                    return col, 'regression'
            else:
                return col, 'classification'
    priority_targets = ['target', 'label', 'class', 'price', 'close', 'adj close', 'sales', 'revenue', 'profit', 'churn']
    for col in df.columns:
        if col.lower() in priority_targets:
            if (df[col].dtype == 'object' or df[col].nunique() <= 10):
                return col, 'classification'
            else:
                return col, 'regression'
    last_col = df.columns[-1]
    if last_col in numeric_cols:
        if df[last_col].dtype == 'object' or df[last_col].nunique() <= min(10, len(df)*0.05):
            return last_col, 'classification'
        else:
            return last_col, 'regression'
    else:
        return last_col, 'classification'

def show_data_overview(df):
    st.subheader("📌 Basic Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", df.isnull().sum().sum())
    with st.expander("🔍 Preview (first 5 rows)"):
        st.dataframe(df.head(20), use_container_width=True)
    with st.expander("📐 Column Data Types"):
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Column", "Data Type"]
        st.dataframe(dtypes_df)
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(include='all').T)

def show_unique_values(df):
    st.subheader("🔍 Unique Values Analysis")
    unique_data = []
    for col in df.columns:
        unique_count = len(df[col].unique())
        unique_data.append({"Column": col, "Unique Values Count": unique_count})
    unique_df = pd.DataFrame(unique_data)
    st.dataframe(unique_df, use_container_width=True)
    with st.expander("📋 View Unique Values for a Column"):
        selected_col = st.selectbox("Select column", df.columns, key="unique_values_selectbox")
        unique_values = pd.Series(df[selected_col].unique())
        display_values = unique_values.astype(str)
        display_values = display_values.replace(["nan", "NaT", "None"], "NaN")
        st.write(f"### Unique values in '{selected_col}'")
        st.write(display_values.tolist())

def show_missing_analysis(df):
    st.subheader("❓ Missing Value Analysis")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count,
        "Missing Percentage": missing_pct
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Percentage", ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True)
        fig = px.bar(missing_df, x="Column", y="Missing Percentage",
                     title="Missing Percentage per Column",
                     color_discrete_sequence=["#7cb9e8"],
                     text="Missing Percentage")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_color="#4a90d9")
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="missing_bar")
        st.subheader("Missing Pattern Heatmap")
        missing_matrix = df.isnull().astype(int)
        fig_miss = px.imshow(missing_matrix.T, aspect="auto",
                             color_continuous_scale=["#0a0f1a", "#7cb9e8"],
                             title="Light Blue = Missing")
        fig_miss.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig_miss, use_container_width=True, key="missing_heatmap")
        st.info("💡 Use visualisations below to decide dropping vs imputation.")
    else:
        st.success("✅ No missing values found!")

def numeric_analysis(df, numeric_cols):
    if not numeric_cols:
        st.warning("No numeric columns.")
        return
    selected = st.multiselect("Select numeric columns for detailed histograms", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="numeric_histogram_multiselect")
    for idx, col in enumerate(selected):
        data = df[col].dropna()
        if len(data) == 0:
            continue
        mean_val = data.mean()
        median_val = data.median()
        try:
            mode_val = get_mode(data)
        except:
            mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
        x_range = []
        kde_scaled = []
        if len(data) > 1 and data.std() != 0:
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                kde_vals = kde(x_range)
                hist_counts, _ = np.histogram(data, bins=30)
                max_hist = hist_counts.max()
                kde_scaled = (kde_vals / kde_vals.max()) * max_hist
            except:
                pass
        counts, bin_edges = np.histogram(data, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = np.diff(bin_edges)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=counts, width=bar_width * 0.9, name='Data', opacity=0.85,
                             marker=dict(color='#4a90d9', line=dict(color='white', width=1))))
        fig.add_trace(go.Scatter(x=x_range, y=kde_scaled, mode='lines',
                                 name='Density Curve', line=dict(color='#ffb347', width=3)))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#00ffcc", annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#ff66cc", annotation_text=f"Median: {median_val:.2f}")
        fig.add_vline(x=mode_val, line_dash="dash", line_color="#ffff00", annotation_text=f"Mode: {mode_val:.2f}")
        fig.update_layout(barmode='overlay', bargap=0.08, bargroupgap=0.02,
                          title=f"Distribution of {col} (with KDE & statistics)",
                          xaxis_title=col, yaxis_title="Frequency",
                          plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"),
                          legend=dict(font=dict(color="#c0d4f0")))
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{idx}")

def boxplot_analysis(df, numeric_cols, key_suffix=""):
    st.subheader("📦 Box Plot Analysis (Outlier Visualization)")
    if not numeric_cols:
        st.warning("No numeric columns available.")
        return
    selected_cols = st.multiselect(
        "Select columns for box plot analysis",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))],
        key=f"boxplot_multiselect_{key_suffix}"
    )
    for idx, col in enumerate(selected_cols):
        fig = px.box(df, y=col, title=f"Box Plot - {col}", color_discrete_sequence=["#7cb9e8"])
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig, use_container_width=True, key=f"box_{col}_{key_suffix}_{idx}")

def categorical_analysis(df, categorical_cols, target_col=None, problem_type=None):
    if not categorical_cols:
        st.warning("No categorical columns.")
        return
    selected = st.multiselect("Select categorical columns for count plots", categorical_cols,
                              default=categorical_cols[:min(3, len(categorical_cols))],
                              key="categorical_multiselect")
    for idx, col in enumerate(selected):
        freq = df[col].value_counts().reset_index()
        freq.columns = [col, 'count']
        fig = px.bar(freq, x=col, y='count', title=f"Count Plot: {col}",
                     text='count', color=col, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig, use_container_width=True, key=f"cat_bar_{col}_{idx}")
        if problem_type == 'classification' and target_col and target_col != col and target_col in df.columns:
            st.subheader(f"📊 {col} grouped by Target: {target_col}")
            grouped = df.groupby([target_col, col]).size().reset_index(name='count')
            fig2 = px.bar(grouped, x=col, y='count', color=target_col, text='count',
                          title=f"{col} distribution by {target_col}",
                          barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_traces(textposition='outside')
            fig2.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                               font=dict(family="Times New Roman", color="#c0d4f0"))
            st.plotly_chart(fig2, use_container_width=True, key=f"cat_group_{col}_{idx}")

def correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].dropna(axis=1, how='all').corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        title="Correlation Matrix (Numeric Columns)")
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
    else:
        st.warning("Need at least 2 numeric columns for correlation.")

def multicollinearity_summary(df, numeric_cols):
    st.subheader("📌 Multicollinearity Analysis Summary")
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns.")
        return
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > 0.80:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr.append({
                    "Column 1": col1,
                    "Column 2": col2,
                    "Correlation": round(corr_value, 2),
                    "Issue": "Multicollinearity Detected",
                    "Recommended Solution": "Consider dropping one column or use PCA/VIF."
                })
    if high_corr:
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    else:
        st.success("✅ No significant multicollinearity detected.")

def time_series_analysis(df, datetime_cols, numeric_cols):
    if not datetime_cols:
        st.info("No datetime columns available for time series analysis.")
        return
    st.subheader("📈 Time Series Analysis")
    selected_time = st.selectbox("Select datetime column", datetime_cols)
    if numeric_cols:
        selected_metrics = st.multiselect("Select numeric columns to plot over time", numeric_cols, default=numeric_cols[:2])
        if selected_metrics:
            df_sorted = df.sort_values(selected_time)
            for idx, metric in enumerate(selected_metrics):
                fig = px.line(df_sorted, x=selected_time, y=metric, title=f"{metric} over time",
                              color_discrete_sequence=["#7cb9e8"])
                fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
                st.plotly_chart(fig, use_container_width=True, key=f"ts_line_{metric}_{idx}")
                window = st.slider(f"Rolling window for {metric}", min_value=2, max_value=min(30, len(df_sorted)), value=7, key=f"roll_{metric}_{idx}")
                df_sorted[f'{metric}_rolling'] = df_sorted[metric].rolling(window=window).mean()
                fig_roll = px.line(df_sorted, x=selected_time, y=[metric, f'{metric}_rolling'],
                                   title=f"{metric} with {window}-period rolling average",
                                   color_discrete_sequence=["#7cb9e8", "#ffb347"])
                fig_roll.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
                st.plotly_chart(fig_roll, use_container_width=True, key=f"ts_roll_{metric}_{idx}")

def scatter_plot_matrix(df, numeric_cols):
    if len(numeric_cols) >= 2:
        st.subheader("🔍 Scatter Plot Matrix (Pair Plot)")
        selected_numeric = st.multiselect("Select up to 5 numeric columns for pair plot", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        if len(selected_numeric) >= 2:
            fig = px.scatter_matrix(df, dimensions=selected_numeric, title="Pair Plot",
                                    color_discrete_sequence=["#7cb9e8"])
            fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
            st.plotly_chart(fig, use_container_width=True, key="scatter_matrix")
    else:
        st.warning("Need at least 2 numeric columns for scatter plot matrix.")

def advanced_scatter_plot(df, numeric_cols, categorical_cols=None):
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot.")
        return
    st.subheader("🎯 Interactive Scatter Plot")
    col_x = st.selectbox("X-axis", numeric_cols, index=0)
    col_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    color_col = st.selectbox("Color by (optional)", ["None"] + (categorical_cols if categorical_cols else []))
    plot_type = st.radio("Plot type", ["2D Scatter", "3D Scatter"], horizontal=True)
    if plot_type == "2D Scatter":
        if color_col != "None":
            fig = px.scatter(df, x=col_x, y=col_y, color=color_col, title=f"{col_y} vs {col_x}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}",
                             color_discrete_sequence=["#7cb9e8"])
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="scatter_2d")
    else:
        if len(numeric_cols) >= 3:
            col_z = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
            if color_col != "None":
                fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z, color=color_col,
                                    title=f"3D Scatter: {col_x}, {col_y}, {col_z}")
            else:
                fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z, title=f"3D Scatter: {col_x}, {col_y}, {col_z}",
                                    color_discrete_sequence=["#7cb9e8"])
            fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
            st.plotly_chart(fig, use_container_width=True, key="scatter_3d")
        else:
            st.warning("Need at least 3 numeric columns for 3D scatter plot.")

def intelligent_outlier_treatment(df):
    st.markdown("""
    <h3 class='outlier-title'>
    🚨 Intelligent Outlier Detection & Treatment
    </h3>
    """, unsafe_allow_html=True)
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    skip_keywords = ['id', 'zip', 'zipcode', 'postal', 'latitude', 'longitude']
    valid_cols = []
    skipped_cols = []
    for col in numeric_cols:
        if any(word in col.lower() for word in skip_keywords):
            skipped_cols.append(col)
        else:
            valid_cols.append(col)
    st.markdown("""
    <h3 class='select-title'>
    🎯 Select Columns for Outlier Treatment
    </h3>
    """, unsafe_allow_html=True)
    selected_cols = st.multiselect("Choose columns", valid_cols, default=valid_cols, key="outlier_treatment_multiselect")
    outlier_results = []
    for idx, col in enumerate(selected_cols):
        data = df_processed[col].dropna()
        st.markdown(f"### 📦 Before Treatment: {col}")
        fig_before = px.box(df_processed, y=col, title=f"{col} - Before Outlier Treatment", color_discrete_sequence=["#ff6b6b"])
        fig_before.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig_before, use_container_width=True, key=f"outlier_before_{col}_{idx}")
        if len(data) == 0:
            continue
        skewness = data.skew()
        unique_ratio = data.nunique() / len(data)
        is_count_variable = (pd.api.types.is_integer_dtype(data) and unique_ratio < 0.05)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (data < lower) | (data > upper)
        outlier_count = outliers.sum()
        outlier_pct = round((outlier_count / len(data)) * 100, 2)
        if outlier_count == 0:
            treatment = "No Treatment Needed"
            reason = "No significant outliers detected."
        elif is_count_variable:
            bins = 5
            df_processed[col] = pd.cut(df_processed[col], bins=bins, labels=False)
            treatment = "Binning"
            reason = "Count variable detected. Binning used to reduce skewness and extreme values."
        elif abs(skewness) > 1:
            shift_value = abs(df_processed[col].min()) + 1
            df_processed[col] = np.log1p(df_processed[col] + shift_value)
            treatment = "Log Transformation"
            reason = "Highly skewed distribution detected. Log transformation applied."
        else:
            df_processed = df_processed[~((df_processed[col] < lower) | (df_processed[col] > upper))]
            treatment = "Outliers Removed"
            reason = "Moderate outliers detected. IQR-based removal applied."
        st.markdown(f"### ✅ After Treatment: {col}")
        fig_after = px.box(df_processed, y=col, title=f"{col} - After Outlier Treatment", color_discrete_sequence=["#7cb9e8"])
        fig_after.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig_after, use_container_width=True, key=f"outlier_after_{col}_{idx}")
        problem_detected = "High Skewness + Extreme Outliers" if abs(skewness) > 1 else "Moderate Outliers"
        outlier_results.append({
            "Column": col,
            "Problem Detected": problem_detected,
            "Outlier Count": outlier_count,
            "Outlier %": outlier_pct,
            "Treatment": treatment,
            "Reason": reason
        })
    outlier_df = pd.DataFrame(outlier_results)
    st.subheader("📋 Outlier Detection Summary")
    st.dataframe(outlier_df, use_container_width=True)
    if skipped_cols:
        st.info(f"Skipped identifier/geospatial columns: {', '.join(skipped_cols)}")
    return df_processed, outlier_df

def intelligent_preprocessing(df):
    st.markdown("""
    <h3 class='refine-title'>
    📑 Automated Data Refinement System
    </h3>
    """, unsafe_allow_html=True)
    df_processed = df.copy()
    recommendation_data = []
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    for col in df_processed.columns:
        missing_pct = (df_processed[col].isnull().sum() / len(df_processed)) * 100
        method_used = ""
        reason = ""
        if missing_pct > 60 and not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
            df_processed.drop(columns=[col], inplace=True)
            method_used = "Drop Column"
            reason = "Column has more than 60% missing values." 

        # ---------- Advanced Datetime Missing Value Handling ----------
        elif pd.api.types.is_datetime64_any_dtype(df_processed[col]):
            # Store original missing count
            original_missing = df_processed[col].isnull().sum()

            # Sort values before interpolation
            temp_sorted = df_processed.sort_values(by=col).copy()

            # CASE 1 → Small missing percentage
            if missing_pct <= 10:

                 # Time interpolation
                 df_processed[col] = df_processed[col].interpolate(method='linear')

                 # Forward fill + backward fill safety
                 df_processed[col] = df_processed[col].fillna(method='ffill')
                 df_processed[col] = df_processed[col].fillna(method='bfill')

                 method_used = "Time Interpolation + Forward/Backward Fill"

                 reason = (
                    "Small datetime gaps detected. "
                    "Applied interpolation and sequential fill.")

            # CASE 2 → Medium missing percentage
            elif missing_pct <= 40:

                # Forward fill first
                df_processed[col] = df_processed[col].fillna(method='ffill')

                # Backward fill remaining
                df_processed[col] = df_processed[col].fillna(method='bfill')
                method_used = "Forward Fill + Backward Fill"
                reason = (
                    "Moderate datetime gaps detected. "
                    "Sequential datetime filling applied.")

            # CASE 3 → Very high missing percentage
            else:

                # Check if column is critical
                important_date_keywords = [
                    'date', 'time', 'timestamp',
                    'created', 'updated', 'year',
                    'month', 'day']

                if any(keyword in col.lower() for keyword in important_date_keywords):
                    df_processed[col] = df_processed[col].fillna(method='ffill')
                    df_processed[col] = df_processed[col].fillna(method='bfill')

                    method_used = "Retained Important Datetime Column"

                    reason = (
                        "Datetime column considered important. "
                        "Retained and filled sequentially.")

                else:
                    df_processed.drop(columns=[col], inplace=True)

                    method_used = "Dropped Datetime Column"

                    reason = (
                        "High missing percentage and column "
                        "not considered critical.")

                    




        elif col in numeric_cols:
            skewness = df_processed[col].skew()
            if abs(skewness) < 0.5:
                fill_value = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(fill_value)
                method_used = "Mean Imputation"
                reason = "Normally distributed numeric data."
            else:
                fill_value = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(fill_value)
                method_used = "Median Imputation"
                reason = "Skewed numeric data with outliers."
        elif col in categorical_cols:
            mode_series = df_processed[col].mode()
            mode_value = mode_series[0] if not mode_series.empty else "Unknown"
            df_processed[col] = df_processed[col].fillna(mode_value)
            method_used = "Mode Imputation"
            reason = "Categorical/string column."
        if missing_pct > 0:
            recommendation_data.append({
                "Column": col,
                "Missing %": round(missing_pct, 2),
                "Method Used": method_used,
                "Reason": reason
            })
    imputation_confirmation = []
    for col in df_processed.columns:
        current_missing = df_processed[col].isnull().sum()
        if current_missing == 0:
            imputation_confirmation.append({"Column": col, "Missing Values After Imputation": current_missing})
    imputation_df = pd.DataFrame(imputation_confirmation)
    skewness_data = []
    for col in numeric_cols:
        if col in df_processed.columns:
            skew_val = df_processed[col].skew()
            if abs(skew_val) < 0.5:
                interpretation = "Approximately Symmetric"
            elif abs(skew_val) < 1:
                interpretation = "Moderately Skewed"
            else:
                interpretation = "Highly Skewed"
            skewness_data.append({
                "Column": col,
                "Skewness": round(skew_val, 2),
                "Interpretation": interpretation
            })
    df_processed, outlier_df = intelligent_outlier_treatment(df_processed)
    recommendations_df = pd.DataFrame(recommendation_data)
    skewness_df = pd.DataFrame(skewness_data)
    return (df_processed, recommendations_df, imputation_df, skewness_df, outlier_df)

def clean_data_sidebar(df):
    st.sidebar.header("🧹 Data Cleaning")
    with st.sidebar.expander("🗑️ Drop Columns"):
        cols_to_drop = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            df.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df = df
            st.rerun()
    with st.sidebar.expander("🔄 Change Data Type"):
        col = st.selectbox("Column", df.columns)
        new_type = st.selectbox("New type", ["int", "float", "str", "datetime", "category", "bool"])
        if st.button("Convert Type"):
            try:
                if new_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif new_type == "int":
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif new_type == "float":
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                elif new_type == "str":
                    df[col] = df[col].astype(str)
                elif new_type == "category":
                    df[col] = df[col].astype("category")
                elif new_type == "bool":
                    df[col] = df[col].astype(bool)
                st.session_state.df = df
                st.sidebar.success(f"Converted {col} to {new_type}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))
    with st.sidebar.expander("👥 Duplicate Rows"):
        dup_count = df.duplicated().sum()
        st.write(f"**Duplicate rows:** {dup_count}")
        if dup_count > 0 and st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.sidebar.success("Duplicates removed!")
            st.rerun()
    return df

# ---------- Main App ----------
st.title("✨ Gen AI Powered Data Analyst System")
st.markdown("---")

with st.sidebar:
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    # Auto-convert datetime columns (improved)
    df, datetime_report = auto_detect_datetime(df)
    if uploaded_file.name != st.session_state.get("uploaded_name", ""):
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.df = df.copy()
    st.session_state.df = clean_data_sidebar(st.session_state.df)
    df_clean = st.session_state.df

    suggested_target, auto_problem = detect_target_and_problem(df_clean)
    st.sidebar.subheader("🎯 Target Column Setup")
    target_option = st.sidebar.radio("Target selection", ["Auto-detect", "Choose manually"])
    if target_option == "Auto-detect":
        target_col = suggested_target
        problem_type = auto_problem
    else:
        target_col = st.sidebar.selectbox("Select target column", df_clean.columns)
        if df_clean[target_col].dtype in ['object', 'category', 'bool'] or df_clean[target_col].nunique() <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
    st.sidebar.info(f"**Detected Problem:** {problem_type.upper()} | **Target:** {target_col}")

    tab_overview, tab_unique, tab_missing, tab_viz, tab_preprocess, tab_problem = st.tabs(
        ["📄 Data Overview", "🔍 Unique Values", "❓ Missing Values", "📊 Visualizations", "🧠 Intelligent Preprocessing", "📝 Problem Statement"]
    )

    with tab_overview:
        show_data_overview(df_clean)
    with tab_unique:
        show_unique_values(df_clean)
    with tab_missing:
        show_missing_analysis(df_clean)
    with tab_viz:
        st.header("Exploratory Data Analysis")
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns.tolist()
        st.subheader("🕒 Datetime Detection Report")
        st.dataframe(datetime_report, use_container_width=True)
        st.subheader("📈 Numeric Columns Analysis")
        numeric_analysis(df_clean, numeric_cols)
        boxplot_analysis(df_clean, numeric_cols, key_suffix="original")
        st.subheader("📊 Categorical Columns Analysis")
        categorical_analysis(df_clean, categorical_cols, target_col, problem_type)
        if datetime_cols:
            time_series_analysis(df_clean, datetime_cols, numeric_cols)
        st.subheader("📉 Advanced Scatter Plots")
        advanced_scatter_plot(df_clean, numeric_cols, categorical_cols)
        scatter_plot_matrix(df_clean, numeric_cols)
        st.subheader("🔗 Correlation & Multicollinearity")
        correlation_heatmap(df_clean, numeric_cols)
        multicollinearity_summary(df_clean, numeric_cols)
        st.markdown("---")
        st.caption("ℹ️ Use the side panel to clean data further. Next: Missing imputation, AutoML, and RAG+LLM.")
    with tab_preprocess:
        st.markdown("""
        <h1 class='pipeline-title'>
        ⚙️ Advanced Data Processing Pipeline
        </h1>
        """, unsafe_allow_html=True)
        if st.button("Run Intelligent Preprocessing"):
            with st.spinner("Processing dataset intelligently..."):
                (processed_df, recommendations_df, imputation_df, skewness_df, outlier_df) = intelligent_preprocessing(df_clean)
                st.session_state.df = processed_df
                st.subheader("📌 Missing Value Recommendations")
                st.dataframe(recommendations_df, use_container_width=True)
                st.subheader("✅ Missing Value Imputation Status")
                st.dataframe(imputation_df, use_container_width=True)
                processed_numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
                boxplot_analysis(processed_df, processed_numeric_cols, key_suffix="processed")
                st.subheader("📈 Distribution & Skewness Analysis")
                st.dataframe(skewness_df, use_container_width=True)
                st.subheader("🚨 Outlier Detection & Treatment")
                st.dataframe(outlier_df, use_container_width=True)
                st.success("✅ Intelligent preprocessing completed successfully.")
    with tab_problem:
        st.subheader("Business Problem Statement")
        problem_text = st.text_area(
            "Paste or type the problem description here:",
            height=150,
            placeholder="Example: Predict customer churn based on usage patterns...",
            help="This will be used later for RAG and LLM analysis."
        )
        if problem_text:
            st.success("Problem statement saved.")
            st.session_state.problem_statement = problem_text
        else:
            st.info("Please enter your problem statement.")
else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    st.markdown("""
    **Features ready:**
    - 📁 CSV upload (auto‑encoding)
    - 🕒 **Auto‑detection of datetime columns** (improved with name hints)
    - 🧹 Clean data: drop columns, change types, remove duplicates
    - 🎯 Auto‑detects target column and problem type
    - 💬 Problem statement input for RAG/LLM
    - 🔍 Unique values analysis
    - 📊 **Advanced visualisations** with unique keys (no more duplicate ID errors)
    """)