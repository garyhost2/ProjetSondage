import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from streamlit.components.v1 import html  # ← for rendering raw HTML

# ─────────────────────────────── App config ──────────────────────────────
st.set_page_config(
    page_title="Tunisia Survey Sampling",
    layout="wide",
    page_icon="📊"
)

# ─────────────────────────────── Data loader ─────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# ──────────────────────────────── CSS tweaks ─────────────────────────────
st.markdown("""
<style>
    .main {background-color: #F5F5F5;}
    .stButton>button {border-radius: 5px; transition: all 0.3s ease;}
    .stDownloadButton>button {background-color: #4CAF50; color: white;}
    .stNumberInput, .stSelectbox {max-width: 300px;}
    .highlight {border-left: 5px solid #4CAF50; padding: 0.5em;}
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #f5f5f5; color: #444; text-align: center;
        padding: 15px; border-top: 1px solid #ddd; z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────── Sidebar nav ───────────────────────────────
with st.sidebar:
    st.title("🇹🇳 Sampling App")
    st.markdown("---")
    method = st.radio(
        "Select Method:",
        ["📋 EDA", "🎲 SRSWOR", "📚 Stratified"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload Sampling Frame",
        type="xlsx",
        help="Upload Cadre Tunisie.xlsx file"
    )
    if uploaded_file:
        st.success("File uploaded successfully!")
        st.markdown(f"""
        **File Info:**
        - Name: `{uploaded_file.name}`
        - Size: {uploaded_file.size//1024} KB
        """)

if not uploaded_file:
    st.warning("Please upload a sampling frame file to begin.")
    st.stop()

df = load_data(uploaded_file)

# Column mappings
STRAT_VAR_MAP = {
    'Region': 'Region',
    'Governorate': 'GOVERNORATE',
    'Delegation': 'DELEGATION'
}
AUX_VAR_MAP = {
    'Urban/Rural': 'Area',
    'Block Size': 'pop_block'
}

# ──────────────────────────────── 1. EDA ────────────────────────────────
if method == "📋 EDA":
    st.header("📊 Exploratory Data Analysis", divider="rainbow")

    with st.expander("🔍 Data Overview", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Blocks", len(df))
        c2.metric("Numerical Features",
                  len(df.select_dtypes(include=np.number).columns))
        c3.metric("Categorical Features",
                  len(df.select_dtypes(exclude=np.number).columns))
        st.markdown("#### First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📈 Distribution Analysis"):
        t1, t2 = st.tabs(["Numerical Distributions", "Categorical Distributions"])

        with t1:
            num_col = st.selectbox(
                "Select Numerical Variable",
                df.select_dtypes(include=np.number).columns,
                key="num_dist"
            )
            fig = px.histogram(
                df,
                x=num_col,
                title=f"Distribution of {num_col}",
                color_discrete_sequence=["#4CAF50"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            cat_col = st.selectbox(
                "Select Categorical Variable",
                ['Region', 'GOVERNORATE', 'DELEGATION', 'Area'],
                key="cat_dist"
            )
            fig = px.pie(
                df,
                names=cat_col,
                title=f"Distribution of {cat_col}",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔗 Correlation Analysis"):
        corr = df.select_dtypes(include=np.number).corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Greens'
        ))
        fig.update_layout(title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── 2. Simple Random Sampling ─────────────────────
elif method == "🎲 SRSWOR":
    st.header("🎲 Simple Random Sampling", divider="rainbow")

    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("### Parameters")
            with st.form("srswor_params"):
                n = st.number_input(
                    "Sample Size (n)",
                    min_value=1,
                    max_value=len(df),
                    value=min(500, len(df))
                )
                compare_var = st.selectbox(
                    "Comparison Variable",
                    ['Region', 'GOVERNORATE', 'DELEGATION', 'Area']
                )
                submitted = st.form_submit_button("Generate Sample")

        if submitted:
            with st.spinner("Sampling in progress..."):
                sample = df.sample(n=n, random_state=42)

            with col2:
                st.markdown("### Sample Preview")
                st.dataframe(
                    sample.head(10),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("---")
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "📥 Download Full Sample (CSV)",
                        data=sample.to_csv(index=False).encode('utf-8'),
                        file_name="srs_sample.csv",
                        mime="text/csv"
                    )
                with dl2:
                    st.download_button(
                        "📊 Download Summary Stats (CSV)",
                        data=sample.describe().to_csv().encode('utf-8'),
                        file_name="srs_stats.csv",
                        mime="text/csv"
                    )

            st.markdown("---")
            st.markdown("### 🆚 Population vs Sample Comparison")
            pop_dist = df[compare_var].value_counts(normalize=True).reset_index()
            sample_dist = sample[compare_var].value_counts(normalize=True).reset_index()

            v1, v2 = st.columns(2)
            with v1:
                st.markdown("##### Distribution Comparison")
                fig = px.bar(
                    pd.concat([
                        pop_dist.assign(Source="Population"),
                        sample_dist.assign(Source="Sample")
                    ]),
                    x=compare_var,
                    y="proportion",
                    color="Source",
                    barmode="group",
                    color_discrete_sequence=["#4CAF50", "#FF5722"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.markdown("##### Statistical Summary")
                styled_stats = (
                    sample.describe().T
                          .style
                          .background_gradient(cmap=plt.cm.Greens)
                )
                # render the styled HTML
                html(styled_stats.to_html(), height=350)

# ───────────────────────── 3. Stratified Sampling ────────────────────────
else:
    st.header("📚 Stratified Sampling", divider="rainbow")

    with st.form("stratified_params"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Sampling Parameters")
            n = st.number_input(
                "Total Sample Size (n)",
                min_value=1,
                max_value=len(df),
                value=min(500, len(df))
            )
            strat_display = st.selectbox(
                "Stratification Variable",
                list(STRAT_VAR_MAP.keys())
            )
        with c2:
            st.markdown("### Analysis Parameters")
            aux_display = st.selectbox(
                "Auxiliary Variable",
                list(AUX_VAR_MAP.keys())
            )
            st.markdown("#### Expected Columns:")
            st.code(
                f"Stratification: {STRAT_VAR_MAP[strat_display]}\n"
                f"Auxiliary:      {AUX_VAR_MAP[aux_display]}",
                language="text"
            )
        run_strat = st.form_submit_button("Run Stratified Sampling")

    if run_strat:
        with st.spinner("Performing stratified sampling..."):
            strat_var = STRAT_VAR_MAP[strat_display]
            aux_var = AUX_VAR_MAP[aux_display]

            strata = df.groupby(strat_var).size().reset_index(name='Nh')
            strata['nh'] = np.round(n * strata['Nh'] / strata['Nh'].sum()).astype(int)

            diff = n - strata['nh'].sum()
            if diff != 0:
                strata.iloc[-1, -1] += diff  # fix rounding

            samples = [
                df[df[strat_var] == row[strat_var]]
                  .sample(n=int(row['nh']), random_state=42)
                for _, row in strata.iterrows()
            ]
            final_sample = pd.concat(samples)

        st.success("Sampling completed successfully!")

        with st.expander("📋 Allocation Table", expanded=True):
            styled_alloc = (
                strata
                .style
                .background_gradient(subset=['Nh', 'nh'], cmap=plt.cm.Greens)
            )
            html(styled_alloc.to_html(), height=300)
            st.download_button(
                "📥 Download Allocation Table",
                data=strata.to_csv(index=False).encode('utf-8'),
                file_name="allocation.csv",
                mime="text/csv"
            )

        with st.expander("🔍 Sample Analysis", expanded=True):
            t1, t2 = st.tabs(["Sample Preview", "Auxiliary Analysis"])

            with t1:
                st.dataframe(
                    final_sample.head(10),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "📥 Download Full Sample",
                    data=final_sample.to_csv(index=False).encode('utf-8'),
                    file_name="stratified_sample.csv",
                    mime="text/csv"
                )

            with t2:
                a1, a2 = st.columns(2)

                with a1:
                    st.markdown(f"##### {aux_display} Distribution")
                    fig = px.pie(
                        final_sample,
                        names=aux_var,
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with a2:
                    st.markdown("##### Descriptive Statistics")
                    desc_df = final_sample[aux_var].describe().to_frame().T
                    if pd.api.types.is_numeric_dtype(final_sample[aux_var]):
                        desc_df = desc_df.round(2)

                    styled_desc = desc_df.style.background_gradient(cmap=plt.cm.Greens)
                    html(styled_desc.to_html(), height=200)

# ─────────────────────────────── footer ──────────────────────────────────
st.markdown("""
<div class="footer">
    Done by Mohamed Yassine Ben Zekri
</div>
""", unsafe_allow_html=True)
