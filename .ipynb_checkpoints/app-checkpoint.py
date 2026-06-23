import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    km        = pickle.load(open("models/kmeans_model.pkl", "rb"))
    scaler    = pickle.load(open("models/scaler.pkl",       "rb"))
    label_map = pickle.load(open("models/label_map.pkl",    "rb"))
    sim_df    = pd.read_csv("models/product_similarity.csv", index_col=0)
    rfm       = pd.read_csv("models/rfm_clustered.csv")
    return km, scaler, label_map, sim_df, rfm

km, scaler, label_map, sim_df, rfm = load_models()

SEGMENT_COLORS = {
    "High-Value": "#2a9d8f",
    "Regular":    "#457b9d",
    "Occasional": "#e9c46a",
    "At-Risk":    "#e63946",
}
SEGMENT_ICONS = {
    "High-Value": "💎",
    "Regular":    "🛍️",
    "Occasional": "🔄",
    "At-Risk":    "⚠️",
}
SEGMENT_DESC = {
    "High-Value": "Recent, frequent, and high-spending customer. Reward with loyalty perks.",
    "Regular":    "Steady purchaser with moderate spend. Upsell with targeted offers.",
    "Occasional": "Infrequent buyer. Re-engage with promotions or new arrivals.",
    "At-Risk":    "Hasn't purchased in a long time. Immediate win-back campaign needed.",
}

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=60)
st.sidebar.title("Shopper Spectrum")
st.sidebar.caption("E-Commerce Customer Intelligence")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Clustering", "🎯 Recommendation"])

# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendation")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(rfm):,}")
    col2.metric("High-Value", f"{(rfm['Segment']=='High-Value').sum():,}")
    col3.metric("At-Risk",    f"{(rfm['Segment']=='At-Risk').sum():,}")
    col4.metric("Products in Recommender", f"{len(sim_df):,}")

    st.markdown("---")
    st.markdown("### 📌 Segment Overview")

    cols = st.columns(4)
    for col, (seg, color) in zip(cols, SEGMENT_COLORS.items()):
        count = (rfm["Segment"] == seg).sum()
        pct   = count / len(rfm) * 100
        col.markdown(
            f"""
            <div style='background:{color}22; border-left:5px solid {color};
                        padding:14px; border-radius:8px;'>
                <h3 style='color:{color}; margin:0'>{SEGMENT_ICONS[seg]} {seg}</h3>
                <h2 style='margin:4px 0'>{count} <span style='font-size:14px'>({pct:.1f}%)</span></h2>
                <p style='font-size:12px; color:#555; margin:0'>{SEGMENT_DESC[seg]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🗺️ How to Use")
    st.info("📊 **Clustering tab** — Enter a customer's RFM values to predict their segment.\n\n"
            "🎯 **Recommendation tab** — Enter a product name to get 5 similar product recommendations.")

# ══════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════
elif page == "📊 Clustering":
    st.title("📊 Customer Segmentation")
    st.markdown("Enter RFM values for a customer to predict which segment they belong to.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Enter Customer RFM Values")
        recency   = st.number_input("🗓️ Recency (days since last purchase)", min_value=0,   max_value=1000, value=30,   step=1)
        frequency = st.number_input("🔁 Frequency (number of purchases)",    min_value=1,   max_value=500,  value=5,    step=1)
        monetary  = st.number_input("💰 Monetary (total spend £)",           min_value=0.0, max_value=100000.0, value=500.0, step=10.0)

        predict_btn = st.button("🔍 Predict Segment", use_container_width=True)

    with col2:
        st.markdown("#### RFM Reference Ranges")
        summary = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
        st.dataframe(summary.style.format({"Recency":"{:.0f}d","Frequency":"{:.1f}","Monetary":"£{:.0f}"}),
                     use_container_width=True)

    st.markdown("---")

    if predict_btn:
        input_arr = np.array([[recency, frequency, monetary]])
        scaled    = scaler.transform(input_arr)
        cluster   = km.predict(scaled)[0]
        segment   = label_map[cluster]
        color     = SEGMENT_COLORS[segment]

        st.markdown(
            f"""
            <div style='background:{color}22; border:2px solid {color};
                        padding:24px; border-radius:12px; text-align:center;'>
                <h1 style='color:{color}; margin:0'>{SEGMENT_ICONS[segment]} {segment}</h1>
                <p style='font-size:16px; margin-top:8px'>{SEGMENT_DESC[segment]}</p>
                <hr style='border-color:{color}44'>
                <p style='color:#555'>R={recency}d &nbsp;|&nbsp; F={frequency} purchases &nbsp;|&nbsp; M=£{monetary:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 💡 Recommended Action")
        actions = {
            "High-Value": "- Enrol in VIP loyalty program\n- Send exclusive early-access offers\n- Personalised thank-you notes",
            "Regular":    "- Cross-sell complementary products\n- Offer bundle discounts\n- Send monthly newsletters",
            "Occasional": "- Re-engagement email with discount\n- Showcase new arrivals\n- Flash sale notifications",
            "At-Risk":    "- Urgent win-back campaign (20%+ discount)\n- Personalised 'We miss you' email\n- Survey to understand churn reason",
        }
        st.markdown(actions[segment])

# ══════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Recommendation":
    st.title("🎯 Product Recommendation")
    st.markdown("Enter a product name to get 5 similar product recommendations.")
    st.markdown("---")

    all_products = sorted(sim_df.index.tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        product_input = st.text_input("🔎 Enter Product Name", placeholder="e.g. GREEN VINTAGE SPOT BEAKER")
        use_dropdown  = st.checkbox("Or pick from catalogue", value=False)
        if use_dropdown:
            product_input = st.selectbox("Select a product", all_products)

    rec_btn = st.button("✨ Get Recommendations", use_container_width=False)

    if rec_btn and product_input:
        query = product_input.strip().upper()

        # Exact or fuzzy match
        if query in sim_df.index:
            matched = query
        else:
            matches = [p for p in sim_df.index if query in p]
            if not matches:
                st.error(f"❌ Product '{product_input}' not found. Try the dropdown to browse available products.")
                st.stop()
            matched = matches[0]
            st.info(f"Matched to: **{matched}**")

        recs = (sim_df[matched]
                .drop(matched)
                .sort_values(ascending=False)
                .head(5))

        st.markdown(f"#### Recommendations for: **{matched}**")
        st.markdown("---")

        for i, (prod, score) in enumerate(recs.items(), 1):
            bar_pct = int(score * 100)
            st.markdown(
                f"""
                <div style='background:#f8f9fa; border-radius:8px;
                            padding:12px 16px; margin-bottom:8px;
                            border-left:4px solid #457b9d;'>
                    <span style='font-size:15px; font-weight:600'>#{i} &nbsp; {prod}</span>
                    <div style='margin-top:6px; background:#dee2e6; border-radius:4px; height:8px;'>
                        <div style='width:{bar_pct}%; background:#457b9d;
                                    border-radius:4px; height:8px;'></div>
                    </div>
                    <span style='font-size:11px; color:#888'>Similarity: {score:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif rec_btn:
        st.warning("Please enter a product name first.")

    with st.expander("📋 Browse full product catalogue"):
        st.dataframe(pd.DataFrame({"Product Name": all_products}), use_container_width=True)
