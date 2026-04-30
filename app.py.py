import streamlit as st
import pandas as pd
import pickle

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Diamond Analytics",
    layout="wide",
    page_icon="💎"
)

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open("diamond_pipeline.pkl", "rb"))

# ---------------- TITLE ---------------- #
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    💎 Diamond Price Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("### Enter diamond specifications below")

st.divider()

# ---------------- INPUT SECTION ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📏 Dimensions")
    carat = st.number_input("Carat", value=1.0)
    x = st.number_input("Length (mm)", value=5.0)
    y = st.number_input("Width (mm)", value=5.0)
    z = st.number_input("Height (mm)", value=3.0)

with col2:
    st.markdown("#### 📊 Physical Properties")
    depth = st.number_input("Depth (%)", value=60.0)
    table = st.number_input("Table (%)", value=55.0)

with col3:
    st.markdown("#### 💎 Quality")
    cut = st.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'])
    color = st.selectbox("Color", ['D','E','F','G','H','I','J'])
    clarity = st.selectbox("Clarity", ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])

st.divider()

# ---------------- FEATURE ENGINEERING ---------------- #
volume = x * y * z
dimension_ratio = (x + y) / (2 * z) if z != 0 else 0

def carat_category(c):
    if c < 0.5:
        return "Light"
    elif c <= 1.5:
        return "Medium"
    else:
        return "Heavy"

carat_cat = carat_category(carat)

# ---------------- DATAFRAME ---------------- #
input_df = pd.DataFrame([{
    'carat': carat,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'volume': volume,
    'dimension_ratio': dimension_ratio,
    'carat_category': carat_cat
}])

# ---------------- BUTTON ---------------- #
st.markdown("### 🎯 Prediction")

if st.button("Predict Price", use_container_width=True):
    try:
        price = model.predict(input_df)[0]

        st.success("Prediction Successful!")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                label="💵 Estimated Price",
                value=f"${price:,.2f}"
            )

        with colB:
            st.metric(
                label="💎 Carat Category",
                value=carat_cat
            )

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- FOOTER ---------------- #
st.divider()
st.caption("Built with ❤️ using Machine Learning & Streamlit")