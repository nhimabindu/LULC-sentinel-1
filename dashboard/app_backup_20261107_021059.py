
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sentinel-1 LULC Explorer",
    page_icon="🌍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

MAP_DIR = ROOT / "outputs/monthly_maps"
VIS_DIR = ROOT / "outputs/visualizations/monthly_maps"
STATS_DIR = ROOT / "outputs/statistics"
EVAL_DIR = ROOT / "outputs/evaluation"

CITIES = ["Heidelberg", "Nairobi", "Sydney"]

MONTHS = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

CLASS_NAMES = [
    "Tree cover",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare vegetation",
    "Permanent water"
]

CLASS_COLORS = [
    "#26734D",
    "#A8D96C",
    "#E6C84F",
    "#D94A4A",
    "#C49A6C",
    "#3485C5"
]

cmap = ListedColormap(CLASS_COLORS)
cmap.set_bad("lightgray")

norm = BoundaryNorm(
    np.arange(-0.5, 6.5, 1),
    cmap.N
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


monthly_df = load_csv(
    STATS_DIR / "common_mask_monthly_class_percentages.csv"
)

change_df = load_csv(
    STATS_DIR / "month_to_month_class_change_pp.csv"
)

annual_df = load_csv(
    STATS_DIR / "january_december_class_change.csv"
)

transition_df = load_csv(
    STATS_DIR / "month_to_month_pixel_transitions.csv"
)

test_metrics = load_csv(
    EVAL_DIR / "fcn8s_test_overall_metrics.csv"
)

city_metrics = load_csv(
    EVAL_DIR / "fcn8s_city_wise_test_metrics.csv"
)

class_metrics = load_csv(
    EVAL_DIR / "fcn8s_city_class_metrics.csv"
)

model_comparison = load_csv(
    EVAL_DIR / "model_comparison_validation.csv"
)


# --------------------------------------------------
# 3. HEADER
# --------------------------------------------------

st.title("🌍 Sentinel-1 Land Use / Land Cover Explorer")

st.markdown(
    """
    **Monthly LULC classification using Sentinel-1 SAR imagery**

    Explore monthly land-cover predictions, class percentages,
    area changes and model evaluation for three study cities.
    """
)

st.divider()


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

selected_city = st.sidebar.selectbox(
    "Select study city",
    CITIES
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Study period:** January–December 2025

    **Satellite:** Sentinel-1

    **Input bands:** VV and VH

    **Selected model:** FCN-8s

    **Resolution:** 10 metres
    """
)


# --------------------------------------------------
# 5. TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Monthly Maps",
    "Class Distribution",
    "Change Analysis",
    "Model Evaluation"
])


# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

with tab1:

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    if not test_metrics.empty:

        metrics = test_metrics.iloc[0]

        col1.metric(
            "Test Accuracy",
            f"{metrics['Test Overall Accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Test mIoU",
            f"{metrics['Test Mean IoU'] * 100:.2f}%"
        )

        col3.metric(
            "Mean F1",
            f"{metrics['Test Mean F1'] * 100:.2f}%"
        )

    col4.metric(
        "Monthly Maps",
        "36"
    )

    st.divider()

    st.subheader(f"{selected_city} — Annual LULC Overview")

    image_path = (
        VIS_DIR /
        f"{selected_city.lower()}_12_month_lulc_maps.png"
    )

    if image_path.exists():
        st.image(
            str(image_path),
            caption=f"{selected_city}: 12 monthly LULC predictions",
            use_container_width=True
        )
    else:
        st.warning("Monthly map visualisation is not available.")

    st.info(
        "The displayed maps are model predictions based on Sentinel-1 VV/VH."
    )


# --------------------------------------------------
# TAB 2: MONTHLY MAP EXPLORER
# --------------------------------------------------

with tab2:

    st.header("Monthly LULC Map Explorer")

    selected_month = st.selectbox(
        "Choose month",
        MONTHS
    )

    month_number = MONTHS.index(selected_month) + 1

    month_code = f"{month_number:02d}"

    map_path = (
        MAP_DIR /
        selected_city /
        f"{selected_city.lower()}_2025_{month_code}_lulc.tif"
    )

    if map_path.exists():

        with rasterio.open(map_path) as src:

            prediction = src.read(1)

            valid = (
                (src.read_masks(1) > 0) &
                (prediction != 255)
            )

            masked = np.ma.masked_where(
                ~valid,
                prediction
            )

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.imshow(
            masked,
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        ax.set_title(
            f"{selected_city} — {selected_month} 2025"
        )

        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        # Download original georeferenced prediction
        st.download_button(
            "Download GeoTIFF prediction",
            data=map_path.read_bytes(),
            file_name=map_path.name,
            mime="image/tiff"
        )

    else:
        st.warning("Monthly prediction map not found.")

    st.subheader("LULC Class Legend")

    legend_cols = st.columns(3)

    for i, name in enumerate(CLASS_NAMES):

        legend_cols[i % 3].markdown(
            f"<span style='color:{CLASS_COLORS[i]}'>"
            f"■</span> **{name}**",
            unsafe_allow_html=True
        )


# --------------------------------------------------
# TAB 3: CLASS DISTRIBUTION
# --------------------------------------------------

with tab3:

    st.header("Monthly Class Distribution")

    if not monthly_df.empty:

        city_data = monthly_df[
            monthly_df["City"] == selected_city
        ].copy()

        pivot = city_data.pivot(
            index="Month",
            columns="Class",
            values="Percentage"
        )

        pivot = pivot.reindex(
            [f"{i:02d}" for i in range(1, 13)]
        )

        pivot = pivot.reindex(columns=CLASS_NAMES)

        pivot.index = MONTHS

        st.subheader("Monthly predicted class coverage")

        st.line_chart(pivot)

        st.subheader("Monthly class percentages")

        st.dataframe(
            pivot.round(2),
            use_container_width=True
        )

        st.subheader("Monthly area statistics")

        area_table = city_data.pivot(
            index="Month",
            columns="Class",
            values="Area_Hectares"
        )

        area_table = area_table.reindex(
            [f"{i:02d}" for i in range(1, 13)]
        )

        area_table.index = MONTHS

        st.dataframe(
            area_table.round(2),
            use_container_width=True
        )

    else:
        st.warning("Monthly statistics are unavailable.")


# --------------------------------------------------
# TAB 4: CHANGE ANALYSIS
# --------------------------------------------------

with tab4:

    st.header("Land-cover Change Analysis")

    st.subheader("January to December net change")

    if not annual_df.empty:

        city_annual = annual_df[
            annual_df["City"] == selected_city
        ]

        annual_table = city_annual.set_index("Class")[
            [
                "January_Percentage",
                "December_Percentage",
                "Net_Change_Percentage_Points",
                "Net_Area_Change_Hectares"
            ]
        ]

        annual_table.columns = [
            "January (%)",
            "December (%)",
            "Net change (pp)",
            "Net area change (ha)"
        ]

        st.dataframe(
            annual_table.round(3),
            use_container_width=True
        )

    st.divider()

    st.subheader("Month-to-month predicted pixel changes")

    if not transition_df.empty:

        city_transition = transition_df[
            transition_df["City"] == selected_city
        ].copy()

        city_transition["Period"] = (
            city_transition["From_Month"] +
            " → " +
            city_transition["To_Month"]
        )

        chart_data = city_transition.set_index("Period")[
            ["Changed_Pixel_Percentage"]
        ]

        st.line_chart(chart_data)

        st.dataframe(
            city_transition[
                [
                    "Period",
                    "Changed_Pixels",
                    "Unchanged_Pixels",
                    "Changed_Pixel_Percentage"
                ]
            ].round(2),
            use_container_width=True
        )

    st.caption(
        "These changes describe differences between monthly model "
        "predictions. They are not independently verified real-world "
        "land-cover changes."
    )


# --------------------------------------------------
# TAB 5: MODEL EVALUATION
# --------------------------------------------------

with tab5:

    st.header("Model Performance")

    st.subheader("Independent test performance by city")

    if not city_metrics.empty:

        st.dataframe(
            city_metrics.round(4),
            use_container_width=True
        )

        city_chart = city_metrics.set_index("City")[
            ["Overall Accuracy", "Mean IoU", "Mean F1"]
        ]

        st.bar_chart(city_chart)

    st.divider()

    st.subheader("Class-wise IoU")

    if not class_metrics.empty:

        selected_classes = class_metrics[
            class_metrics["City"] == selected_city
        ]

        class_pivot = selected_classes.set_index("Class")[
            ["IoU"]
        ]

        st.bar_chart(class_pivot)

        st.dataframe(
            selected_classes.round(4),
            use_container_width=True
        )

    st.divider()

    st.subheader("Validation model comparison")

    if not model_comparison.empty:

        st.dataframe(
            model_comparison.round(4),
            use_container_width=True
        )

    st.warning(
        "Performance is measured against WorldCover 2021 proxy labels. "
        "Sentinel-1 imagery is from 2025, so these metrics should not "
        "be interpreted as definitive 2025 ground-truth accuracy."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Sentinel-1 LULC Project | 2025 | "
    "Six-class semantic segmentation | FCN-8s"
)
