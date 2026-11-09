
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import folium
import io
import base64

from PIL import Image
from folium.plugins import Fullscreen, MousePosition, MeasureControl
from streamlit_folium import st_folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds

import plotly.express as px
import plotly.graph_objects as go

from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentinel-1 LULC | Research Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent

MAP_DIR = ROOT / "outputs" / "monthly_maps"

VIS_DIR = (
    ROOT / "outputs"
    / "visualizations"
    / "all_city_monthly_maps"
)

STATS_DIR = ROOT / "outputs" / "statistics"

EVAL_DIR = ROOT / "outputs" / "evaluation"

MODEL_PATH = ROOT / "models" / "fcn8s_best.pt"


# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================

YEAR = 2025

CITIES = [
    "Berlin",
    "Munich",
    "Hamburg",
    "Bengaluru",
    "New York",
    "Sao Paulo",
    "Cairo",
    "Freiburg",
    "Heidelberg",
    "Nairobi",
    "Sydney",
    "Singapore"
]

CITY_PREFIX = {
    "Berlin": "berlin",
    "Munich": "munich",
    "Hamburg": "hamburg",
    "Bengaluru": "bengaluru",
    "New York": "new_york",
    "Sao Paulo": "sao_paulo",
    "Cairo": "cairo",
    "Freiburg": "freiburg",
    "Heidelberg": "heidelberg",
    "Nairobi": "nairobi",
    "Sydney": "sydney",
    "Singapore": "singapore"
}

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
    "#287A55",
    "#A8D96C",
    "#E6C84F",
    "#D94A4A",
    "#C49A6C",
    "#3485C5"
]

COLOR_MAP = dict(zip(CLASS_NAMES, CLASS_COLORS))

CMAP = ListedColormap(CLASS_COLORS)
CMAP.set_bad("#E5E7EB")

NORM = BoundaryNorm(
    np.arange(-0.5, 6.5, 1),
    CMAP.N
)


# ============================================================
# 3. PROFESSIONAL DASHBOARD STYLING
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #F5F7FA;
    }

    h1 {
        color: #17324D;
        font-weight: 750;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #24445F;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E1E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20, 40, 60, 0.04);
    }

    [data-testid="stSidebar"] {
        background-color: #EDF2F7;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .research-note {
        background-color: #EAF2F8;
        border-left: 4px solid #31708E;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0;
    }

    .small-muted {
        color: #64748B;
        font-size: 13px;
    }

    .section-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string):

    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def month_number(value):

    if pd.isna(value):
        return np.nan

    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]

    if pd.notna(numeric):
        number = int(numeric)

        if 1 <= number <= 12:
            return number

    text = str(value).strip().capitalize()

    if text in MONTHS:
        return MONTHS.index(text) + 1

    abbreviations = {
        name[:3]: index + 1
        for index, name in enumerate(MONTHS)
    }

    return abbreviations.get(text[:3], np.nan)


def month_label(value):

    number = month_number(value)

    if pd.isna(number):
        return str(value)

    return MONTHS[int(number) - 1]


def percentage_value(value):

    if pd.isna(value):
        return np.nan

    value = float(value)

    if 0 <= abs(value) <= 1:
        return value * 100

    return value


def metric_value(dataframe, column):

    if dataframe.empty or column not in dataframe.columns:
        return None

    return percentage_value(dataframe.iloc[0][column])


def clean_month_column(dataframe):

    if dataframe.empty or "Month" not in dataframe.columns:
        return dataframe

    result = dataframe.copy()

    result["MonthNo"] = result["Month"].apply(month_number)

    return result


def prediction_path(city, month):

    prefix = CITY_PREFIX[city]

    return (
        MAP_DIR / city
        / f"{prefix}_{YEAR}_{month:02d}_lulc.tif"
    )


def annual_map_path(city):

    prefix = CITY_PREFIX[city]

    return VIS_DIR / f"{prefix}_12_month_lulc_maps.png"


@st.cache_data(show_spinner=False)
def read_prediction(path_string):

    with rasterio.open(path_string) as src:

        prediction = src.read(1, masked=True)

        values = prediction.filled(255)

        valid = (
            ~np.ma.getmaskarray(prediction)
            & np.isin(values, range(6))
        )

        masked = np.ma.masked_where(
            ~valid,
            values
        )

        return masked


def show_download(path):

    if path.exists():

        st.download_button(
            label="Download GeoTIFF",
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/tiff",
            use_container_width=True
        )


def empty_message(message):

    st.info(message)


def style_table(dataframe):

    return dataframe.style.format(
        precision=2,
        na_rep="—"
    )


def show_class_legend():

    cols = st.columns(3)

    for index, name in enumerate(CLASS_NAMES):

        cols[index % 3].markdown(
            f"<span style='color:{CLASS_COLORS[index]}'>"
            f"■</span> **{name}**",
            unsafe_allow_html=True
        )


# ============================================================
# 5. LOAD PROJECT DATA
# ============================================================

monthly_df = load_csv(
    str(STATS_DIR / "common_mask_monthly_class_percentages.csv")
)

change_df = load_csv(
    str(STATS_DIR / "month_to_month_class_change_pp.csv")
)

annual_df = load_csv(
    str(STATS_DIR / "january_december_class_change.csv")
)

transition_df = load_csv(
    str(STATS_DIR / "month_to_month_pixel_transitions.csv")
)

test_metrics = load_csv(
    str(EVAL_DIR / "fcn8s_test_overall_metrics.csv")
)

city_metrics = load_csv(
    str(EVAL_DIR / "fcn8s_city_wise_test_metrics.csv")
)

class_metrics = load_csv(
    str(EVAL_DIR / "fcn8s_city_class_metrics.csv")
)

model_comparison = load_csv(
    str(EVAL_DIR / "model_comparison_validation.csv")
)

dw_monthly = load_csv(
    str(EVAL_DIR / "all_12_cities_dynamic_world_monthly_metrics.csv")
)

dw_city_summary = load_csv(
    str(EVAL_DIR / "all_12_cities_dynamic_world_city_summary.csv")
)

dw_global_summary = load_csv(
    str(EVAL_DIR / "all_12_cities_dynamic_world_global_summary.csv")
)

dw_class_metrics = load_csv(
    str(EVAL_DIR / "overall_class_metrics_adequate_coverage.csv")
)

monthly_df = clean_month_column(monthly_df)
change_df = change_df.copy()
transition_df = transition_df.copy()
dw_monthly = clean_month_column(dw_monthly)


# ============================================================
# 6. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("🌍 LULC Research Explorer")

st.sidebar.caption(
    "Sentinel-1 based land-cover analysis"
)

st.sidebar.divider()

selected_city = st.sidebar.selectbox(
    "Select study city",
    CITIES,
    index=CITIES.index("Heidelberg")
)

selected_month = st.sidebar.selectbox(
    "Select month",
    MONTHS,
    index=0
)

month_no = MONTHS.index(selected_month) + 1

compare_enabled = st.sidebar.checkbox(
    "Enable map comparison",
    value=False
)

compare_month = None

if compare_enabled:

    compare_month = st.sidebar.selectbox(
        "Compare with month",
        MONTHS,
        index=5
    )

    compare_month_no = MONTHS.index(compare_month) + 1

st.sidebar.divider()

st.sidebar.markdown("### Project configuration")

st.sidebar.markdown("""
**Study period:** January–December 2025

**Satellite:** Sentinel-1 SAR

**Input bands:** VV and VH

**Selected model:** FCN-based segmentation

**Output classes:** 6

**Nominal pixel size:** 10 m
""")

st.sidebar.divider()

st.sidebar.caption(
    "Research prototype | Master's Project"
)

st.sidebar.caption(
    "Coordinate-based inference is not enabled in this dashboard version."
)


# ============================================================
# 7. HEADER
# ============================================================

st.title("🌍 Sentinel-1 Land Use / Land Cover Explorer")

st.markdown(
    """
    **Multi-city monthly land-cover mapping and change analysis**

    A research dashboard for exploring Sentinel-1 VV/VH-derived
    land-cover predictions across twelve study areas during 2025.
    """
)

st.markdown(
    '<div class="research-note">'
    '<b>Study scope:</b> 12 cities | 144 monthly prediction maps | '
    '6 land-cover classes | Sentinel-1 VV/VH'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 8. DATA AVAILABILITY
# ============================================================

map_count = 0

for city in CITIES:

    for month in range(1, 13):

        if prediction_path(city, month).exists():
            map_count += 1

city_monthly = monthly_df[
    monthly_df["City"] == selected_city
].copy() if not monthly_df.empty else pd.DataFrame()

city_annual = annual_df[
    annual_df["City"] == selected_city
].copy() if not annual_df.empty else pd.DataFrame()

city_transition = transition_df[
    transition_df["City"] == selected_city
].copy() if not transition_df.empty else pd.DataFrame()

city_change = change_df[
    change_df["City"] == selected_city
].copy() if not change_df.empty else pd.DataFrame()

dw_city_monthly = dw_monthly[
    dw_monthly["City"] == selected_city
].copy() if not dw_monthly.empty else pd.DataFrame()


# ============================================================
# 9. NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

    "Overview",
    "Monthly Maps",
    "Class Distribution",
    "Change Analysis",
    "Model Evaluation",
    "Cross-Sensor Validation",
    "Terrestrial View",

])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader("Project overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Study cities",
        len(CITIES)
    )

    col2.metric(
        "Prediction maps",
        f"{map_count}/144"
    )

    col3.metric(
        "Land-cover classes",
        "6"
    )

    col4.metric(
        "Study period",
        "12 months"
    )

    st.divider()

    st.subheader(f"{selected_city} — Annual land-cover overview")

    image_path = annual_map_path(selected_city)

    if image_path.exists():

        st.image(
            str(image_path),
            caption=(
                f"{selected_city}: Monthly Sentinel-1 "
                "land-cover predictions for 2025"
            ),
            use_container_width=True
        )

    else:
        empty_message("Annual visualisation is not available.")

    st.subheader("Model test performance")

    if not test_metrics.empty:

        m = test_metrics.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Overall Accuracy",
            f"{percentage_value(m['Test Overall Accuracy']):.2f}%"
        )

        b.metric(
            "Mean IoU",
            f"{percentage_value(m['Test Mean IoU']):.2f}%"
        )

        c.metric(
            "Mean F1",
            f"{percentage_value(m['Test Mean F1']):.2f}%"
        )

    st.divider()

    st.subheader("Selected city — monthly data availability")

    if not city_monthly.empty:

        available = city_monthly.groupby(
            "MonthNo"
        )["Percentage"].count()

        availability = pd.DataFrame({
            "Month": range(1, 13),
            "Month Name": MONTHS,
            "Available class records": [
                int(available.get(i, 0))
                for i in range(1, 13)
            ]
        })

        fig = px.bar(
            availability,
            x="Month Name",
            y="Available class records",
            title="Monthly statistics availability",
            color_discrete_sequence=["#3478A5"]
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Class records",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="research-note">'
        '<b>Interpretation:</b> The mapped outputs represent '
        'model predictions from Sentinel-1 VV/VH observations. '
        'They should not be interpreted as independently verified '
        'land-cover ground truth.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TAB 2 — MONTHLY MAPS
# ============================================================

with tab2:

    st.subheader("Monthly LULC map explorer")

    st.caption(
        f"Selected area: {selected_city} | "
        f"Selected period: {selected_month} {YEAR}"
    )

    current_path = prediction_path(
        selected_city,
        month_no
    )

    if current_path.exists():

        current_prediction = read_prediction(
            str(current_path)
        )

        if compare_enabled and compare_month_no != month_no:

            previous_path = prediction_path(
                selected_city,
                compare_month_no
            )

            if previous_path.exists():

                previous_prediction = read_prediction(
                    str(previous_path)
                )

                left, right = st.columns(2)

                with left:

                    st.markdown(f"**{selected_month}**")

                    fig, ax = plt.subplots(figsize=(7, 6))

                    ax.imshow(
                        current_prediction,
                        cmap=CMAP,
                        norm=NORM,
                        interpolation="nearest"
                    )

                    ax.set_title(selected_month)
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                    show_download(current_path)

                with right:

                    st.markdown(f"**{compare_month}**")

                    fig, ax = plt.subplots(figsize=(7, 6))

                    ax.imshow(
                        previous_prediction,
                        cmap=CMAP,
                        norm=NORM,
                        interpolation="nearest"
                    )

                    ax.set_title(compare_month)
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                    show_download(previous_path)

            else:

                st.warning(
                    f"Map for {compare_month} is unavailable."
                )

        else:

            fig, ax = plt.subplots(figsize=(10, 7))

            ax.imshow(
                current_prediction,
                cmap=CMAP,
                norm=NORM,
                interpolation="nearest"
            )

            ax.set_title(
                f"{selected_city} — {selected_month} {YEAR}"
            )

            ax.axis("off")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            show_download(current_path)

    else:

        st.warning("Prediction map not found.")

    st.divider()

    st.subheader("Land-cover legend")

    show_class_legend()

    st.caption(
        "NoData pixels are excluded from the displayed classification."
    )


# ============================================================
# TAB 3 — CLASS DISTRIBUTION
# ============================================================

with tab3:

    st.subheader("Monthly land-cover distribution")

    if not city_monthly.empty:

        pivot = city_monthly.pivot_table(
            index="MonthNo",
            columns="Class",
            values="Percentage",
            aggfunc="mean"
        ).reindex(
            range(1, 13)
        ).reindex(
            columns=CLASS_NAMES
        )

        plot_data = (
            pivot.reset_index()
            .rename(columns={"MonthNo": "Month"})
            .melt(
                id_vars="Month",
                var_name="Class",
                value_name="Percentage"
            )
        )

        fig = px.area(
            plot_data,
            x="Month",
            y="Percentage",
            color="Class",
            color_discrete_map=COLOR_MAP,
            title="Monthly land-cover percentage trends"
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=MONTHS
            ),
            yaxis_title="Class coverage (%)",
            xaxis_title="Month",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        selected_distribution = city_monthly[
            city_monthly["MonthNo"] == month_no
        ].copy()

        left, right = st.columns([1.1, 1])

        with left:

            st.markdown(
                f"**Class distribution — {selected_month}**"
            )

            if not selected_distribution.empty:

                fig = px.pie(
                    selected_distribution,
                    names="Class",
                    values="Percentage",
                    hole=0.45,
                    color="Class",
                    color_discrete_map=COLOR_MAP
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="label+percent"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                empty_message("No class statistics for this month.")

        with right:

            st.markdown("**Class percentage and area**")

            if not selected_distribution.empty:

                display_columns = [
                    "Class",
                    "Percentage",
                    "Area (ha)"
                ]

                available_columns = [
                    column
                    for column in display_columns
                    if column in selected_distribution.columns
                ]

                st.dataframe(
                    style_table(
                        selected_distribution[available_columns]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

        st.subheader("Monthly class percentage table")

        st.dataframe(
            style_table(pivot.rename(index=lambda x: MONTHS[x - 1])),
            use_container_width=True
        )

        csv_data = pivot.to_csv().encode("utf-8")

        st.download_button(
            "Download class distribution CSV",
            data=csv_data,
            file_name=f"{CITY_PREFIX[selected_city]}_class_distribution.csv",
            mime="text/csv"
        )

    else:

        empty_message("Monthly class statistics are unavailable.")


# ============================================================
# TAB 4 — CHANGE ANALYSIS
# ============================================================

with tab4:

    st.subheader("Land-cover change analysis")

    if not city_annual.empty:

        st.markdown(
            "**January–December net percentage-point change**"
        )

        annual_plot = city_annual.copy()

        annual_plot["Net_Change_Percentage_Points"] = pd.to_numeric(
            annual_plot["Net_Change_Percentage_Points"],
            errors="coerce"
        )

        fig = px.bar(
            annual_plot,
            x="Net_Change_Percentage_Points",
            y="Class",
            orientation="h",
            color="Net_Change_Percentage_Points",
            color_continuous_scale="RdYlGn",
            title="Annual net change by land-cover class"
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )

        fig.update_layout(
            xaxis_title="Net change (percentage points)",
            yaxis_title="Land-cover class"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        annual_display = annual_plot.rename(
            columns={
                "January_Percentage": "January (%)",
                "December_Percentage": "December (%)",
                "Net_Change_Percentage_Points": "Net change (pp)",
                "Net_Area_Change_Hectares": "Net area change (ha)"
            }
        )

        st.dataframe(
            style_table(annual_display),
            use_container_width=True,
            hide_index=True
        )

    else:

        empty_message("Annual change statistics are unavailable.")

    st.divider()

    st.subheader("Month-to-month class changes")

    if not city_change.empty:

        selected_class = st.selectbox(
            "Filter by land-cover class",
            ["All classes"] + CLASS_NAMES,
            key="change_class_filter"
        )

        filtered_change = city_change.copy()

        if selected_class != "All classes":

            filtered_change = filtered_change[
                filtered_change["Class"] == selected_class
            ]

        filtered_change["FromNo"] = (
            filtered_change["From_Month"].apply(month_number)
        )

        filtered_change["ToNo"] = (
            filtered_change["To_Month"].apply(month_number)
        )

        filtered_change["Period"] = (
            filtered_change["From_Month"].apply(month_label)
            + " → "
            + filtered_change["To_Month"].apply(month_label)
        )

        filtered_change = filtered_change.sort_values(
            ["FromNo", "ToNo"]
        )

        if not filtered_change.empty:

            fig = px.line(
                filtered_change,
                x="Period",
                y="Change_Percentage_Points",
                color="Class",
                markers=True,
                title="Month-to-month class percentage-point change",
                color_discrete_map=COLOR_MAP
            )

            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Change (percentage points)",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            style_table(filtered_change),
            use_container_width=True,
            hide_index=True
        )

    else:

        empty_message("Month-to-month class change data unavailable.")

    st.divider()

    st.subheader("Monthly pixel transition")

    if not city_transition.empty:

        city_transition = city_transition.copy()

        city_transition["FromNo"] = (
            city_transition["From_Month"].apply(month_number)
        )

        city_transition["ToNo"] = (
            city_transition["To_Month"].apply(month_number)
        )

        city_transition["Period"] = (
            city_transition["From_Month"].apply(month_label)
            + " → "
            + city_transition["To_Month"].apply(month_label)
        )

        city_transition = city_transition.sort_values(
            ["FromNo", "ToNo"]
        )

        fig = px.line(
            city_transition,
            x="Period",
            y="Changed_Pixel_Percentage",
            markers=True,
            title="Percentage of pixels changing between months"
        )

        fig.update_layout(
            xaxis_title="Month-to-month period",
            yaxis_title="Changed pixels (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            city_transition[
                [
                    "Period",
                    "Changed_Pixels",
                    "Unchanged_Pixels",
                    "Changed_Pixel_Percentage"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        '<div class="research-note">'
        '<b>Interpretation:</b> Change statistics describe differences '
        'between monthly model predictions. They do not independently '
        'confirm real-world land-cover changes.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TAB 5 — MODEL EVALUATION
# ============================================================

with tab5:

    st.subheader("Model performance evaluation")

    if not test_metrics.empty:

        metrics = test_metrics.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test overall accuracy",
            f"{percentage_value(metrics['Test Overall Accuracy']):.2f}%"
        )

        c2.metric(
            "Test mean IoU",
            f"{percentage_value(metrics['Test Mean IoU']):.2f}%"
        )

        c3.metric(
            "Test mean F1",
            f"{percentage_value(metrics['Test Mean F1']):.2f}%"
        )

        st.caption(
            f"Best checkpoint epoch: {metrics.get('Checkpoint Epoch', 'N/A')}"
        )

    st.divider()

    st.subheader("Independent test performance — 3 unseen cities")

    if not city_metrics.empty:

        city_plot = city_metrics.copy()

        metric_columns = [
            "Overall Accuracy",
            "Mean IoU",
            "Mean F1"
        ]

        for column in metric_columns:

            if column in city_plot.columns:
                city_plot[column] = city_plot[column].apply(
                    percentage_value
                )

        display_columns = [
            col for col in [
                "City",
                "Overall Accuracy",
                "Mean IoU",
                "Mean F1",
                "Valid Pixels"
            ]
            if col in city_plot.columns
        ]

        st.dataframe(
            city_plot[display_columns].round(3),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            city_plot,
            x="City",
            y=[
                "Overall Accuracy",
                "Mean IoU",
                "Mean F1"
            ],
            barmode="group",
            title="Held-out test results by city"
        )

        fig.update_layout(
            yaxis_title="Score (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Class-wise test performance")

    if not class_metrics.empty:

        available_cities = class_metrics["City"].unique().tolist()

        selected_eval_city = st.selectbox(
            "Select test city",
            available_cities,
            key="evaluation_city"
        )

        selected_classes = class_metrics[
            class_metrics["City"] == selected_eval_city
        ].copy()

        for column in ["IoU", "F1 Score"]:

            if column in selected_classes.columns:

                selected_classes[column] = (
                    selected_classes[column].apply(percentage_value)
                )

        fig = px.bar(
            selected_classes,
            x="Class",
            y=["IoU", "F1 Score"],
            barmode="group",
            title=f"Class-wise performance — {selected_eval_city}"
        )

        fig.update_layout(
            yaxis_title="Score (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            selected_classes,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Validation model comparison")

    if not model_comparison.empty:

        comparison = model_comparison.copy()

        for column in [
            "Overall Accuracy",
            "Mean IoU",
            "Mean F1"
        ]:

            if column in comparison.columns:

                comparison[column] = (
                    comparison[column].apply(percentage_value)
                )

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        '<div class="research-note">'
        '<b>Training-label limitation:</b> The model uses ESA WorldCover '
        '2021 proxy labels while the Sentinel-1 imagery is from 2025. '
        'The reported test metrics therefore measure agreement with '
        'these proxy labels, not definitive 2025 ground-truth accuracy.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TAB 6 — CROSS-SENSOR VALIDATION
# ============================================================

with tab6:

    st.subheader("Sentinel-1 vs Dynamic World")

    st.markdown(
        """
        This section compares Sentinel-1 model predictions with
        Dynamic World classifications derived from Sentinel-2.
        """
    )

    st.markdown(
        '<div class="research-note">'
        '<b>Important:</b> Dynamic World is an independent satellite-derived '
        'comparison product, not verified ground truth. The metrics below '
        'measure cross-sensor product agreement.'
        '</div>',
        unsafe_allow_html=True
    )

    if not dw_global_summary.empty:

        adequate_summary = dw_global_summary[
            dw_global_summary["Coverage Group"]
            == "ADEQUATE_COVERAGE_ONLY"
        ]

        all_summary = dw_global_summary[
            dw_global_summary["Coverage Group"]
            == "ALL_EVALUATED_MONTHS"
        ]

        if not adequate_summary.empty:

            m = adequate_summary.iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Overall agreement",
                f"{m['Overall Agreement (%)']:.2f}%"
            )

            c2.metric(
                "Macro F1",
                f"{m['Macro F1 (%)']:.2f}%"
            )

            c3.metric(
                "Mean IoU",
                f"{m['Mean IoU (%)']:.2f}%"
            )

            st.caption(
                f"Adequate-coverage city-months: "
                f"{int(m['City-Months'])}"
            )

        if not all_summary.empty:

            st.markdown("**All evaluated months versus adequate coverage**")

            st.dataframe(
                dw_global_summary.round(3),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.subheader("Cross-sensor agreement by city")

    if not dw_city_summary.empty:

        adequate_city = dw_city_summary[
            dw_city_summary["Coverage Group"]
            == "ADEQUATE_COVERAGE_ONLY"
        ].copy()

        st.dataframe(
            adequate_city.round(3),
            use_container_width=True,
            hide_index=True
        )

        if not adequate_city.empty:

            fig = px.bar(
                adequate_city,
                x="City",
                y=[
                    "Overall Agreement (%)",
                    "Macro F1 (%)",
                    "Mean IoU (%)"
                ],
                barmode="group",
                title="City-wise cross-sensor agreement"
            )

            fig.update_layout(
                yaxis_title="Agreement metric (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("Monthly cross-sensor coverage")

    if not dw_monthly.empty:

        coverage_city = dw_monthly[
            dw_monthly["City"] == selected_city
        ].copy()

        if not coverage_city.empty:

            coverage_city["MonthName"] = (
                coverage_city["MonthNo"].apply(
                    lambda x: MONTHS[int(x) - 1]
                    if pd.notna(x) else "Unknown"
                )
            )

            fig = px.bar(
                coverage_city,
                x="MonthName",
                y="Comparable Coverage (%)",
                color="Coverage Category",
                title=f"Monthly comparable coverage — {selected_city}"
            )

            fig.add_hline(
                y=80,
                line_dash="dash",
                line_color="green",
                annotation_text="80% coverage threshold"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                coverage_city[
                    [
                        "Date",
                        "Comparable Pixels",
                        "Comparable Coverage (%)",
                        "Overall Agreement (%)",
                        "Macro F1 (%)",
                        "Mean IoU (%)",
                        "Coverage Category",
                        "Status"
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.subheader("Class-wise cross-sensor agreement")

    if not dw_class_metrics.empty:

        st.dataframe(
            dw_class_metrics.round(3),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            dw_class_metrics,
            x="Class",
            y=["F1 (%)", "IoU (%)"],
            barmode="group",
            title="Class-wise Dynamic World agreement"
        )

        fig.update_layout(
            yaxis_title="Score (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        empty_message(
            "Class-wise cross-sensor metrics are not available."
        )


# ============================================================


# ============================================================
# TERRESTRIAL MAP TAB
# ============================================================

with tab7:

    st.subheader("🌐 Interactive Terrestrial LULC Explorer")

    st.markdown(
        """
        Explore Sentinel-1 land-cover predictions over satellite imagery.
        Zoom, pan, switch basemaps and click the map to inspect coordinates.
        """
    )

    map_col1, map_col2, map_col3 = st.columns([1.2, 1, 1])

    with map_col1:

        map_city = st.selectbox(
            "Study city",
            CITIES,
            index=CITIES.index(selected_city)
            if "selected_city" in globals()
            and selected_city in CITIES else 0,
            key="terrestrial_city"
        )

    with map_col2:

        map_month_name = st.selectbox(
            "Prediction month",
            MONTHS,
            index=MONTHS.index(selected_month)
            if "selected_month" in globals()
            and selected_month in MONTHS else 0,
            key="terrestrial_month"
        )

    with map_col3:

        overlay_opacity = st.slider(
            "LULC overlay opacity",
            min_value=0.2,
            max_value=1.0,
            value=0.75,
            step=0.05,
            key="terrestrial_opacity"
        )

    map_month_number = MONTHS.index(map_month_name) + 1

    map_prefix = CITY_PREFIX[map_city]

    terrestrial_path = (
        MAP_DIR / map_city
        / f"{map_prefix}_2025_{map_month_number:02d}_lulc.tif"
    )

    if terrestrial_path.exists():

        # ----------------------------------------------------
        # Reproject categorical raster to geographic CRS
        # ----------------------------------------------------

        @st.cache_data(show_spinner=False)
        def prepare_terrestrial_overlay(path_string):

            with rasterio.open(path_string) as src:

                if src.crs is None:
                    raise ValueError(
                        "Prediction raster has no coordinate reference system."
                    )

                prediction = src.read(1, masked=True)

                source_array = (
                    prediction.filled(255).astype(np.uint8)
                )

                dst_crs = "EPSG:4326"

                dst_transform, dst_width, dst_height = (
                    calculate_default_transform(
                        src.crs,
                        dst_crs,
                        src.width,
                        src.height,
                        *src.bounds
                    )
                )

                destination = np.full(
                    (dst_height, dst_width),
                    255,
                    dtype=np.uint8
                )

                reproject(
                    source=source_array,
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=255,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest
                )

            # Create transparent RGBA image
            rgba = np.zeros(
                (dst_height, dst_width, 4),
                dtype=np.uint8
            )

            for class_id, hex_colour in enumerate(CLASS_COLORS):

                rgb = tuple(
                    int(channel * 255)
                    for channel in to_rgb(hex_colour)
                )

                mask = destination == class_id

                rgba[mask, :3] = rgb
                rgba[mask, 3] = 255

            image = Image.fromarray(rgba)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            encoded_image = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

            west, south, east, north = array_bounds(
                dst_height,
                dst_width,
                dst_transform
            )

            valid_pixels = int(
                np.isin(destination, range(6)).sum()
            )

            return (
                encoded_image,
                west,
                south,
                east,
                north,
                valid_pixels
            )

        try:

            with st.spinner(
                "Preparing georeferenced LULC satellite overlay..."
            ):

                (
                    encoded_image,
                    west,
                    south,
                    east,
                    north,
                    valid_pixels
                ) = prepare_terrestrial_overlay(
                    str(terrestrial_path)
                )

            # ------------------------------------------------
            # Create interactive map
            # ------------------------------------------------

            centre = [
                (south + north) / 2,
                (west + east) / 2
            ]

            terrain_map = folium.Map(
                location=centre,
                zoom_start=13,
                tiles=None,
                control_scale=True
            )

            # Satellite basemap
            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ),
                attr="Tiles © Esri and its contributors",
                name="Satellite imagery",
                overlay=False,
                control=True,
                max_zoom=19
            ).add_to(terrain_map)

            # Street map
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="Street map",
                overlay=False,
                control=True
            ).add_to(terrain_map)

            # Terrain basemap
            folium.TileLayer(
                tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                attr="© OpenStreetMap contributors, SRTM | OpenTopoMap",
                name="Terrain",
                overlay=False,
                control=True,
                max_zoom=17
            ).add_to(terrain_map)

            # LULC prediction overlay
            folium.raster_layers.ImageOverlay(
                image="data:image/png;base64," + encoded_image,
                bounds=[
                    [south, west],
                    [north, east]
                ],
                opacity=overlay_opacity,
                name="Sentinel-1 LULC prediction",
                interactive=True,
                cross_origin=False,
                zindex=2
            ).add_to(terrain_map)

            # Map controls
            folium.LayerControl(
                position="topright",
                collapsed=False
            ).add_to(terrain_map)

            Fullscreen(
                position="topleft"
            ).add_to(terrain_map)

            MousePosition(
                position="bottomleft",
                separator=" | ",
                prefix="Coordinates:"
            ).add_to(terrain_map)

            MeasureControl(
                position="topleft",
                primary_length_unit="kilometers",
                secondary_length_unit="meters"
            ).add_to(terrain_map)

            # Class legend
            legend_items = "".join(
                f"""
                <div style="margin-bottom:6px;">
                    <span style="
                        display:inline-block;
                        width:13px;
                        height:13px;
                        background:{CLASS_COLORS[i]};
                        margin-right:8px;
                        border-radius:3px;
                    "></span>{CLASS_NAMES[i]}
                </div>
                """
                for i in range(6)
            )

            legend_html = f"""
            <div style="
                position: fixed;
                bottom: 35px;
                right: 15px;
                z-index: 9999;
                background: white;
                padding: 13px 16px;
                border-radius: 9px;
                box-shadow: 0 2px 9px rgba(0,0,0,0.25);
                font-size: 12px;
                color: #243447;
                min-width: 175px;
            ">
                <b style="font-size:13px;">LULC Classes</b>
                <hr style="margin:7px 0;">
                {legend_items}
            </div>
            """

            terrain_map.get_root().html.add_child(
                folium.Element(legend_html)
            )

            st.markdown(
                f"**{map_city} | {map_month_name} {2025}**"
            )

            st.caption(
                f"{valid_pixels:,} valid classified pixels | "
                f"Source: Sentinel-1 FCN prediction"
            )

            # Render interactive Folium map
            map_state = st_folium(
                terrain_map,
                height=680,
                use_container_width=True,
                key=f"terrestrial_{map_city}_{map_month_number}",
                returned_objects=["last_object_clicked"]
            )

            # Show clicked geographic coordinates
            if isinstance(map_state, dict):

                clicked = (
                    map_state.get("last_object_clicked")
                    or map_state.get("last_clicked")
                )

                if clicked:

                    latitude = clicked.get("lat")
                    longitude = clicked.get("lng")

                    if latitude is not None and longitude is not None:

                        st.success(
                            f"Selected coordinates: "
                            f"Latitude {latitude:.6f}, "
                            f"Longitude {longitude:.6f}"
                        )

                        st.code(
                            f"{latitude:.6f}, {longitude:.6f}"
                        )

            # Download the original georeferenced map
            st.download_button(
                "Download original LULC GeoTIFF",
                data=terrestrial_path.read_bytes(),
                file_name=terrestrial_path.name,
                mime="image/tiff"
            )

            st.caption(
                "Satellite basemap tiles are provided by external "
                "map services and require an internet connection."
            )

        except Exception as error:

            st.error(
                f"Could not prepare terrestrial map: {error}"
            )

    else:

        st.warning(
            "The selected monthly prediction map is unavailable."
        )

    st.markdown(
        """
        **Research note:** The satellite image is used as a visual
        background. The coloured overlay represents your existing
        Sentinel-1 model prediction; it is not a new prediction
        generated from the satellite basemap.
        """
    )


# 10. FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns([3, 1])

with footer_left:

    st.caption(
        "Sentinel-1 Land Use / Land Cover Research Project | 2025"
    )

    st.caption(
        "Six-class semantic segmentation | FCN-based architecture"
    )

with footer_right:

    st.caption("Master's Research Prototype")
