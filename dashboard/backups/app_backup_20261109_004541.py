from __future__ import annotations

from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium

from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from folium.plugins import Fullscreen, MousePosition, MeasureControl
from streamlit_folium import st_folium


# ============================================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# ============================================================================

st.set_page_config(
    page_title="Sentinel-1 LULC | Research Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = ROOT / "outputs" / "monthly_maps"
VIS_DIR = ROOT / "outputs" / "visualizations" / "all_city_monthly_maps"
STATS_DIR = ROOT / "outputs" / "statistics"
EVAL_DIR = ROOT / "outputs" / "evaluation"
MODEL_PATH = ROOT / "models" / "fcn8s_best.pt"

YEAR = 2025
COVERAGE_THRESHOLD = 80.0

CITIES = [
    "Berlin", "Munich", "Hamburg", "Bengaluru", "New York", "Sao Paulo",
    "Cairo", "Freiburg", "Heidelberg", "Nairobi", "Sydney", "Singapore",
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
    "Singapore": "singapore",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

CLASS_NAMES = [
    "Tree cover", "Grassland", "Cropland", "Built-up",
    "Bare vegetation", "Permanent water",
]

CLASS_COLORS = [
    "#26734D", "#A8D96C", "#E6C84F", "#D94A4A", "#C49A6C", "#3485C5",
]
COLOR_MAP = dict(zip(CLASS_NAMES, CLASS_COLORS))
CMAP = ListedColormap(CLASS_COLORS)
CMAP.set_bad("#E5E7EB")
NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), CMAP.N)


# ============================================================================
# 2. STYLE
# ============================================================================

st.markdown(
    """
    <style>
      .stApp { background: #F5F7FA; }
      h1 { color: #17324D; font-weight: 750; letter-spacing: -0.7px; }
      h2, h3 { color: #24445F; font-weight: 650; }
      [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E1E8F0;
        padding: 16px; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20,40,60,.04);
      }
      [data-testid="stSidebar"] { background: #EDF2F7; }
      .research-note {
        background: #EAF2F8; border-left: 4px solid #31708E;
        padding: 13px 17px; border-radius: 6px; margin: 10px 0;
      }
      .muted { color: #64748B; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# 3. HELPER FUNCTIONS AND CACHED DATA LOADERS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def month_number(value):
    """Convert numeric, zero-padded or month-name values to 1..12."""
    if pd.isna(value):
        return np.nan
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        number = int(numeric)
        if 1 <= number <= 12:
            return number
    text = str(value).strip().capitalize()
    if text in MONTHS:
        return MONTHS.index(text) + 1
    abbreviations = {name[:3]: index + 1 for index, name in enumerate(MONTHS)}
    return abbreviations.get(text[:3], np.nan)


def month_label(value) -> str:
    number = month_number(value)
    if pd.isna(number):
        return str(value)
    return MONTHS[int(number) - 1]


def as_percent(value):
    """Convert fractional metrics to percent while retaining percent metrics."""
    if pd.isna(value):
        return np.nan
    value = float(value)
    return value * 100 if abs(value) <= 1 else value


def clean_month_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "Month" not in dataframe.columns:
        return dataframe
    result = dataframe.copy()
    result["MonthNo"] = result["Month"].apply(month_number)
    return result


def prediction_path(city: str, month: int) -> Path:
    prefix = CITY_PREFIX[city]
    return MAP_DIR / city / f"{prefix}_{YEAR}_{month:02d}_lulc.tif"


def annual_map_path(city: str) -> Path:
    return VIS_DIR / f"{CITY_PREFIX[city]}_12_month_lulc_maps.png"


@st.cache_data(show_spinner=False)
def read_prediction(path_string: str):
    with rasterio.open(path_string) as src:
        data = src.read(1, masked=True)
        values = data.filled(255)
        valid = (~np.ma.getmaskarray(data)) & np.isin(values, range(6))
        return np.ma.masked_where(~valid, values)


@st.cache_data(show_spinner=False)
def prepare_terrestrial_overlay(path_string: str):
    """Reproject a categorical GeoTIFF to WGS84 and encode a transparent PNG."""
    with rasterio.open(path_string) as src:
        if src.crs is None:
            raise ValueError("Prediction raster has no CRS.")
        source = src.read(1, masked=True).filled(255).astype(np.uint8)
        dst_crs = "EPSG:4326"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        destination = np.full((dst_height, dst_width), 255, dtype=np.uint8)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=255,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=255,
            resampling=Resampling.nearest,
        )

    rgba = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
    for class_id, hex_color in enumerate(CLASS_COLORS):
        rgb = tuple(int(channel * 255) for channel in to_rgb(hex_color))
        mask = destination == class_id
        rgba[mask, :3] = rgb
        rgba[mask, 3] = 255

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    valid_pixels = int(np.isin(destination, range(6)).sum())
    return encoded, west, south, east, north, valid_pixels


def show_download(path: Path, label: str = "Download GeoTIFF"):
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/tiff",
            use_container_width=True,
            key=f"download_{path.name}_{label.replace(' ', '_')}",
        )


def show_class_legend():
    columns = st.columns(3)
    for index, name in enumerate(CLASS_NAMES):
        columns[index % 3].markdown(
            f"<span style='color:{CLASS_COLORS[index]}'>■</span> **{name}**",
            unsafe_allow_html=True,
        )


def safe_metric(dataframe: pd.DataFrame, column: str):
    if dataframe.empty or column not in dataframe.columns:
        return None
    return as_percent(dataframe.iloc[0][column])


def display_dataframe(dataframe: pd.DataFrame, precision: int = 2):
    if dataframe.empty:
        st.info("No records are available for this selection.")
    else:
        st.dataframe(
            dataframe.style.format(precision=precision, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================================
# 4. LOAD PROJECT DATA
# ============================================================================

monthly_df = clean_month_column(load_csv(str(STATS_DIR / "common_mask_monthly_class_percentages.csv")))
change_df = load_csv(str(STATS_DIR / "month_to_month_class_change_pp.csv"))
annual_df = load_csv(str(STATS_DIR / "january_december_class_change.csv"))
transition_df = load_csv(str(STATS_DIR / "month_to_month_pixel_transitions.csv"))

test_metrics = load_csv(str(EVAL_DIR / "fcn8s_test_overall_metrics.csv"))
test_city_metrics = load_csv(str(EVAL_DIR / "fcn8s_city_wise_test_metrics.csv"))
test_class_metrics = load_csv(str(EVAL_DIR / "fcn8s_city_class_metrics.csv"))
model_comparison = load_csv(str(EVAL_DIR / "model_comparison_validation.csv"))

dw_monthly = clean_month_column(load_csv(str(EVAL_DIR / "all_12_cities_dynamic_world_monthly_metrics.csv")))
dw_city_summary = load_csv(str(EVAL_DIR / "all_12_cities_dynamic_world_city_summary.csv"))
dw_global_summary = load_csv(str(EVAL_DIR / "all_12_cities_dynamic_world_global_summary.csv"))
dw_class_metrics = load_csv(str(EVAL_DIR / "overall_class_metrics_adequate_coverage.csv"))

# Fallback: compute the class-level table from saved confusion matrices if needed.
if dw_class_metrics.empty:
    confusion = load_csv(str(EVAL_DIR / "all_12_cities_dynamic_world_confusion_matrices.csv"))
    if not confusion.empty and not dw_monthly.empty:
        eligible = dw_monthly[
            (dw_monthly["Status"] == "EVALUATED")
            & (pd.to_numeric(dw_monthly["Comparable Coverage (%)"], errors="coerce") >= COVERAGE_THRESHOLD)
        ][["City", "Month"]]
        selected = confusion.merge(eligible, on=["City", "Month"], how="inner")
        matrix = selected.pivot_table(
            index="S1_Class", columns="DW_Class", values="Pixel_Count",
            aggfunc="sum", fill_value=0,
        ).reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0).to_numpy(dtype=float)
        rows = []
        for i, class_name in enumerate(CLASS_NAMES):
            tp = matrix[i, i]
            row_total = matrix[i, :].sum()
            col_total = matrix[:, i].sum()
            f1_den = row_total + col_total
            iou_den = f1_den - tp
            rows.append({
                "Class": class_name,
                "F1 (%)": 2 * tp / f1_den * 100 if f1_den else np.nan,
                "IoU (%)": tp / iou_den * 100 if iou_den else np.nan,
            })
        dw_class_metrics = pd.DataFrame(rows)


# ============================================================================
# 5. SIDEBAR NAVIGATION AND GLOBAL FILTERS
# ============================================================================

st.sidebar.title("🌍 LULC Research Explorer")
st.sidebar.caption("Sentinel-1 based land-cover analysis")
st.sidebar.divider()

selected_city = st.sidebar.selectbox(
    "Select study city", CITIES, index=CITIES.index("Heidelberg")
)
selected_month = st.sidebar.selectbox("Select month", MONTHS, index=0)
selected_month_no = MONTHS.index(selected_month) + 1

map_compare = st.sidebar.checkbox("Enable monthly map comparison", value=False)
compare_month = None
if map_compare:
    compare_month = st.sidebar.selectbox("Compare with month", MONTHS, index=5)

st.sidebar.divider()
st.sidebar.markdown("### Project configuration")
st.sidebar.markdown(
    "**Study period:** January–December 2025  \n"
    "**Satellite:** Sentinel-1 SAR  \n"
    "**Input bands:** VV and VH  \n"
    "**Model:** Lightweight FCN-8s-inspired segmentation  \n"
    "**Output classes:** Six  \n"
    "**Nominal pixel size:** 10 m"
)
st.sidebar.caption("Research prototype | Master's project")
st.sidebar.caption("Coordinate-based fresh inference is not enabled in this build.")

# Data slices for the selected city.
city_monthly = monthly_df[monthly_df["City"] == selected_city].copy() if not monthly_df.empty else pd.DataFrame()
city_annual = annual_df[annual_df["City"] == selected_city].copy() if not annual_df.empty else pd.DataFrame()
city_transition = transition_df[transition_df["City"] == selected_city].copy() if not transition_df.empty else pd.DataFrame()
city_change = change_df[change_df["City"] == selected_city].copy() if not change_df.empty else pd.DataFrame()
city_dw_monthly = dw_monthly[dw_monthly["City"] == selected_city].copy() if not dw_monthly.empty else pd.DataFrame()


# ============================================================================
# 6. HEADER AND NAVIGATION
# ============================================================================

st.title("🌍 Sentinel-1 Land Use / Land Cover Explorer")
st.markdown(
    "**Multi-city monthly land-cover mapping, change analysis and cross-sensor evaluation**\n\n"
    "Explore georeferenced Sentinel-1 VV/VH-derived predictions across twelve study areas during 2025."
)
st.markdown(
    '<div class="research-note"><b>Study scope:</b> 12 cities | 144 monthly prediction maps | '
    '6 land-cover classes | Sentinel-1 VV/VH</div>',
    unsafe_allow_html=True,
)
st.divider()

(
    tab_overview,
    tab_maps,
    tab_distribution,
    tab_change,
    tab_model,
    tab_cross_sensor,
    tab_terrestrial,
) = st.tabs([
    "Overview",
    "Monthly Maps",
    "Class Distribution",
    "Change Analysis",
    "Model Evaluation",
    "Cross-Sensor Validation",
    "Terrestrial View",
])


# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================

with tab_overview:
    st.subheader("Project overview")
    map_count = sum(prediction_path(city, month).exists() for city in CITIES for month in range(1, 13))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study cities", str(len(CITIES)))
    c2.metric("Prediction maps", f"{map_count}/144")
    c3.metric("Land-cover classes", "6")
    c4.metric("Study period", "12 months")

    st.divider()
    st.subheader(f"{selected_city} — annual land-cover overview")
    annual_image = annual_map_path(selected_city)
    if annual_image.exists():
        st.image(str(annual_image), caption=f"{selected_city}: 12 monthly Sentinel-1 LULC predictions", use_container_width=True)
    else:
        st.info("Annual map mosaic is not available for this city.")

    st.subheader("Independent test performance")
    if not test_metrics.empty:
        metrics = test_metrics.iloc[0]
        a, b, c = st.columns(3)
        if "Test Overall Accuracy" in metrics:
            a.metric("Test overall accuracy", f"{as_percent(metrics['Test Overall Accuracy']):.2f}%")
        if "Test Mean IoU" in metrics:
            b.metric("Test mean IoU", f"{as_percent(metrics['Test Mean IoU']):.2f}%")
        if "Test Mean F1" in metrics:
            c.metric("Test mean F1", f"{as_percent(metrics['Test Mean F1']):.2f}%")
    else:
        st.info("Test metrics file is not available.")

    st.divider()
    st.subheader(f"{selected_city} — class coverage across the year")
    if not city_monthly.empty and "Percentage" in city_monthly.columns:
        trend = city_monthly.pivot_table(index="MonthNo", columns="Class", values="Percentage", aggfunc="mean")
        trend = trend.reindex(index=range(1, 13), columns=CLASS_NAMES)
        plot_data = trend.reset_index().melt(id_vars="MonthNo", var_name="Class", value_name="Percentage")
        fig = px.line(
            plot_data, x="MonthNo", y="Percentage", color="Class", markers=True,
            color_discrete_map=COLOR_MAP, title="Monthly predicted class percentages",
        )
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTHS),
                          xaxis_title="Month", yaxis_title="Class coverage (%)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly statistics are unavailable for this city.")

    st.markdown(
        '<div class="research-note"><b>Interpretation:</b> Maps show model predictions from Sentinel-1 VV/VH. '
        'They are not independently verified land-cover ground truth.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB 2 — MONTHLY MAPS
# ============================================================================

with tab_maps:
    st.subheader("Monthly LULC map explorer")
    current_path = prediction_path(selected_city, selected_month_no)
    if not current_path.exists():
        st.warning(f"Prediction raster not found: {current_path.name}")
    elif map_compare and compare_month is not None and compare_month != selected_month:
        compare_no = MONTHS.index(compare_month) + 1
        compare_path = prediction_path(selected_city, compare_no)
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{selected_month} {YEAR}**")
            image = read_prediction(str(current_path))
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(image, cmap=CMAP, norm=NORM, interpolation="nearest")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            show_download(current_path)
        with right:
            st.markdown(f"**{compare_month} {YEAR}**")
            if compare_path.exists():
                image = read_prediction(str(compare_path))
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.imshow(image, cmap=CMAP, norm=NORM, interpolation="nearest")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                show_download(compare_path)
            else:
                st.warning("Comparison month raster is not available.")
    else:
        image = read_prediction(str(current_path))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(image, cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_title(f"{selected_city} — {selected_month} {YEAR}")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_download(current_path)

    st.divider()
    st.subheader("Land-cover class legend")
    show_class_legend()
    st.caption("NoData pixels are excluded from the displayed classification.")


# ============================================================================
# TAB 3 — CLASS DISTRIBUTION
# ============================================================================

with tab_distribution:
    st.subheader(f"{selected_city} — monthly land-cover distribution")
    if city_monthly.empty:
        st.info("Monthly class statistics are unavailable.")
    else:
        pivot = city_monthly.pivot_table(index="MonthNo", columns="Class", values="Percentage", aggfunc="mean")
        pivot = pivot.reindex(index=range(1, 13), columns=CLASS_NAMES)
        plot_data = pivot.reset_index().melt(id_vars="MonthNo", var_name="Class", value_name="Percentage")

        fig = px.area(
            plot_data, x="MonthNo", y="Percentage", color="Class",
            color_discrete_map=COLOR_MAP, title="Monthly land-cover percentage trends",
        )
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTHS),
            xaxis_title="Month", yaxis_title="Class coverage (%)", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        selected_distribution = city_monthly[city_monthly["MonthNo"] == selected_month_no].copy()
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown(f"**Class composition — {selected_month}**")
            if not selected_distribution.empty:
                fig = px.pie(
                    selected_distribution, names="Class", values="Percentage", hole=0.42,
                    color="Class", color_discrete_map=COLOR_MAP,
                )
                fig.update_traces(textposition="inside", textinfo="label+percent")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No class statistics for this month.")
        with right:
            st.markdown("**Class percentage and area**")
            cols = [c for c in ["Class", "Percentage", "Area (ha)"] if c in selected_distribution.columns]
            display_dataframe(selected_distribution[cols])

        st.subheader("Monthly class percentage table")
        display_dataframe(pivot.rename(index=lambda x: MONTHS[x - 1]).reset_index(names="Month"))
        st.download_button(
            "Download class distribution CSV",
            data=pivot.to_csv().encode("utf-8"),
            file_name=f"{CITY_PREFIX[selected_city]}_class_distribution_2025.csv",
            mime="text/csv",
        )


# ============================================================================
# TAB 4 — CHANGE ANALYSIS
# ============================================================================

with tab_change:
    st.subheader(f"{selected_city} — land-cover change analysis")
    if not city_annual.empty:
        st.markdown("**January–December net percentage-point change**")
        annual_plot = city_annual.copy()
        annual_plot["Net_Change_Percentage_Points"] = pd.to_numeric(annual_plot["Net_Change_Percentage_Points"], errors="coerce")
        fig = px.bar(
            annual_plot, x="Net_Change_Percentage_Points", y="Class", orientation="h",
            color="Net_Change_Percentage_Points", color_continuous_scale="RdYlGn",
            title="Annual net change by land-cover class",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(xaxis_title="Net change (percentage points)", yaxis_title="Class")
        st.plotly_chart(fig, use_container_width=True)

        annual_display = annual_plot.rename(columns={
            "January_Percentage": "January (%)",
            "December_Percentage": "December (%)",
            "Net_Change_Percentage_Points": "Net change (pp)",
            "Net_Area_Change_Hectares": "Net area change (ha)",
        })
        display_dataframe(annual_display)
    else:
        st.info("Annual January–December change statistics are unavailable.")

    st.divider()
    st.subheader("Month-to-month class percentage change")
    if not city_change.empty:
        class_filter = st.selectbox("Filter land-cover class", ["All classes"] + CLASS_NAMES, key="change_class_filter")
        filtered = city_change.copy()
        if class_filter != "All classes":
            filtered = filtered[filtered["Class"] == class_filter]
        filtered["FromNo"] = filtered["From_Month"].apply(month_number)
        filtered["ToNo"] = filtered["To_Month"].apply(month_number)
        filtered["Period"] = filtered["From_Month"].apply(month_label) + " → " + filtered["To_Month"].apply(month_label)
        filtered = filtered.sort_values(["FromNo", "ToNo"])
        if not filtered.empty:
            fig = px.line(
                filtered, x="Period", y="Change_Percentage_Points", color="Class",
                markers=True, color_discrete_map=COLOR_MAP,
                title="Month-to-month class percentage-point change",
            )
            fig.update_layout(xaxis_title="Period", yaxis_title="Change (percentage points)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        display_dataframe(filtered)
    else:
        st.info("Month-to-month class change data are unavailable.")

    st.divider()
    st.subheader("Month-to-month predicted pixel changes")
    if not city_transition.empty:
        transitions = city_transition.copy()
        transitions["FromNo"] = transitions["From_Month"].apply(month_number)
        transitions["ToNo"] = transitions["To_Month"].apply(month_number)
        transitions["Period"] = transitions["From_Month"].apply(month_label) + " → " + transitions["To_Month"].apply(month_label)
        transitions = transitions.sort_values(["FromNo", "ToNo"])
        fig = px.line(
            transitions, x="Period", y="Changed_Pixel_Percentage", markers=True,
            title="Share of pixels with a different prediction between months",
        )
        fig.update_layout(xaxis_title="Period", yaxis_title="Changed pixels (%)")
        st.plotly_chart(fig, use_container_width=True)
        display_dataframe(transitions[["Period", "Changed_Pixels", "Unchanged_Pixels", "Changed_Pixel_Percentage"]])
    else:
        st.info("Pixel transition statistics are unavailable.")

    st.markdown(
        '<div class="research-note"><b>Interpretation:</b> These values describe differences between monthly model predictions. '
        'They do not independently confirm real-world land-cover change.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB 5 — MODEL EVALUATION (ALL 12 CITY OVERVIEW + HELD-OUT TEST SPLIT)
# ============================================================================

with tab_model:
    st.subheader("Evaluation overview — all 12 study cities")
    st.caption(
        "All 12 cities are shown below using the available Sentinel-1 vs Dynamic World "
        "cross-sensor comparison. This is product agreement, not ground-truth model accuracy. "
        "The supervised held-out test results are shown separately further down."
    )

    # Show the complete study set, including cities with no usable comparison months.
    split_by_city = {
        "Berlin": "Train", "Munich": "Train", "Hamburg": "Train",
        "Bengaluru": "Train", "New York": "Train", "Sao Paulo": "Train",
        "Cairo": "Train", "Freiburg": "Validation", "Singapore": "Validation",
        "Heidelberg": "Test", "Nairobi": "Test", "Sydney": "Test",
    }
    if not dw_city_summary.empty:
        all_cities = dw_city_summary[
            dw_city_summary["Coverage Group"] == "ALL_EVALUATED_MONTHS"
        ].copy()
        all_cities = all_cities.drop_duplicates(subset=["City"]).set_index("City")
        all_cities = all_cities.reindex(CITIES).rename_axis("City").reset_index()
        all_cities["Study split"] = all_cities["City"].map(split_by_city)
        all_cities["Evaluated Months"] = pd.to_numeric(
            all_cities.get("Evaluated Months", pd.Series(index=all_cities.index, dtype=float)),
            errors="coerce",
        ).fillna(0).astype(int)
        for metric_col in ["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"]:
            if metric_col in all_cities.columns:
                all_cities[metric_col] = pd.to_numeric(all_cities[metric_col], errors="coerce")

        overview_cols = [
            col for col in [
                "City", "Study split", "Evaluated Months", "Comparable Pixels",
                "Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)",
            ] if col in all_cities.columns
        ]
        st.dataframe(
            all_cities[overview_cols].style.format(
                {col: "{:.2f}" for col in ["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"] if col in overview_cols},
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart_data = all_cities.dropna(subset=[
            col for col in ["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"]
            if col in all_cities.columns
        ])
        if not chart_data.empty:
            fig = px.bar(
                chart_data, x="City",
                y=[col for col in ["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"] if col in chart_data.columns],
                barmode="group", title="Cross-sensor comparison across all study cities",
                hover_data=[col for col in ["Study split", "Evaluated Months"] if col in chart_data.columns],
            )
            fig.update_layout(xaxis_title="Study city", yaxis_title="Score (%)", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download all 12 city evaluation results",
            data=all_cities[overview_cols].to_csv(index=False).encode("utf-8"),
            file_name="all_12_cities_evaluation_overview.csv",
            mime="text/csv",
        )
    else:
        st.warning("The all-city Dynamic World summary file is missing, so the 12-city comparison cannot be displayed.")

    st.markdown(
        '<div class="research-note"><b>How to read this table:</b> The metrics above compare the '
        'Sentinel-1 model map with Dynamic World for each city. They are cross-sensor agreement '
        'metrics, not verified ground-truth accuracy. Evaluated-month counts differ by city because '
        'some monthly comparisons have missing scenes or insufficient valid pixels.</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Supervised held-out test results — 3 unseen cities")
    st.caption(
        "Only Heidelberg, Nairobi and Sydney belong to the held-out test split. "
        "These are the cities used for the reported independent supervised test metrics; "
        "the training and validation cities must not be presented as unseen test cities."
    )
    if not test_metrics.empty:
        m = test_metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        if "Test Overall Accuracy" in m:
            c1.metric("Held-out test OA", f"{as_percent(m['Test Overall Accuracy']):.2f}%")
        if "Test Mean IoU" in m:
            c2.metric("Held-out test mIoU", f"{as_percent(m['Test Mean IoU']):.2f}%")
        if "Test Mean F1" in m:
            c3.metric("Held-out test mean F1", f"{as_percent(m['Test Mean F1']):.2f}%")

    if not test_city_metrics.empty:
        test_plot = test_city_metrics.copy()
        for col in ["Overall Accuracy", "Mean IoU", "Mean F1"]:
            if col in test_plot.columns:
                test_plot[col] = test_plot[col].apply(as_percent)
        st.dataframe(test_plot, use_container_width=True, hide_index=True)
        fig = px.bar(
            test_plot, x="City", y=[col for col in ["Overall Accuracy", "Mean IoU", "Mean F1"] if col in test_plot.columns],
            barmode="group", title="Supervised held-out test results by unseen city",
        )
        fig.update_layout(yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Held-out city metrics are unavailable.")

    st.divider()
    st.subheader("Class-wise held-out test performance")
    if not test_class_metrics.empty:
        available_test_cities = test_class_metrics["City"].dropna().unique().tolist()
        selected_test_city = st.selectbox("Select unseen test city", available_test_cities, key="test_eval_city")
        classes = test_class_metrics[test_class_metrics["City"] == selected_test_city].copy()
        for col in ["IoU", "F1 Score"]:
            if col in classes.columns:
                classes[col] = classes[col].apply(as_percent)
        fig = px.bar(classes, x="Class", y=[col for col in ["IoU", "F1 Score"] if col in classes.columns], barmode="group", title=f"Class-wise test results — {selected_test_city}")
        fig.update_layout(yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)
        display_dataframe(classes)

    st.divider()
    st.subheader("Validation model comparison")
    if not model_comparison.empty:
        comparison = model_comparison.copy()
        for col in ["Overall Accuracy", "Mean IoU", "Mean F1"]:
            if col in comparison.columns:
                comparison[col] = comparison[col].apply(as_percent)
        display_dataframe(comparison)

    st.markdown(
        '<div class="research-note"><b>Training-label limitation:</b> ESA WorldCover 2021 proxy labels were used with 2025 Sentinel-1 imagery. '
        'Supervised test scores therefore measure agreement with those proxy labels, not definitive 2025 ground-truth accuracy.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB 6 — CROSS-SENSOR VALIDATION
# ============================================================================

with tab_cross_sensor:
    st.subheader("Sentinel-1 vs Dynamic World cross-sensor validation")
    st.markdown(
        '<div class="research-note"><b>Important:</b> Dynamic World is an independently produced Sentinel-2-derived classification product. '
        'The metrics below measure cross-sensor product agreement, not accuracy against verified ground truth.</div>',
        unsafe_allow_html=True,
    )

    if not dw_global_summary.empty:
        st.markdown("**Global summary**")
        display_dataframe(dw_global_summary)
        adequate_global = dw_global_summary[dw_global_summary["Coverage Group"] == "ADEQUATE_COVERAGE_ONLY"]
        if not adequate_global.empty:
            m = adequate_global.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Adequate city-months", int(m["City-Months"]))
            c2.metric("Comparable pixels", f"{int(m['Comparable Pixels']):,}")
            c3.metric("Overall agreement", f"{m['Overall Agreement (%)']:.2f}%")
            c4.metric("Mean IoU", f"{m['Mean IoU (%)']:.2f}%")

    st.divider()
    st.subheader("All-city comparison table")
    if not dw_city_summary.empty:
        adequate_city = dw_city_summary[dw_city_summary["Coverage Group"] == "ADEQUATE_COVERAGE_ONLY"].copy()
        adequate_city = adequate_city.set_index("City").reindex(CITIES).reset_index()
        display_dataframe(adequate_city)
        fig = px.bar(
            adequate_city, x="City",
            y=["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"],
            barmode="group", title="Adequate-coverage cross-sensor results by city",
        )
        fig.update_layout(yaxis_title="Metric (%)", xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Monthly coverage and agreement — {selected_city}")
    if not city_dw_monthly.empty:
        selected_dw = city_dw_monthly.copy()
        selected_dw["MonthName"] = selected_dw["MonthNo"].apply(lambda x: MONTHS[int(x) - 1] if pd.notna(x) else "Unknown")
        fig = px.bar(
            selected_dw, x="MonthName", y="Comparable Coverage (%)",
            color="Coverage Category", title="Comparable pixel coverage by month",
        )
        fig.add_hline(y=COVERAGE_THRESHOLD, line_dash="dash", line_color="green", annotation_text="80% coverage threshold")
        st.plotly_chart(fig, use_container_width=True)
        columns = [c for c in [
            "Date", "Comparable Pixels", "Comparable Coverage (%)",
            "Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)",
            "Coverage Category", "Status",
        ] if c in selected_dw.columns]
        display_dataframe(selected_dw[columns])
    else:
        st.info("Monthly cross-sensor records are unavailable for this city.")

    st.divider()
    st.subheader("Class-wise cross-sensor agreement — adequate coverage")
    if not dw_class_metrics.empty:
        display_dataframe(dw_class_metrics)
        metric_cols = [c for c in ["F1 (%)", "IoU (%)"] if c in dw_class_metrics.columns]
        if metric_cols:
            fig = px.bar(dw_class_metrics, x="Class", y=metric_cols, barmode="group", title="Class-wise cross-sensor F1 and IoU")
            fig.update_layout(yaxis_title="Score (%)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Class-wise cross-sensor metric file is not available.")

    st.caption("Low-coverage months remain visible and are flagged. Very small sample coverage can produce unstable or misleading metric values.")


# ============================================================================
# TAB 7 — TERRESTRIAL VIEW
# ============================================================================

with tab_terrestrial:
    st.subheader("🌐 Interactive terrestrial LULC explorer")
    st.markdown("View the georeferenced Sentinel-1 prediction over satellite imagery. Switch basemaps, zoom, measure distances and click the map to inspect coordinates.")

    map_col1, map_col2, map_col3 = st.columns([1.2, 1, 1])
    with map_col1:
        map_city = st.selectbox("Map study city", CITIES, index=CITIES.index(selected_city), key="terrestrial_city")
    with map_col2:
        map_month_name = st.selectbox("Map prediction month", MONTHS, index=MONTHS.index(selected_month), key="terrestrial_month")
    with map_col3:
        overlay_opacity = st.slider("LULC overlay opacity", 0.2, 1.0, 0.75, 0.05, key="terrestrial_opacity")

    map_month_no = MONTHS.index(map_month_name) + 1
    terrestrial_path = prediction_path(map_city, map_month_no)

    if terrestrial_path.exists():
        try:
            with st.spinner("Preparing georeferenced LULC overlay..."):
                encoded, west, south, east, north, valid_pixels = prepare_terrestrial_overlay(str(terrestrial_path))

            centre = [(south + north) / 2, (west + east) / 2]
            terrain_map = folium.Map(location=centre, zoom_start=13, tiles=None, control_scale=True)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri and its contributors",
                name="Satellite imagery", overlay=False, control=True, max_zoom=19,
            ).add_to(terrain_map)
            folium.TileLayer(
                tiles="OpenStreetMap", name="Street map", overlay=False, control=True,
            ).add_to(terrain_map)
            folium.TileLayer(
                tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                attr="© OpenStreetMap contributors, SRTM | OpenTopoMap",
                name="Terrain", overlay=False, control=True, max_zoom=17,
            ).add_to(terrain_map)

            folium.raster_layers.ImageOverlay(
                image="data:image/png;base64," + encoded,
                bounds=[[south, west], [north, east]],
                opacity=overlay_opacity,
                name="Sentinel-1 LULC prediction",
                interactive=False,
                cross_origin=False,
                zindex=2,
            ).add_to(terrain_map)

            folium.LayerControl(position="topright", collapsed=False).add_to(terrain_map)
            Fullscreen(position="topleft").add_to(terrain_map)
            MousePosition(position="bottomleft", separator=" | ", prefix="Coordinates:").add_to(terrain_map)
            MeasureControl(position="topleft", primary_length_unit="kilometers", secondary_length_unit="meters").add_to(terrain_map)

            legend_items = "".join(
                f'<div style="margin:5px 0"><span style="display:inline-block;width:13px;height:13px;background:{CLASS_COLORS[i]};margin-right:8px;border-radius:3px"></span>{CLASS_NAMES[i]}</div>'
                for i in range(6)
            )
            legend_html = (
                '<div style="position:fixed;bottom:35px;right:15px;z-index:9999;'
                'background:white;padding:12px 15px;border-radius:9px;'
                'box-shadow:0 2px 9px rgba(0,0,0,.25);font-size:12px;color:#243447;min-width:175px">'
                '<b style="font-size:13px">LULC classes</b><hr style="margin:7px 0">'
                + legend_items + '</div>'
            )
            terrain_map.get_root().html.add_child(folium.Element(legend_html))

            st.markdown(f"**{map_city} | {map_month_name} {YEAR}**")
            st.caption(f"{valid_pixels:,} valid classified pixels | Overlay: Sentinel-1 model predictions")
            map_state = st_folium(
                terrain_map,
                height=680,
                use_container_width=True,
                key=f"terrestrial_{map_city}_{map_month_no}",
                returned_objects=["last_clicked"],
            )
            clicked = (map_state or {}).get("last_clicked") if isinstance(map_state, dict) else None
            if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
                lat, lon = clicked["lat"], clicked["lng"]
                st.success(f"Selected coordinates: latitude {lat:.6f}, longitude {lon:.6f}")
                st.code(f"{lat:.6f}, {lon:.6f}")

            show_download(terrestrial_path, "Download original LULC GeoTIFF")
            st.caption("Basemap tiles are provided by external services and require an internet connection.")
        except Exception as error:
            st.error(f"Could not prepare terrestrial view: {type(error).__name__}: {error}")
    else:
        st.warning("The selected monthly prediction map is unavailable.")

    st.markdown(
        '<div class="research-note"><b>Research note:</b> The satellite basemap is a visual background. '
        'The coloured overlay is the saved Sentinel-1 model prediction. Clicking the map reports coordinates only; '
        'it does not run fresh inference for arbitrary coordinates.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 7. FOOTER
# ============================================================================

st.divider()
st.caption("Sentinel-1 Land Use / Land Cover Research Project | 2025 | 12 study cities | Six-class semantic segmentation")
st.caption("Research prototype. See the README and report for data provenance, evaluation design and limitations.")
