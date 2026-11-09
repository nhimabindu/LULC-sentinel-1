from __future__ import annotations

from pathlib import Path
import base64
import io
import re
import unicodedata

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
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as transform_coordinates
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


def _normalise_city_name(value: str) -> str:
    """Normalise city names so spelling, spaces and accents do not matter."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def resolve_city_name(value: str):
    """Resolve user-entered text to one of the 12 supported study cities."""
    aliases = {
        "newyorkcity": "New York",
        "nyc": "New York",
        "saopaulo": "Sao Paulo",
        "sampa": "Sao Paulo",
    }
    key = _normalise_city_name(value)
    if not key:
        return None
    if key in aliases:
        return aliases[key]
    for city in CITIES:
        if _normalise_city_name(city) == key:
            return city
    return None


@st.cache_data(show_spinner=False)
def locate_city_centre(city: str, month: int):
    """Return the centre of a city's saved prediction raster and its centre-pixel class."""
    raster_path = prediction_path(city, month)
    if not raster_path.exists():
        return None

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None or src.width == 0 or src.height == 0:
                return None

            row, col = src.height // 2, src.width // 2
            x, y = src.xy(row, col, offset="center")
            xs, ys = transform_coordinates(src.crs, "EPSG:4326", [x], [y])
            pixel = src.read(
                1, window=((row, row + 1), (col, col + 1)), masked=True
            )[0, 0]
            class_id = None
            if not np.ma.is_masked(pixel):
                pixel_value = int(pixel)
                if pixel_value in range(6):
                    class_id = pixel_value

            return {
                "city": city,
                "class_id": class_id,
                "row": int(row),
                "col": int(col),
                "latitude": float(ys[0]),
                "longitude": float(xs[0]),
                "path": str(raster_path),
            }
    except Exception:
        return None


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
# 5. SIDEBAR CITY SEARCH AND GLOBAL FILTERS
# ============================================================================

st.sidebar.title("🌍 LULC Research Explorer")
st.sidebar.caption("Sentinel-1 based land-cover analysis")
st.sidebar.divider()

st.sidebar.markdown("### Find a study city")
with st.sidebar.form("city_search_form"):
    city_input = st.text_input(
        "Enter city name",
        value="Heidelberg",
        placeholder="e.g. Heidelberg, Berlin, Bengaluru",
        help="Enter one of the 12 study cities. You can type the city name directly.",
    )
    selected_month = st.selectbox("Select month", MONTHS, index=0, key="city_month")
    city_submitted = st.form_submit_button("🔎 Show city", use_container_width=True)

selected_month_no = MONTHS.index(selected_month) + 1
selected_city = resolve_city_name(city_input)
location_match = locate_city_centre(selected_city, selected_month_no) if selected_city else None

latitude = location_match["latitude"] if location_match else None
longitude = location_match["longitude"] if location_match else None

if not selected_city:
    st.sidebar.error("City not recognised. Enter one of the 12 study cities listed below.")
    st.sidebar.caption(", ".join(CITIES))
elif not location_match:
    st.sidebar.warning(f"The saved {selected_month} prediction map for {selected_city} is not available.")
else:
    st.sidebar.success(f"Showing **{selected_city}**")
    st.sidebar.caption("The map opens at the centre of the saved study-area raster.")
    if location_match["class_id"] is not None:
        detected_class = CLASS_NAMES[location_match["class_id"]]
        st.sidebar.caption(f"Centre pixel class: {detected_class}")
    else:
        st.sidebar.caption("The centre pixel is NoData; the full map remains available.")

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
st.sidebar.caption("City searches open existing monthly predictions. No new model inference is performed.")

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
    if selected_city:
        st.subheader(f"{selected_city} — annual land-cover overview")
        annual_image = annual_map_path(selected_city)
        if annual_image.exists():
            st.image(str(annual_image), caption=f"{selected_city}: 12 monthly Sentinel-1 LULC predictions", use_container_width=True)
        else:
            st.info("Annual map mosaic is not available for this city.")
    else:
        st.info("Enter one of the supported study city names in the sidebar to view its annual overview.")

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
    if selected_city:
        st.subheader(f"{selected_city} — class coverage across the year")
    else:
        st.subheader("Class coverage across the year")
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
    if not selected_city:
        st.warning("Enter a supported study city in the sidebar to view its monthly map.")
    else:
        current_path = prediction_path(selected_city, selected_month_no)
        if not current_path.exists():
            st.warning(f"Prediction raster not found: {current_path.name}")
        elif map_compare and compare_month is not None and compare_month != selected_month:
            compare_no = MONTHS.index(compare_month) + 1
            compare_path = prediction_path(selected_city, compare_no)
            left, right = st.columns(2)
            with left:
                st.markdown(f"**{selected_month} {YEAR}**")
                image_data = read_prediction(str(current_path))
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.imshow(image_data, cmap=CMAP, norm=NORM, interpolation="nearest")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                show_download(current_path)
            with right:
                st.markdown(f"**{compare_month} {YEAR}**")
                if compare_path.exists():
                    image_data = read_prediction(str(compare_path))
                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.imshow(image_data, cmap=CMAP, norm=NORM, interpolation="nearest")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    show_download(compare_path)
                else:
                    st.warning("Comparison month raster is not available.")
        else:
            image_data = read_prediction(str(current_path))
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(image_data, cmap=CMAP, norm=NORM, interpolation="nearest")
            ax.set_title(f"{selected_city} — {selected_month} {YEAR}")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            show_download(current_path)

    st.divider()
    st.subheader("Land-cover class legend")
    show_class_legend()
    st.caption("NoData pixels are excluded from the displayed classification.")


# TAB 3 — CLASS DISTRIBUTION
# ============================================================================

with tab_distribution:
    st.subheader(f"{selected_city or 'Selected location'} — monthly land-cover distribution")
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
    st.subheader(f"{selected_city or 'Selected location'} — land-cover change analysis")
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
# TAB 5 — MODEL EVALUATION (TEST SPLIT + ALL 12 CITY COMPARISON)
# ============================================================================

with tab_model:
    st.subheader("Model performance evaluation")
    st.caption("The held-out test set comprises Heidelberg, Nairobi and Sydney. The separate all-city comparison below uses Dynamic World and is not ground-truth test accuracy.")

    if not test_metrics.empty:
        m = test_metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        if "Test Overall Accuracy" in m:
            c1.metric("Held-out test OA", f"{as_percent(m['Test Overall Accuracy']):.2f}%")
        if "Test Mean IoU" in m:
            c2.metric("Held-out test mIoU", f"{as_percent(m['Test Mean IoU']):.2f}%")
        if "Test Mean F1" in m:
            c3.metric("Held-out test mean F1", f"{as_percent(m['Test Mean F1']):.2f}%")

    st.divider()
    st.subheader("Independent test performance — 3 unseen cities")
    if not test_city_metrics.empty:
        test_plot = test_city_metrics.copy()
        for col in ["Overall Accuracy", "Mean IoU", "Mean F1"]:
            if col in test_plot.columns:
                test_plot[col] = test_plot[col].apply(as_percent)
        st.dataframe(test_plot, use_container_width=True, hide_index=True)
        fig = px.bar(
            test_plot, x="City", y=["Overall Accuracy", "Mean IoU", "Mean F1"],
            barmode="group", title="Held-out test results by unseen city",
        )
        fig.update_layout(yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Held-out city metrics are unavailable.")

    st.divider()
    st.subheader("All 12 study cities — cross-sensor comparison")
    st.caption("The following 12-city chart compares Sentinel-1 predictions with Dynamic World. It is cross-sensor product agreement, not model accuracy against verified ground truth.")
    if not dw_city_summary.empty:
        all_cities = dw_city_summary[dw_city_summary["Coverage Group"] == "ALL_EVALUATED_MONTHS"].copy()
        all_cities = all_cities.set_index("City").reindex(CITIES).reset_index()
        all_cities["Evaluated Months"] = pd.to_numeric(all_cities["Evaluated Months"], errors="coerce").fillna(0).astype(int)
        fig = px.bar(
            all_cities, x="City",
            y=["Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)"],
            barmode="group", title="Cross-sensor comparison across all 12 cities",
        )
        fig.update_layout(xaxis_title="Study city", yaxis_title="Metric (%)", xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        display_dataframe(all_cities[[
            "City", "Evaluated Months", "Comparable Pixels",
            "Overall Agreement (%)", "Macro F1 (%)", "Mean IoU (%)",
        ]])
        st.download_button(
            "Download all 12 city comparison results",
            data=all_cities.to_csv(index=False).encode("utf-8"),
            file_name="all_12_cities_cross_sensor_comparison.csv",
            mime="text/csv",
        )
    else:
        st.warning("All-city Dynamic World summary file is not available.")

    st.divider()
    st.subheader("Class-wise held-out test performance")
    if not test_class_metrics.empty:
        classes = test_class_metrics.copy()
        for col in ["IoU", "F1 Score"]:
            if col in classes.columns:
                classes[col] = classes[col].apply(as_percent)

        if "F1 Score" in classes.columns:
            fig = px.bar(
                classes, x="Class", y="F1 Score", color="City", barmode="group",
                title="Class-wise F1-score across the three unseen test cities",
            )
            fig.update_layout(yaxis_title="F1-score (%)")
            st.plotly_chart(fig, use_container_width=True)
        if "IoU" in classes.columns:
            fig = px.bar(
                classes, x="Class", y="IoU", color="City", barmode="group",
                title="Class-wise IoU across the three unseen test cities",
            )
            fig.update_layout(yaxis_title="IoU (%)")
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
        'Test scores therefore measure agreement with those proxy labels, not definitive 2025 ground-truth accuracy.</div>',
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
    st.subheader(f"Monthly coverage and agreement — {selected_city or 'selected city'}")
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
    st.subheader("🌐 Interactive city-based LULC explorer")
    st.markdown("Enter a study city in the sidebar. The map opens at the centre of that city’s saved monthly prediction raster.")

    map_col1, map_col2 = st.columns([1.2, 1])
    with map_col1:
        if selected_city and location_match:
            st.markdown(f"**Selected city:** {selected_city}")
            st.caption(f"Raster centre: {latitude:.6f}, {longitude:.6f}")
        else:
            st.markdown("**Select a supported study city in the sidebar.**")
    with map_col2:
        overlay_opacity = st.slider("LULC overlay opacity", 0.2, 1.0, 0.75, 0.05, key="terrestrial_opacity")

    if not selected_city or not location_match:
        st.warning("Enter a supported study city in the sidebar and choose a month with an available prediction map.")
    else:
        map_city = selected_city
        map_month_no = selected_month_no
        map_month_name = selected_month
        terrestrial_path = prediction_path(map_city, map_month_no)

        if terrestrial_path.exists():
            try:
                with st.spinner("Preparing georeferenced LULC overlay..."):
                    encoded, west, south, east, north, valid_pixels = prepare_terrestrial_overlay(str(terrestrial_path))

                # Centre the map on the middle of the selected city's saved raster.
                centre = [float(latitude), float(longitude)]
                terrain_map = folium.Map(location=centre, zoom_start=12, tiles=None, control_scale=True)

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

                if location_match["class_id"] is not None:
                    selected_class = CLASS_NAMES[location_match["class_id"]]
                    selected_color = CLASS_COLORS[location_match["class_id"]]
                    marker_popup = f"<b>{selected_class}</b><br>Class ID: {location_match['class_id']}<br>{map_city} raster centre | {map_month_name} {YEAR}"
                    st.success(f"{map_city} raster-centre pixel class: {selected_class}")
                else:
                    marker_popup = f"NoData centre pixel<br>{map_city} | {map_month_name} {YEAR}"
                    st.warning("The centre pixel is NoData or outside the six mapped classes.")
                    selected_color = "#334155"

                folium.CircleMarker(
                    location=centre, radius=7, color="white", weight=2,
                    fill=True, fill_color=selected_color, fill_opacity=1.0,
                    tooltip="Approximate centre of selected study area", popup=folium.Popup(marker_popup, max_width=260),
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
                    'box-shadow:0 2px 9px rgba(0,0,0,.25);font-size:12px;color:#243447;min-width:175px>'
                    '<b style="font-size:13px">LULC classes</b><hr style="margin:7px 0">'
                    + legend_items + '</div>'
                )
                terrain_map.get_root().html.add_child(folium.Element(legend_html))

                st.markdown(f"**Automatically detected area:** {map_city} | **Prediction:** {map_month_name} {YEAR}")
                st.caption(f"{valid_pixels:,} valid classified pixels | The marker shows the centre of the selected raster, not an exact city-centre boundary.")
                map_state = st_folium(
                    terrain_map, height=680, use_container_width=True,
                    key=f"terrestrial_city_{map_city}_{map_month_no}",
                    returned_objects=["last_clicked"],
                )
                clicked = (map_state or {}).get("last_clicked") if isinstance(map_state, dict) else None
                if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
                    click_lat, click_lon = clicked["lat"], clicked["lng"]
                    st.info(f"Map clicked at latitude {click_lat:.6f}, longitude {click_lon:.6f}.")

                show_download(terrestrial_path, "Download original LULC GeoTIFF")
                st.caption("Basemap tiles are provided by external services and require an internet connection.")
            except Exception as error:
                st.error(f"Could not prepare terrestrial view: {type(error).__name__}: {error}")
        else:
            st.warning("The selected monthly prediction map is unavailable.")

    st.markdown(
        '<div class="research-note"><b>Research note:</b> City searches display existing georeferenced monthly prediction rasters. '
        'The marker reports the class at the raster centre only; it does not represent the entire city. The dashboard does not run fresh inference for arbitrary locations.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 7. FOOTER
# ============================================================================

st.divider()
st.caption("Sentinel-1 Land Use / Land Cover Research Project | 2025 | 12 study cities | Six-class semantic segmentation")
st.caption("Research prototype. See the README and report for data provenance, evaluation design and limitations.")
