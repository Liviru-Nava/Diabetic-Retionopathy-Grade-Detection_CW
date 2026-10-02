"""Diabetic retinopathy screening app for clinic use.

Upload a fundus photo, see every preprocessing step, get an ensemble grade with a
Grad-CAM explanation, explore the CNN architectures, and download a one-page
printable report. Photos, patient details and results live only in this browser
session's memory and are never written to disk.
"""

import base64
import re
import secrets
import zlib
from datetime import date, datetime
from html import escape

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

from dr_inference import (
    ARCHITECTURE_DISPLAY_NAMES,
    analyse_fundus_photo,
    decode_uploaded_image_to_rgb,
    load_app_configuration,
    load_ensemble_models,
    summarise_model_architecture,
)
from dr_report_pdf import build_screening_report_pdf

APP_NAME = "DRxVision"
APP_ICON_FILE = Path(__file__).resolve().parent / "assets" / "drxvision_icon_retina.png"
APP_ICON_IMAGE = Image.open(APP_ICON_FILE)
APP_ICON_DATA_URI = "data:image/png;base64," + base64.b64encode(APP_ICON_FILE.read_bytes()).decode("ascii")

st.set_page_config(page_title=f"{APP_NAME} | DR screening", page_icon=APP_ICON_IMAGE, layout="wide", initial_sidebar_state="collapsed")

CLINIC_NAME = "Diabetic Eye Screening Clinic"
MAXIMUM_CASES_KEPT_IN_SESSION = 5
SEVERITY_COLOURS = ["#2F7D5B", "#7C8B2A", "#C18A1A", "#C0582A", "#9E2A34"]
ARCHITECTURE_COLOURS = {"efficientnet_b3": "#3346A8", "efficientnet_b0": "#2F7FC1", "resnet50": "#7A4FB5"}
URGENCY_DISPLAY_TEXT = {"routine": "Routine follow-up", "refer": "Referral advised", "urgent": "Urgent referral"}
PREPROCESSING_METHOD_DISPLAY = {
    "clahe": ("CLAHE contrast", "Lighting is evened out in small patches so small spots stand out."),
    "ben_graham": ("Ben Graham method", "The blurred background is subtracted, leaving sharp details like spots and vessels."),
    "resize_only": ("No enhancement", "The model was trained on unenhanced photos, so this step leaves the photo as it is."),
}
ARCHITECTURE_DESCRIPTIONS = {
    "efficientnet_b3": "Scales depth, width and input size together. Its MBConv blocks use cheap depthwise convolutions and squeeze-and-excitation, which re-weights channels so the network focuses on the most useful features.",
    "efficientnet_b0": "The smallest EfficientNet, with the same building blocks as B3 but fewer and narrower layers. The fastest of the three.",
    "resnet50": "Built from bottleneck blocks with shortcut connections that let the signal skip layers. It learns differently from the EfficientNets, so it tends to make different mistakes, which is what makes the ensemble stronger.",
}


def build_preprocessing_steps_for_display(preprocessing_method):
    method_title, method_explanation = PREPROCESSING_METHOD_DISPLAY[preprocessing_method]
    return [
        ("original", "1. Original", "The photo exactly as it was uploaded."),
        ("cropped", "2. Border cropped", "Black space around the eye is trimmed away."),
        ("squared", "3. Padded to square", "Black padding keeps the eye round instead of stretching it."),
        ("masked", "4. Circular mask", "Everything outside the round eye is blanked."),
        ("enhanced", f"5. {method_title}", method_explanation),
        ("resized", "6. Resized", "Scaled to the fixed size the models were trained on."),
    ]


INTERFACE_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea, select {
    font-family: 'IBM Plex Sans', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
h1, h2, h3, .dr-serif { font-family: 'IBM Plex Serif', Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }
[data-testid="stAppViewContainer"] .block-container { padding-top: 1.6rem; max-width: 1260px; }
/* The side panel is no longer used: patient details and session controls sit on the main screen. */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
.dr-panel-heading { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.05rem; font-weight: 600; color: #16213E; margin-bottom: 0.2rem; }
.dr-session-note { color: #5A6878; font-size: 0.85rem; line-height: 1.4; }
.dr-clinic-badge { background: linear-gradient(135deg, #3346A8, #1E2A6E); color: #FFFFFF; border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.6rem; }
.dr-clinic-badge .dr-clinic-name { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.08rem; font-weight: 600; }
.dr-clinic-badge .dr-clinic-sub { font-size: 0.8rem; opacity: 0.85; margin-top: 0.15rem; }
.dr-hero { background: linear-gradient(120deg, #16213E 0%, #1E2A6E 45%, #3346A8 100%); color: #FFFFFF; border-radius: 16px;
    padding: 1.6rem 1.9rem; margin-bottom: 1.3rem; display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem;
    box-shadow: 0 8px 28px rgba(30, 42, 110, 0.28); }
.dr-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.55rem; font-weight: 600; font-size: 1.02rem; letter-spacing: 0.01em; color: rgba(255,255,255,0.92); }
.dr-brand img { width: 34px; height: 34px; border-radius: 9px; box-shadow: 0 2px 8px rgba(0,0,0,0.25); }
.dr-hero h1 { color: #FFFFFF; font-size: 1.95rem; margin: 0; font-weight: 600; }
.dr-hero p { margin: 0.4rem 0 0 0; color: rgba(255,255,255,0.86); max-width: 60ch; }
.dr-hero-stats { display: flex; gap: 1.6rem; }
.dr-hero-stat { text-align: right; }
.dr-hero-stat .dr-hero-value { font-size: 1.45rem; font-weight: 600; font-family: 'IBM Plex Serif', Georgia, serif; }
.dr-hero-stat .dr-hero-label { font-size: 0.75rem; color: rgba(255,255,255,0.78); }
.dr-card { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 14px; padding: 1.2rem 1.35rem; margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(27,42,58,0.04), 0 6px 20px rgba(27,42,58,0.06); }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 14px !important; border-color: #E3E9EE !important;
    box-shadow: 0 1px 2px rgba(27,42,58,0.04), 0 6px 20px rgba(27,42,58,0.06); }
.dr-ribbon { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 0.3rem 0 1.3rem 0; }
.dr-segment { border: 1px solid var(--grade-colour); border-radius: 10px; background: #FFFFFF; padding: 0.6rem 0.7rem 0.55rem 0.7rem; }
.dr-segment .dr-grade-number { font-size: 0.75rem; color: #5A6878; }
.dr-segment .dr-grade-name { font-weight: 500; color: #1B2A3A; font-size: 0.95rem; }
.dr-segment .dr-probability-track { height: 6px; background: #E4E9EE; border-radius: 3px; margin-top: 0.45rem; overflow: hidden; }
.dr-segment .dr-probability-fill { height: 100%; background: var(--grade-colour); border-radius: 3px; }
.dr-segment .dr-probability-text { font-size: 0.8rem; color: #5A6878; margin-top: 0.25rem; }
.dr-segment.dr-selected { background: var(--grade-colour); box-shadow: 0 6px 18px rgba(27,42,58,0.18); transform: translateY(-2px); }
.dr-segment.dr-selected .dr-grade-number, .dr-segment.dr-selected .dr-grade-name, .dr-segment.dr-selected .dr-probability-text { color: #FFFFFF; }
.dr-segment.dr-selected .dr-probability-track { background: rgba(255,255,255,0.35); }
.dr-segment.dr-selected .dr-probability-fill { background: #FFFFFF; }
.dr-result-grade { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 2.6rem; line-height: 1.05; font-weight: 600; margin: 0; }
.dr-result-subtitle { color: #5A6878; margin: 0.25rem 0 1rem 0; }
.dr-figures { display: flex; gap: 2.5rem; margin-bottom: 1.1rem; }
.dr-figure-label { color: #5A6878; font-size: 0.82rem; }
.dr-figure-value { font-size: 1.7rem; font-weight: 600; color: #1B2A3A; }
.dr-action { background: #F6F9FA; border-left: 5px solid var(--grade-colour); border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin-bottom: 0.9rem; }
.dr-action .dr-action-heading { font-weight: 600; color: var(--grade-colour); font-size: 0.9rem; }
.dr-action .dr-action-text { color: #1B2A3A; font-size: 1.02rem; margin-top: 0.15rem; }
.dr-action .dr-action-finding { color: #5A6878; font-size: 0.9rem; margin-top: 0.35rem; }
.dr-note { border: 1px solid #E3E9EE; background: #FFFFFF; border-radius: 8px; padding: 0.55rem 0.8rem; margin-bottom: 0.45rem; font-size: 0.9rem; color: #1B2A3A; }
.dr-note.dr-review { border-left: 4px solid #C18A1A; }
.dr-note.dr-quality { border-left: 4px solid #5A6878; }
.dr-note.dr-clear { border-left: 4px solid #2F7D5B; }
.dr-case-meta { color: #5A6878; font-size: 0.92rem; margin: -0.3rem 0 0.8rem 0; }
.dr-step-text { color: #5A6878; font-size: 0.82rem; line-height: 1.35; }
.dr-empty-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.dr-empty-step { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 12px; padding: 0.9rem 1rem; box-shadow: 0 4px 14px rgba(27,42,58,0.05); }
.dr-empty-step .dr-step-number { display: inline-block; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: #3346A8; color: #FFFFFF;
    text-align: center; line-height: 1.6rem; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem; }
.dr-empty-step .dr-empty-step-title { font-weight: 600; color: #1B2A3A; }
.dr-empty-step .dr-empty-step-text { color: #5A6878; font-size: 0.9rem; }
.dr-flow { display: flex; flex-direction: column; align-items: center; gap: 0; margin: 0.4rem 0 1rem 0; }
.dr-flow-box { background: #FFFFFF; border: 1px solid #D6DEE6; border-radius: 12px; padding: 0.7rem 1.1rem; text-align: center; min-width: 260px;
    box-shadow: 0 4px 14px rgba(27,42,58,0.06); }
.dr-flow-box .dr-flow-title { font-weight: 600; color: #1B2A3A; }
.dr-flow-box .dr-flow-text { font-size: 0.82rem; color: #5A6878; }
.dr-flow-box.dr-flow-dark { background: #1B2A3A; border-color: #1B2A3A; }
.dr-flow-box.dr-flow-dark .dr-flow-title, .dr-flow-box.dr-flow-dark .dr-flow-text { color: #FFFFFF; }
.dr-flow-connector { width: 2px; height: 22px; background: #B8C4CE; }
.dr-flow-columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; width: 100%; }
.dr-model-column { border-radius: 14px; border: 1px solid #E3E9EE; background: #FFFFFF; overflow: hidden; box-shadow: 0 4px 14px rgba(27,42,58,0.06); }
.dr-model-column-header { color: #FFFFFF; padding: 0.7rem 0.9rem; }
.dr-model-column-header .dr-model-name { font-weight: 600; font-size: 1.02rem; }
.dr-model-column-header .dr-model-meta { font-size: 0.78rem; opacity: 0.9; }
.dr-model-layers { padding: 0.7rem 0.8rem 0.8rem 0.8rem; display: flex; flex-direction: column; gap: 5px; }
.dr-layer { border-radius: 7px; padding: 0.35rem 0.6rem; font-size: 0.8rem; display: flex; justify-content: space-between; gap: 0.6rem; }
.dr-layer .dr-layer-shape { color: #5A6878; font-variant-numeric: tabular-nums; white-space: nowrap; }
@media (max-width: 760px) {
    .dr-hero { flex-direction: column; align-items: flex-start; }
    .dr-hero-stat { text-align: left; }
    .dr-ribbon, .dr-empty-steps, .dr-flow-columns { grid-template-columns: 1fr; }
    .dr-figures { gap: 1.2rem; }
}
.dr-feature-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; }
.dr-feature { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 14px; padding: 1rem 1.05rem; box-shadow: 0 4px 16px rgba(22,33,62,0.06); }
.dr-feature-icon { width: 38px; height: 38px; border-radius: 10px; background: #EEF1FB; display: flex; align-items: center; justify-content: center; margin-bottom: 0.55rem; }
.dr-feature-title { font-weight: 600; color: #16213E; margin-bottom: 0.15rem; }
.dr-feature-text { color: #5A6878; font-size: 0.88rem; line-height: 1.4; }
.dr-section-title { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.25rem; font-weight: 600; color: #16213E; margin: 1.4rem 0 0.7rem 0; }
.dr-grade-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.7rem; }
.dr-grade-card { background: #FFFFFF; border: 1px solid #E3E9EE; border-top: 5px solid var(--grade-colour); border-radius: 12px; padding: 0.8rem 0.9rem; }
.dr-grade-card .dr-grade-card-number { font-size: 0.75rem; color: #5A6878; }
.dr-grade-card .dr-grade-card-name { font-weight: 600; color: #16213E; }
.dr-grade-card .dr-grade-card-text { font-size: 0.82rem; color: #5A6878; margin-top: 0.3rem; line-height: 1.35; }
.dr-diagram-card { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 16px; padding: 0.6rem; box-shadow: 0 6px 22px rgba(22,33,62,0.07); margin-bottom: 1rem; }
.dr-ensemble-strip { display: flex; align-items: center; justify-content: center; gap: 0.7rem; flex-wrap: wrap; margin: 0.4rem 0 1.2rem 0; }
.dr-ensemble-chip { color: #FFFFFF; border-radius: 999px; padding: 0.45rem 0.95rem; font-weight: 500; font-size: 0.9rem; }
.dr-ensemble-sign { color: #5A6878; font-size: 1.2rem; font-weight: 600; }
.dr-ensemble-result { background: #16213E; color: #FFFFFF; border-radius: 12px; padding: 0.5rem 1rem; font-weight: 500; font-size: 0.9rem; }
@media (max-width: 760px) { .dr-feature-grid, .dr-grade-strip { grid-template-columns: 1fr; } }
.dr-tl-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin-bottom: 0.6rem; }
.dr-tl-step { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 12px; padding: 0.85rem 0.95rem; position: relative; }
.dr-tl-step .dr-tl-number { font-size: 0.75rem; font-weight: 600; color: #3346A8; }
.dr-tl-step .dr-tl-title { font-weight: 600; color: #16213E; margin: 0.1rem 0 0.25rem 0; }
.dr-tl-step .dr-tl-text { font-size: 0.84rem; color: #5A6878; line-height: 1.4; }
.dr-tl-step.dr-tl-phase-one { border-top: 4px solid #7C8595; }
.dr-tl-step.dr-tl-phase-two { border-top: 4px solid #E07A2F; }
.dr-timeline { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 14px; padding: 1rem 1.2rem 0.6rem 1.2rem; margin-bottom: 0.6rem; }
.dr-timeline-row { display: grid; grid-template-columns: 150px 1fr 210px; align-items: center; gap: 0.9rem; margin-bottom: 0.85rem; }
.dr-timeline-name { font-weight: 600; color: #16213E; font-size: 0.92rem; }
.dr-timeline-track { position: relative; height: 22px; background: repeating-linear-gradient(90deg, #F1F3F7 0 6px, #FFFFFF 6px 10px); border-radius: 6px; }
.dr-timeline-segment { position: absolute; top: 0; height: 100%; }
.dr-timeline-marker { position: absolute; top: -5px; width: 4px; height: 32px; background: #16213E; border-radius: 2px; }
.dr-timeline-result { font-size: 0.84rem; color: #5A6878; }
.dr-timeline-legend { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: 0.8rem; color: #5A6878; padding: 0.2rem 0 0.4rem 0; }
.dr-legend-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 0.35rem; }
.dr-ingredient-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; }
.dr-ingredient { background: #FFFFFF; border: 1px solid #E3E9EE; border-left: 4px solid #3346A8; border-radius: 10px; padding: 0.75rem 0.9rem; }
.dr-ingredient-title { font-weight: 600; color: #16213E; font-size: 0.93rem; }
.dr-ingredient-text { font-size: 0.83rem; color: #5A6878; margin-top: 0.2rem; line-height: 1.4; }
@media (max-width: 760px) { .dr-tl-steps, .dr-ingredient-grid { grid-template-columns: 1fr; } .dr-timeline-row { grid-template-columns: 1fr; } }

/* ===== Interface refinements: pill tabs, retina banner, calmer controls, referral meter ===== */
[data-testid="stAppViewContainer"] { background: radial-gradient(1100px 480px at 88% -8%, #E4EAFB 0%, rgba(228,234,251,0) 62%), #F5F7FA; }
[data-testid="stAppViewContainer"] .block-container { padding-top: 3.6rem; }
[data-testid="stAppDeployButton"] { display: none !important; }

/* Tabs become a floating pill bar. The selected tab is filled, so it is clear where you are. */
div:has(> [role="tablist"]) { border-bottom: none !important; }
[role="tablist"] { gap: 0.3rem; background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 14px; padding: 0.3rem;
    box-shadow: 0 1px 2px rgba(22,33,62,0.04), 0 8px 24px rgba(22,33,62,0.06); margin-bottom: 0.6rem; }
.react-aria-SelectionIndicator { display: none !important; }
[data-testid="stTab"] { height: 2.5rem; border-radius: 10px; padding: 0 1.05rem; background: transparent; transition: background 0.15s ease; }
[data-testid="stTab"] p { color: #5A6878; font-weight: 500; font-size: 0.95rem; }
[data-testid="stTab"]:hover { background: #EEF1FB; }
[data-testid="stTab"][data-selected="true"] { background: linear-gradient(135deg, #1E2A6E, #3346A8); box-shadow: 0 4px 14px rgba(51,70,168,0.30); }
[data-testid="stTab"][data-selected="true"] p { color: #FFFFFF; }

/* Banner: a faint retina motif and trust chips */
.dr-hero { position: relative; overflow: hidden;
    background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='460' height='460' viewBox='0 0 460 460'><g fill='none' stroke='%23ffffff' stroke-opacity='0.11'><circle cx='230' cy='230' r='215'/><circle cx='230' cy='230' r='160'/><circle cx='230' cy='230' r='105'/></g><circle cx='300' cy='200' r='30' fill='%23F2C18D' fill-opacity='0.17'/><path d='M300 200 C 240 150, 170 140, 90 165 M300 200 C 250 245, 180 285, 105 290 M300 200 C 325 140, 325 90, 290 30 M300 200 C 350 235, 395 300, 400 380' stroke='%23ffffff' stroke-opacity='0.13' stroke-width='3' fill='none'/></svg>") right 14% center / auto 170% no-repeat,
        linear-gradient(120deg, #16213E 0%, #1E2A6E 45%, #3346A8 100%); }
.dr-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.9rem; }
.dr-chip { border: 1px solid rgba(255,255,255,0.26); background: rgba(255,255,255,0.08); color: #FFFFFF; border-radius: 999px; padding: 0.18rem 0.75rem; font-size: 0.78rem; }
.dr-chip::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #6FD3A0; margin-right: 0.45rem; }
.dr-hero-stat { text-align: left; background: rgba(255,255,255,0.09); border: 1px solid rgba(255,255,255,0.16); border-radius: 12px; padding: 0.6rem 0.95rem; backdrop-filter: blur(2px); }

/* Controls */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #B8C4E8; background: #F7F9FF; border-radius: 14px; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E3E9EE; border-radius: 12px; padding: 0.7rem 0.95rem; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Serif', Georgia, serif; color: #16213E; }
[data-testid="stPopover"] button { border-radius: 999px; border: 1px solid #D6DEF3; background: #FFFFFF; box-shadow: 0 2px 8px rgba(22,33,62,0.06); }
[data-testid="stDataFrame"] { border: 1px solid #E3E9EE; border-radius: 12px; overflow: hidden; }
button[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #1E2A6E, #3346A8); border: none; border-radius: 10px; }
button[data-testid="stBaseButton-primary"]:hover { background: linear-gradient(135deg, #243285, #3D52C4); }
button[data-testid="stBaseButton-primary"]:disabled { background: #DDE3F2; color: #8793B4; box-shadow: none; }
[data-testid="stHeader"] { background: transparent; }

/* Referral meter: where the chance of referable DR sits between "not referable", "borderline" and "referable" */
.dr-meter { margin: 0.1rem 0 1rem 0; }
.dr-meter-title { font-size: 0.82rem; color: #5A6878; margin-bottom: 0.5rem; }
.dr-meter-track { position: relative; display: flex; height: 12px; }
.dr-meter-zone { height: 100%; }
.dr-meter-zone.dr-zone-low { background: #CFE6DA; border-radius: 999px 0 0 999px; width: 35%; }
.dr-meter-zone.dr-zone-mid { background: #F4E2BC; width: 30%; }
.dr-meter-zone.dr-zone-high { background: #EBC4B8; border-radius: 0 999px 999px 0; width: 35%; }
.dr-meter-marker { position: absolute; top: -5px; width: 6px; height: 22px; border-radius: 3px; background: #16213E; box-shadow: 0 0 0 3px #FFFFFF; }
.dr-meter-scale { display: grid; grid-template-columns: 35% 30% 35%; margin-top: 0.4rem; font-size: 0.74rem; color: #5A6878; text-align: center; }

/* Heat map colour key */
.dr-heat-key { height: 8px; border-radius: 4px; margin: 0.15rem 0 0.2rem 0; background: linear-gradient(90deg, #00007F, #0000FF, #00FFFF, #7FFF7F, #FFFF00, #FF0000, #7F0000); }
.dr-heat-key-labels { display: flex; justify-content: space-between; font-size: 0.74rem; color: #5A6878; margin-bottom: 0.4rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the three grading models...")
def load_configuration_and_models():
    configuration = load_app_configuration()
    return configuration, load_ensemble_models(configuration)


@st.cache_resource(show_spinner="Measuring each network's layers...")
def load_architecture_summaries(_loaded_models, image_size):
    return {name: summarise_model_architecture(model, name, image_size) for name, model in _loaded_models.items()}


FEATURE_ICON_PATHS = {
    "grade": '<path d="M4 18h16M6 14l3-4 3 2 5-6" stroke="#3346A8" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "heatmap": '<circle cx="12" cy="12" r="8" stroke="#3346A8" stroke-width="2" fill="none"/><circle cx="14" cy="10" r="3" fill="#E07A2F"/>',
    "steps": '<rect x="4" y="5" width="6" height="6" rx="1.5" stroke="#3346A8" stroke-width="2" fill="none"/><rect x="14" y="13" width="6" height="6" rx="1.5" stroke="#3346A8" stroke-width="2" fill="none"/><path d="M10 8h4v5" stroke="#3346A8" stroke-width="2" fill="none"/>',
    "report": '<rect x="6" y="3" width="12" height="18" rx="2" stroke="#3346A8" stroke-width="2" fill="none"/><path d="M9 8h6M9 12h6M9 16h4" stroke="#3346A8" stroke-width="2"/>',
}
LANDING_FEATURES = [
    ("grade", "Grade on the ICDR scale", "No DR to Proliferative, with the confidence and chance of needing a referral."),
    ("heatmap", "See where it looked", "A Grad-CAM heat map highlights the parts of the retina that drove the grade."),
    ("steps", "Every step shown", "Watch the photo being cropped, squared, masked and enhanced before grading."),
    ("report", "One-page report", "Download a printable A4 report to share with the patient or another clinician."),
]
GRADE_DESCRIPTIONS = [
    "No visible signs of damage.",
    "Tiny bulges in small vessels (microaneurysms) only.",
    "More damage: bleeds, spots and patches. Refer.",
    "Widespread bleeding and vessel damage. Urgent referral.",
    "New fragile vessels growing. Urgent referral.",
]


def render_landing_sections(configuration):
    grade_cards = "".join(
        f'<div class="dr-grade-card" style="--grade-colour:{SEVERITY_COLOURS[grade_index]}"><div class="dr-grade-card-number">Grade {grade_index}</div>'
        f'<div class="dr-grade-card-name">{escape(stage_name)}</div><div class="dr-grade-card-text">{escape(GRADE_DESCRIPTIONS[grade_index])}</div></div>'
        for grade_index, stage_name in enumerate(configuration["stage_names"])
    )
    st.markdown(f'<div class="dr-section-title">The five grades it recognises</div><div class="dr-grade-strip">{grade_cards}</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="dr-section-title">How it works</div>
        <div class="dr-empty-steps">
            <div class="dr-empty-step"><div class="dr-step-number">1</div><div class="dr-empty-step-title">Add patient details</div>
                <div class="dr-empty-step-text">Use the patient panel at the top. Every field is optional.</div></div>
            <div class="dr-empty-step"><div class="dr-step-number">2</div><div class="dr-empty-step-title">Upload and analyse</div>
                <div class="dr-empty-step-text">Three neural networks grade the photo and their answers are averaged.</div></div>
            <div class="dr-empty-step"><div class="dr-step-number">3</div><div class="dr-empty-step-title">Review and print</div>
                <div class="dr-empty-step-text">Check the result and heat map, add comments and download the report.</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def build_feature_cards_markup():
    return '<div class="dr-feature-grid">' + "".join(
        f'<div class="dr-feature"><div class="dr-feature-icon"><svg width="22" height="22" viewBox="0 0 24 24">{FEATURE_ICON_PATHS[icon]}</svg></div>'
        f'<div class="dr-feature-title">{escape(title)}</div><div class="dr-feature-text">{escape(text)}</div></div>'
        for icon, title, text in LANDING_FEATURES
    ) + "</div>"


def initialise_session_storage():
    st.session_state.setdefault("case_log", [])
    st.session_state.setdefault("active_case_id", None)
    st.session_state.setdefault("uploader_generation", 0)


def clear_session_storage():
    next_uploader_generation = st.session_state.get("uploader_generation", 0) + 1
    for session_key in list(st.session_state.keys()):
        del st.session_state[session_key]
    st.session_state["uploader_generation"] = next_uploader_generation


def find_case_by_id(case_id):
    return next((case_record for case_record in st.session_state["case_log"] if case_record["report_id"] == case_id), None)


def describe_date_of_birth(date_of_birth):
    if date_of_birth is None:
        return ""
    today = date.today()
    age_in_years = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    return f"{date_of_birth:%d %b %Y} ({age_in_years} years)"


def build_safe_file_name(case_record):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", case_record["patient_details"]["patient_name"]).strip("_")
    return f"DR_report_{safe_name + '_' if safe_name else ''}{case_record['report_id']}.pdf"


def display_patient_name(patient_details):
    return patient_details["patient_name"] or "Unnamed patient"


def render_patient_and_session_panel(configuration):
    """Patient details and session controls, shown across the top of the Screening tab.

    These used to live in the side panel. They are now on the main screen so
    everything needed for a case is visible in one place. The widget keys are
    unchanged, so collect_patient_details_snapshot still reads them as before.
    """
    with st.container(border=True):
        st.markdown(f'<div class="dr-panel-heading">Patient details, {escape(CLINIC_NAME)}</div>', unsafe_allow_html=True)
        name_column, birth_column, sex_column, doctor_column = st.columns([1.4, 1, 0.8, 1.3])
        name_column.text_input("Patient name", key="patient_name", placeholder="Optional")
        birth_column.date_input("Date of birth", value=None, min_value=date(1900, 1, 1), max_value=date.today(),
                                key="date_of_birth", format="DD/MM/YYYY", help="Optional. Pick from the calendar.")
        sex_column.selectbox("Sex", ["Not recorded", "Female", "Male", "Other"], key="sex")
        doctor_column.text_input("Referring doctor", key="referring_clinician", placeholder="Optional")

        session_column, button_column = st.columns([4, 1], vertical_alignment="center")
        with session_column:
            st.markdown(
                f'<div class="dr-session-note">{len(st.session_state["case_log"])} of {MAXIMUM_CASES_KEPT_IN_SESSION} recent cases held in memory. '
                "Nothing is saved: photos, patient details and results exist only in this browser tab and disappear when you clear the session "
                f"or close the tab. Model version {escape(configuration['model_version'])}.</div>",
                unsafe_allow_html=True,
            )
        with button_column:
            st.button("Clear session", on_click=clear_session_storage, width="stretch")


def collect_patient_details_snapshot():
    return {
        "patient_name": st.session_state.get("patient_name", "").strip(),
        "date_of_birth": describe_date_of_birth(st.session_state.get("date_of_birth")),
        "sex": "" if st.session_state.get("sex") == "Not recorded" else st.session_state.get("sex", ""),
        "referring_clinician": st.session_state.get("referring_clinician", "").strip(),
    }


def render_hero(configuration):
    test_performance = configuration.get("test_set_performance", {})
    hero_stats = [
        (f"{test_performance.get('ensemble_qwk', 0):.2f}", "Test QWK"),
        (f"{test_performance.get('referable_dr_sensitivity', 0):.0%}", "Referable cases caught"),
        ("3", "CNNs in the ensemble"),
    ]
    stats_markup = "".join(
        f'<div class="dr-hero-stat"><div class="dr-hero-value">{value}</div><div class="dr-hero-label">{label}</div></div>'
        for value, label in hero_stats
    )
    st.markdown(
        f"""
        <div class="dr-hero">
            <div>
                <div class="dr-brand"><img src="{APP_ICON_DATA_URI}" alt="{APP_NAME} logo"><span>{APP_NAME}</span></div>
                <h1>Diabetic retinopathy screening</h1>
                <p>Grades a retinal photo on the five-step international scale and shows where the model looked.
                It supports a clinician's decision and never replaces it.</p>
                <div class="dr-chips"><span class="dr-chip">Nothing is stored</span><span class="dr-chip">Runs on a CPU</span><span class="dr-chip">Model {configuration['model_version']}</span></div>
            </div>
            <div class="dr-hero-stats">{stats_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis_for_upload(uploaded_file, configuration, loaded_models):
    image_rgb = decode_uploaded_image_to_rgb(uploaded_file.getvalue())
    analysis = analyse_fundus_photo(image_rgb, loaded_models, configuration)
    analysed_at = datetime.now()
    case_record = {
        "report_id": f"DR-{analysed_at:%Y%m%d}-{secrets.token_hex(2).upper()}",
        "analysed_at_display": analysed_at.strftime("%d %b %Y, %H:%M"),
        "facility_name": CLINIC_NAME,
        "patient_details": collect_patient_details_snapshot(),
        "file_name": uploaded_file.name,
        "analysis": analysis,
        "stage_names": configuration["stage_names"],
        "target_image_size": configuration["target_image_size"],
        "test_set_performance": configuration.get("test_set_performance", {}),
        "model_version": configuration["model_version"],
        "clinician_comments": "",
    }
    st.session_state["case_log"] = ([case_record] + st.session_state["case_log"])[:MAXIMUM_CASES_KEPT_IN_SESSION]
    st.session_state["active_case_id"] = case_record["report_id"]


def render_upload_area(configuration, loaded_models):
    upload_column, guidance_column = st.columns([1.15, 1], gap="large")
    with upload_column:
        with st.container(border=True):
            st.markdown("#### Upload a fundus photo")
            uploaded_file = st.file_uploader(
                "Fundus photo (PNG or JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
                key=f"fundus_uploader_{st.session_state['uploader_generation']}",
                help="Use a colour photo of the back of the eye.",
            )
            analyse_clicked = st.button("Analyse photo", type="primary", disabled=uploaded_file is None, width="stretch")
            if analyse_clicked:
                with st.status("Analysing photo", expanded=False) as analysis_status:
                    try:
                        st.write("Cleaning the photo and running the three models")
                        run_analysis_for_upload(uploaded_file, configuration, loaded_models)
                        analysis_status.update(label="Analysis complete", state="complete")
                    except ValueError as reading_error:
                        analysis_status.update(label="Could not analyse this file", state="error")
                        st.error(str(reading_error))
    with guidance_column:
        if uploaded_file is not None:
            st.image(uploaded_file, caption=f"{uploaded_file.name}, ready to analyse", width="stretch")
        elif not st.session_state["case_log"]:
            st.markdown(build_feature_cards_markup(), unsafe_allow_html=True)


def render_severity_ribbon(stage_names, averaged_probabilities, predicted_grade):
    segment_markup = []
    for grade_index, stage_name in enumerate(stage_names):
        probability = float(averaged_probabilities[grade_index])
        selected_class = " dr-selected" if grade_index == predicted_grade else ""
        segment_markup.append(
            f'<div class="dr-segment{selected_class}" style="--grade-colour:{SEVERITY_COLOURS[grade_index]}">'
            f'<div class="dr-grade-number">Grade {grade_index}</div>'
            f'<div class="dr-grade-name">{escape(stage_name)}</div>'
            f'<div class="dr-probability-track"><div class="dr-probability-fill" style="width:{probability * 100:.1f}%"></div></div>'
            f'<div class="dr-probability-text">{probability:.0%} likely</div></div>'
        )
    st.markdown(f'<div class="dr-ribbon">{"".join(segment_markup)}</div>', unsafe_allow_html=True)


def build_referral_meter_markup(referable_probability):
    """A bar split into three zones that matches the app's own review rule: under 35% is not referable, 35 to 65% is
    borderline (flagged for review), over 65% is referable. The dark marker shows where this photo sits."""
    marker_position = max(0.0, min(1.0, referable_probability)) * 100
    return (
        '<div class="dr-meter"><div class="dr-meter-title">Referral meter</div>'
        '<div class="dr-meter-track"><span class="dr-meter-zone dr-zone-low"></span><span class="dr-meter-zone dr-zone-mid"></span>'
        '<span class="dr-meter-zone dr-zone-high"></span>'
        f'<span class="dr-meter-marker" style="left:calc({marker_position:.1f}% - 3px)"></span></div>'
        '<div class="dr-meter-scale"><span>Not referable</span><span>Borderline, review</span><span>Referable</span></div></div>'
    )


def render_result_summary(analysis):
    grade_colour = SEVERITY_COLOURS[analysis["predicted_grade"]]
    guidance = analysis["guidance"]
    referral_meter_markup = build_referral_meter_markup(analysis["referable_probability"])
    note_markup = [f'<div class="dr-note dr-review">Please review: {escape(reason)}</div>' for reason in analysis["review_reasons"]]
    note_markup += [f'<div class="dr-note dr-quality">Photo quality: {escape(warning)}</div>' for warning in analysis["quality_warnings"]]
    if not note_markup:
        note_markup = ['<div class="dr-note dr-clear">No quality problems found, and the three models agreed with good confidence.</div>']
    st.markdown(
        f"""
        <div class="dr-card">
            <p class="dr-result-grade" style="color:{grade_colour}">{escape(analysis['predicted_stage_name'])}</p>
            <p class="dr-result-subtitle">Grade {analysis['predicted_grade']} of 4 on the international (ICDR) scale</p>
            <div class="dr-figures">
                <div><div class="dr-figure-label">Confidence in this grade</div><div class="dr-figure-value">{analysis['confidence']:.0%}</div></div>
                <div><div class="dr-figure-label">Chance of referable DR (grade 2 or worse)</div><div class="dr-figure-value">{analysis['referable_probability']:.0%}</div></div>
            </div>
            {referral_meter_markup}
            <div class="dr-action" style="--grade-colour:{grade_colour}">
                <div class="dr-action-heading">{URGENCY_DISPLAY_TEXT[guidance['urgency']]}</div>
                <div class="dr-action-text">{escape(guidance['action'])}</div>
                <div class="dr-action-finding">{escape(guidance['finding'])}</div>
            </div>
            {"".join(note_markup)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preprocessing_tab(case_record):
    st.write("Every photo goes through the same steps the models saw during training. Nothing here is saved.")
    preprocessing_steps_for_display = build_preprocessing_steps_for_display(case_record["analysis"]["preprocessing_method"])
    step_columns = st.columns(len(preprocessing_steps_for_display))
    for step_column, (image_key, step_title, step_explanation) in zip(step_columns, preprocessing_steps_for_display):
        with step_column:
            st.image(case_record["analysis"]["display_images"][image_key], width="stretch")
            display_title = f"{step_title} to {case_record['target_image_size']} px" if image_key == "resized" else step_title
            st.markdown(f"**{display_title}**")
            st.markdown(f'<div class="dr-step-text">{escape(step_explanation)}</div>', unsafe_allow_html=True)
    image_width, image_height = case_record["analysis"]["image_dimensions"]
    st.caption(f"Uploaded size {image_width} x {image_height} pixels. Preview images are scaled down for display.")


def render_model_breakdown_section(case_record, configuration):
    analysis = case_record["analysis"]
    stage_names = configuration["stage_names"]
    breakdown_rows = []
    for architecture_name, probabilities in analysis["per_model_probabilities"].items():
        breakdown_rows.append({
            "Model": ARCHITECTURE_DISPLAY_NAMES.get(architecture_name, architecture_name),
            "Predicted grade": stage_names[analysis["per_model_grades"][architecture_name]],
            **{stage_name: f"{probabilities[grade_index]:.1%}" for grade_index, stage_name in enumerate(stage_names)},
        })
    breakdown_rows.append({
        "Model": "Ensemble (average of all three)",
        "Predicted grade": analysis["predicted_stage_name"],
        **{stage_name: f"{analysis['averaged_probabilities'][grade_index]:.1%}" for grade_index, stage_name in enumerate(stage_names)},
    })
    st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True, width="stretch")
    st.write(
        "The final grade is the average of three different models, which is more stable than trusting one. "
        + ("All three agreed on this photo." if analysis["all_models_agree"] else "They did not all agree on this photo, so a specialist should take a closer look.")
    )
    grad_cam_name = ARCHITECTURE_DISPLAY_NAMES.get(analysis["grad_cam_architecture"], analysis["grad_cam_architecture"])
    st.caption(
        f"The heat map comes from {grad_cam_name}, the strongest single model on the validation set. "
        f"It shows the evidence for the final grade ({analysis['predicted_stage_name']}). "
        f"On its own, {grad_cam_name} graded this photo as {stage_names[analysis['grad_cam_model_grade']]}."
    )


def render_about_content(configuration):
    """Shown inside the About pop-up at the top left of the page."""
    test_performance = configuration.get("test_set_performance", {})
    performance_items = [
        ("Agreement with graders (QWK)", test_performance.get("ensemble_qwk"), "{:.2f}"),
        ("Exact grade accuracy", test_performance.get("ensemble_accuracy"), "{:.0%}"),
        ("Referable cases caught", test_performance.get("referable_dr_sensitivity"), "{:.0%}"),
        ("Non-referable correctly cleared", test_performance.get("referable_dr_specificity"), "{:.0%}"),
    ]
    for first_item_index in (0, 2):
        for performance_column, (metric_label, metric_value, metric_format) in zip(st.columns(2), performance_items[first_item_index:first_item_index + 2]):
            performance_column.metric(metric_label, metric_format.format(metric_value) if metric_value is not None else "Not available")
    st.caption(f"Measured once on {test_performance.get('test_photos', 'held-out')} test photos that the models never saw during training.")
    st.markdown(
        """
**What it is for.** Helping a screening clinic sort photos by how urgently a patient needs an eye specialist.

**What it is not.** A diagnosis. It has not been clinically validated, and every result must be checked by a qualified professional.

**Known limits.**
- It was trained on one public dataset (APTOS 2019) from Indian clinics, so it may do worse on other cameras or populations.
- Severe is the hardest grade to name exactly, because there were few training photos of it. When it misses, it usually calls it Moderate or Proliferative, which still leads to a referral.
- It only grades diabetic retinopathy. It will not report other eye problems, such as glaucoma or macular degeneration.
- A blurred, dark or badly centred photo can give a wrong grade. The app warns about obvious problems but cannot catch them all.

**Privacy.** Photos and patient details are held in memory for this browser tab only. They are not written to disk, logged or sent anywhere else.
        """
    )


def render_report_section(case_record):
    with st.container(border=True):
        st.markdown("#### Printable report")
        case_record["clinician_comments"] = st.text_area(
            "Clinician comments for the report", value=case_record.get("clinician_comments", ""), key=f"comments_{case_record['report_id']}",
            placeholder="Optional. For example: agree with grade, refer to retina clinic within 3 months.", max_chars=400,
        )
        st.download_button(
            "Download report (PDF)", data=build_screening_report_pdf(case_record), file_name=build_safe_file_name(case_record),
            mime="application/pdf", type="primary", key=f"download_{case_record['report_id']}",
        )
        st.caption("One page, A4 landscape, ready to print or attach to the patient record. It is built in memory each time you download it.")


def render_case(case_record, configuration):
    analysis = case_record["analysis"]
    patient_details = case_record["patient_details"]
    st.markdown(f"## Result for {escape(display_patient_name(patient_details))}")
    case_facts = [patient_details.get("date_of_birth"), patient_details.get("sex"), f"analysed {case_record['analysed_at_display']}", f"report {case_record['report_id']}"]
    st.markdown(f'<p class="dr-case-meta">{escape(", ".join(fact for fact in case_facts if fact))}</p>', unsafe_allow_html=True)

    render_severity_ribbon(configuration["stage_names"], analysis["averaged_probabilities"], analysis["predicted_grade"])
    summary_column, image_column = st.columns([1, 1.15], gap="large")
    with summary_column:
        render_result_summary(analysis)
    with image_column:
        with st.container(border=True):
            original_column, heatmap_column = st.columns(2)
            cleaned_photo = analysis["display_images"]["resized"]
            heat_strength = st.session_state.get(f"heat_strength_{case_record['report_id']}", 100) / 100
            # Blend from the plain cleaned photo (0%) to the full heat map overlay (100%). Nothing is recomputed.
            blended_heat_map = cv2.addWeighted(cleaned_photo, 1 - heat_strength, analysis["display_images"]["grad_cam"], heat_strength, 0)
            original_column.image(cleaned_photo, caption="Cleaned photo the models saw", width="stretch")
            heatmap_column.image(blended_heat_map, caption="Grad-CAM: red areas influenced the grade most", width="stretch")
            st.markdown('<div class="dr-heat-key"></div><div class="dr-heat-key-labels"><span>less influence on the grade</span><span>more influence</span></div>', unsafe_allow_html=True)
            st.slider("Heat map strength", 0, 100, 100, key=f"heat_strength_{case_record['report_id']}", format="%d%%",
                      help="Slide to 0% to compare the heat map against the plain photo.")

    # Preprocessing has its own top-level tab (Image Preparation) and "About this tool" is a pop-up at the top
    # left of the page, so the result ends with one Model breakdown section and then the printable report.
    st.markdown('<div class="dr-section-title">Model breakdown</div>', unsafe_allow_html=True)
    render_model_breakdown_section(case_record, configuration)
    render_report_section(case_record)


def render_session_history():
    earlier_cases = st.session_state["case_log"]
    if len(earlier_cases) < 2:
        return
    case_labels = {
        case_record["report_id"]: f"{case_record['analysed_at_display']}, {display_patient_name(case_record['patient_details'])}, {case_record['analysis']['predicted_stage_name']}"
        for case_record in earlier_cases
    }
    st.selectbox("Cases analysed in this session", options=list(case_labels.keys()), format_func=case_labels.get, key="active_case_id",
                 help="Only the last five cases are kept, and only until the session is cleared or the tab is closed.")


STAGE_PLAIN_MEANINGS = {
    "efficientnet": [
        ("Stem", "First look", "Simple edges and colour changes"),
        ("Stage 1", "Early layers", "Fine edges, like the borders of blood vessels"),
        ("Stage 2", "Early layers", "Fine edges, like the borders of blood vessels"),
        ("Stage 3", "Middle layers", "Textures and tiny dots, like microaneurysms"),
        ("Stage 4", "Middle layers", "Textures and tiny dots, like microaneurysms"),
        ("Stage 5", "Deep layers", "Larger shapes, like bleeds and yellow exudate patches"),
        ("Stage 6", "Deep layers", "Larger shapes, like bleeds and yellow exudate patches"),
        ("Stage 7", "Deepest layers", "Whole-eye patterns that separate the grades"),
        ("Head convolution", "Deepest layers", "Whole-eye patterns that separate the grades"),
    ],
    "resnet": [
        ("Stem", "First look", "Simple edges and colour changes"),
        ("Stage 1", "Early layers", "Fine edges, like the borders of blood vessels"),
        ("Stage 2", "Middle layers", "Textures and tiny dots, like microaneurysms"),
        ("Stage 3", "Deep layers", "Larger shapes, like bleeds and yellow exudate patches"),
        ("Stage 4", "Deepest layers", "Whole-eye patterns that separate the grades"),
    ],
}
DIAGRAM_STAGE_PICKS = {"efficientnet_b3": [0, 2, 3, 5, 8], "efficientnet_b0": [0, 2, 3, 5, 8], "resnet50": [0, 1, 2, 3, 4]}
GLOSSARY_ITEMS = [
    ("Feature map", "A grid of numbers showing where one pattern appears in the photo. A layer with 40 feature maps is looking for 40 different patterns at once."),
    ("Convolution", "A small filter (for example 3 x 3 pixels) that slides across the photo and lights up wherever its pattern appears. This is the kernel shown on the input photo."),
    ("Downsampling", "Halving the width and height of the feature maps. The network sees less detail but a wider area, so deeper layers recognise bigger structures."),
    ("MBConv block", "EfficientNet's building block. It uses cheap per-channel filters plus squeeze-and-excitation, which boosts the most useful feature maps and quietens the rest."),
    ("Bottleneck residual block", "ResNet's building block. A shortcut lets the input skip past the block and be added back, which makes very deep networks easier to train."),
    ("Global average pooling", "Turns each feature map into one number (its average), so the whole photo becomes a single list of numbers."),
    ("Dropout", "During training, randomly switches off 40% of those numbers so the model cannot rely on any single one. It is switched off when the app makes predictions."),
    ("Fully connected layer", "Multiplies the list of numbers by learned weights to give one score per grade."),
    ("Softmax", "Turns the five scores into probabilities that add up to 100%."),
]


TRAINING_RUNS = {
    "efficientnet_b3": {"kept_epoch": 12, "epochs_trained": 18, "best_validation_qwk": 0.8911},
    "efficientnet_b0": {"kept_epoch": 16, "epochs_trained": 22, "best_validation_qwk": 0.8865},
    "resnet50": {"kept_epoch": 13, "epochs_trained": 19, "best_validation_qwk": 0.8967},
}
PHASE_ONE_EPOCHS = 5
MAXIMUM_EPOCHS = 30
TRAINING_VIEW_LABELS = {
    "structure": "The structure",
    "phase_1": "Training phase 1: head only",
    "phase_2": "Training phase 2: fine-tuning",
}
TRANSFER_LEARNING_STEPS = [
    ("Start from ImageNet", "Each network was first trained by its authors on 1.2 million everyday photos, so it already knows edges, textures and shapes."),
    ("Swap the last layer", "The original 1000-object output was removed and replaced with a new layer for the 5 DR grades, with 40% dropout."),
    ("Phase 1: train the head", "For 5 epochs the pretrained layers were locked and only the new layer learned, at a fast rate (0.001), so its random start could not damage them."),
    ("Phase 2: fine-tune everything", "Every layer was unlocked and trained gently (0.0001, slowly lowered on a cosine curve) so the network could adapt its features to retinal lesions."),
]
TRAINING_INGREDIENTS = [
    ("Flips and rotations", "Each training photo was randomly flipped and rotated every epoch, so the model learned the disease, not the exact photo. Chosen in experiment 2."),
    ("Label smoothing 0.1", "The model was taught to be 90% sure rather than 100%, which stops it becoming over-confident on wrong answers."),
    ("Weight decay 0.01", "Keeps the learned weights small (AdamW optimiser), so the model cannot simply memorise the training photos."),
    ("Dropout 40%", "During training, 40% of the numbers entering the new layer were randomly switched off, so no single feature could be relied on."),
    ("Cosine learning rate", "In phase 2 the learning rate was lowered smoothly along a curve, taking big steps early and tiny careful steps at the end."),
    ("Early stopping on QWK", "After every epoch the model was scored on the validation photos. The best version was kept, and training stopped after 6 epochs without improvement."),
]


def parse_feature_shape(shape_text):
    channels, height, width = [int(part) for part in shape_text.split(" x ")]
    return channels, height, width


def encode_image_as_png_base64(image_rgb, side=150):
    import base64
    import io as input_output
    from PIL import Image as PillowImage
    thumbnail = PillowImage.fromarray(image_rgb).resize((side, side))
    buffer = input_output.BytesIO()
    thumbnail.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_cnn_diagram_svg(architecture_name, summary, stage_names, input_image_rgb, model_probabilities, main_colour, input_size, uploaded_dimensions=None, training_view="structure"):
    family = "resnet" if architecture_name == "resnet50" else "efficientnet"
    meaning_lookup = {stage: (group, meaning) for stage, group, meaning in STAGE_PLAIN_MEANINGS[family]}
    picked_stages = [summary["stages"][index] for index in DIAGRAM_STAGE_PICKS[architecture_name]]
    light_fill = "#E8ECFA"
    stack_stroke = main_colour
    head_colour = main_colour
    pooling_colour = main_colour
    if training_view == "phase_1":
        light_fill, stack_stroke, head_colour, pooling_colour = "#ECEEF2", "#9AA3B2", "#E07A2F", "#9AA8C7"
    elif training_view == "phase_2":
        light_fill, stack_stroke, head_colour, pooling_colour = "#FDEBDD", "#E07A2F", "#E07A2F", "#9AA8C7"
    ink, slate, rule = "#16213E", "#5A6878", "#C9D2DB"
    parts = [
        '<svg width="1320" height="540" viewBox="0 0 1320 540" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">',
        '<rect x="0" y="0" width="1320" height="540" rx="16" fill="#FFFFFF"/>',
        f'<text x="85" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="{ink}">Input photo</text>',
    ]
    if input_image_rgb is not None:
        parts.append(f'<image x="20" y="130" width="130" height="130" href="data:image/png;base64,{encode_image_as_png_base64(input_image_rgb)}"/>')
    else:
        parts.append('<rect x="20" y="130" width="130" height="130" fill="#000"/>')
        parts.append('<circle cx="85" cy="195" r="60" fill="#B5502A"/><circle cx="105" cy="195" r="12" fill="#E8B04A" opacity="0.85"/>')
        parts.append('<path d="M100 195 C 70 160, 50 150, 35 150 M100 195 C 70 230, 50 240, 35 245 M100 195 C 120 150, 125 145, 130 140" stroke="#7A1F14" stroke-width="3" fill="none"/>')
    parts.append(f'<rect x="20" y="130" width="130" height="130" fill="none" stroke="{rule}"/>')
    parts.append('<rect x="72" y="172" width="18" height="18" fill="none" stroke="#E07A2F" stroke-width="2"/>')
    parts.append(f'<text x="85" y="330" text-anchor="middle" font-size="13" font-weight="600" fill="{ink}">Preprocessed photo</text>')
    parts.append(f'<text x="85" y="347" text-anchor="middle" font-size="11.5" fill="{slate}">3 maps (RGB), {input_size} x {input_size}</text>')
    if uploaded_dimensions is not None:
        parts.append(f'<text x="85" y="364" text-anchor="middle" font-size="11" fill="{slate}">from a {uploaded_dimensions[0]} x {uploaded_dimensions[1]} upload</text>')
    else:
        parts.append(f'<text x="85" y="364" text-anchor="middle" font-size="11" fill="{slate}">any upload size is resized</text>')

    stack_centres = [245, 385, 520, 645, 765]
    previous_anchor = (90, 181)
    previous_height = input_size
    previous_centre = 85
    for stack_index, (stage_row, centre_x) in enumerate(zip(picked_stages, stack_centres)):
        channels, height, _ = parse_feature_shape(stage_row["Output shape"])
        side = 30 + 110 * (height / 192) ** 0.5
        layer_count = max(2, min(8, int(round(np.log2(channels))) - 3))
        front_x = centre_x - side / 2 - layer_count * 2.5
        front_y = 195 - side / 2 + layer_count * 2.5
        for layer_index in range(layer_count - 1, -1, -1):
            parts.append(f'<rect x="{front_x + layer_index * 5:.1f}" y="{front_y - layer_index * 5:.1f}" width="{side:.1f}" height="{side:.1f}" '
                         f'fill="{light_fill}" stroke="{stack_stroke}" stroke-width="1"/>')
        kernel_x, kernel_y = front_x + side * 0.55, front_y + side * 0.35
        parts.append(f'<rect x="{kernel_x:.1f}" y="{kernel_y:.1f}" width="11" height="11" fill="#FFFFFF" stroke="#E07A2F" stroke-width="1.5"/>')
        parts.append(f'<line x1="{previous_anchor[0]}" y1="{previous_anchor[1]}" x2="{front_x + side * 0.5:.1f}" y2="{front_y + side * 0.5:.1f}" stroke="{slate}" stroke-dasharray="4 3"/>')
        previous_anchor = (round(kernel_x + 11, 1), round(kernel_y + 5, 1))
        group_name, meaning = meaning_lookup.get(stage_row["Stage"], ("Layers", ""))
        if previous_height is not None and height < previous_height:
            parts.append(f'<text x="{(previous_centre + centre_x) / 2:.1f}" y="92" text-anchor="middle" font-size="11" fill="{slate}">shrinks {previous_height // height}x</text>')
        previous_height = height
        previous_centre = centre_x
        parts.append(f'<text x="{centre_x}" y="330" text-anchor="middle" font-size="13" font-weight="600" fill="{ink}">{group_name}</text>')
        parts.append(f'<text x="{centre_x}" y="347" text-anchor="middle" font-size="11.5" fill="{slate}">{channels} maps, {height} x {height}</text>')
    parts.append(f'<text x="520" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="{ink}">Convolution blocks find patterns</text>')

    vector_x, vector_top, vector_height, cell_count = 905, 85, 230, 12
    parts.append(f'<line x1="{previous_anchor[0]}" y1="{previous_anchor[1]}" x2="{vector_x}" y2="{vector_top}" stroke="{slate}" stroke-dasharray="4 3"/>')
    parts.append(f'<line x1="{previous_anchor[0]}" y1="{previous_anchor[1]}" x2="{vector_x}" y2="{vector_top + vector_height}" stroke="{slate}" stroke-dasharray="4 3"/>')
    for cell_index in range(cell_count):
        cell_y = vector_top + cell_index * vector_height / cell_count
        parts.append(f'<rect x="{vector_x}" y="{cell_y:.1f}" width="18" height="{vector_height / cell_count:.1f}" fill="{pooling_colour}" stroke="#FFFFFF" stroke-width="1"/>')
    parts.append(f'<text x="914" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="{ink}">Pooling</text>')
    parts.append(f'<text x="914" y="372" text-anchor="middle" font-size="13" font-weight="600" fill="{ink}">Average pooling</text>')
    parts.append(f'<text x="914" y="389" text-anchor="middle" font-size="11.5" fill="{slate}">{summary["feature_channels"]} numbers</text>')

    node_x = 1035
    node_ys = [120 + index * 38 for index in range(len(stage_names))]
    for cell_index in range(0, cell_count, 2):
        cell_centre_y = vector_top + (cell_index + 0.5) * vector_height / cell_count
        for node_y in node_ys:
            parts.append(f'<line x1="{vector_x + 18}" y1="{cell_centre_y:.1f}" x2="{node_x}" y2="{node_y}" stroke="{rule}" stroke-width="0.8"/>')
    for node_y in node_ys:
        parts.append(f'<circle cx="{node_x}" cy="{node_y}" r="11" fill="{head_colour}"/>')
    parts.append(f'<text x="1030" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="{ink}">New classifier</text>')
    parts.append(f'<text x="1030" y="340" text-anchor="middle" font-size="13" font-weight="600" fill="{ink}">Dropout + fully</text>')
    parts.append(f'<text x="1030" y="357" text-anchor="middle" font-size="13" font-weight="600" fill="{ink}">connected layer</text>')
    parts.append(f'<text x="1030" y="374" text-anchor="middle" font-size="11.5" fill="{slate}">5 outputs</text>')

    box_x, box_width = 1095, 205
    parts.append(f'<text x="{box_x + box_width / 2}" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="{ink}">Softmax output</text>')
    parts.append(f'<rect x="{box_x}" y="{node_ys[0] - 22}" width="{box_width}" height="{node_ys[-1] - node_ys[0] + 44}" rx="10" fill="#F6F8FC" stroke="{rule}"/>')
    predicted_index = int(np.argmax(model_probabilities)) if model_probabilities is not None else None
    for grade_index, (stage_name, node_y) in enumerate(zip(stage_names, node_ys)):
        parts.append(f'<line x1="{node_x + 11}" y1="{node_y}" x2="{box_x + 8}" y2="{node_y}" stroke="{slate}" stroke-dasharray="3 3"/>')
        is_predicted = grade_index == predicted_index
        parts.append(f'<text x="{box_x + 12}" y="{node_y - 3}" font-size="12" font-weight="{600 if is_predicted else 400}" fill="{ink}">{grade_index}  {stage_name}</text>')
        parts.append(f'<rect x="{box_x + 12}" y="{node_y + 4}" width="150" height="6" rx="3" fill="#E4E9EE"/>')
        if model_probabilities is not None:
            probability = float(model_probabilities[grade_index])
            parts.append(f'<rect x="{box_x + 12}" y="{node_y + 4}" width="{max(150 * probability, 1):.1f}" height="6" rx="3" fill="{main_colour if is_predicted else "#9AA8C7"}"/>')
            parts.append(f'<text x="{box_x + box_width - 10}" y="{node_y + 10}" text-anchor="end" font-size="11.5" fill="{ink}">{probability:.0%}</text>')
    if model_probabilities is None:
        parts.append(f'<text x="{box_x + box_width / 2}" y="{node_ys[-1] + 44}" text-anchor="middle" font-size="11" fill="{slate}">Analyse a photo to see live values</text>')

    bracket_notes = {
        "structure": ("Pretrained on ImageNet, then fine-tuned on eye photos", "Added for this project", "Adds up to 100%"),
        "phase_1": ("Frozen: keeps what it learned from ImageNet", "Trained from scratch, learning rate 0.001", "Compared with the true grade"),
        "phase_2": ("Unlocked: fine-tuned gently, learning rate 0.0001", "Keeps learning, learning rate 0.0001", "Compared with the true grade"),
    }[training_view]
    for (bracket_left, bracket_right, bracket_title), bracket_note in zip([
        (20, 850, "Feature extraction"), (865, 1080, "Classification"), (1095, 1300, "Probability of each grade"),
    ], bracket_notes):
        parts.append(f'<path d="M{bracket_left} 440 V452 H{bracket_right} V440" fill="none" stroke="{main_colour}" stroke-width="1.5"/>')
        parts.append(f'<text x="{(bracket_left + bracket_right) / 2}" y="476" text-anchor="middle" font-size="13.5" font-weight="600" fill="{ink}">{bracket_title}</text>')
        parts.append(f'<text x="{(bracket_left + bracket_right) / 2}" y="495" text-anchor="middle" font-size="11.5" fill="{slate}">{bracket_note}</text>')

    def draw_status_pill(centre_x, pill_text, pill_colour, show_lock):
        pill_width = 12 * len(pill_text) * 0.55 + (40 if show_lock else 24)
        left_x = centre_x - pill_width / 2
        parts.append(f'<rect x="{left_x:.1f}" y="402" width="{pill_width:.1f}" height="26" rx="13" fill="{pill_colour}"/>')
        text_x = left_x + (32 if show_lock else 12)
        if show_lock:
            parts.append(f'<rect x="{left_x + 12:.1f}" y="413" width="12" height="9" rx="1.5" fill="#FFFFFF"/>')
            parts.append(f'<path d="M{left_x + 14.5:.1f} 413 v-3 a3.5 3.5 0 0 1 7 0 v3" stroke="#FFFFFF" stroke-width="1.8" fill="none"/>')
        parts.append(f'<text x="{text_x:.1f}" y="419.5" font-size="12" font-weight="600" fill="#FFFFFF">{pill_text}</text>')

    if training_view == "phase_1":
        parts.append(f'<rect x="440" y="10" width="440" height="26" rx="13" fill="#16213E"/>')
        parts.append('<text x="660" y="27.5" text-anchor="middle" font-size="12.5" font-weight="600" fill="#FFFFFF">Phase 1 of 2: epochs 1 to 5, only the new head learns</text>')
        draw_status_pill(435, "Frozen, weights not changed", "#7C8595", True)
        draw_status_pill(972, "Learning", "#E07A2F", False)
    elif training_view == "phase_2":
        parts.append(f'<rect x="420" y="10" width="480" height="26" rx="13" fill="#16213E"/>')
        parts.append('<text x="660" y="27.5" text-anchor="middle" font-size="12.5" font-weight="600" fill="#FFFFFF">Phase 2 of 2: from epoch 6, the whole network learns</text>')
        draw_status_pill(435, "Unlocked, learning slowly", "#E07A2F", False)
        draw_status_pill(972, "Still learning", "#E07A2F", False)
    parts.append("</svg>")
    return "".join(parts)


def build_plain_stage_table(architecture_name, summary):
    family = "resnet" if architecture_name == "resnet50" else "efficientnet"
    meaning_lookup = {stage: (group, meaning) for stage, group, meaning in STAGE_PLAIN_MEANINGS[family]}
    table_rows = []
    for stage_row in summary["stages"]:
        if stage_row["Stage"] == "Global average pooling":
            group_name, meaning = "Average pooling", "Squeezes each feature map into one number"
            output_text = f"{stage_row['Output shape']} numbers"
        elif stage_row["Stage"] == "New classifier head":
            group_name, meaning = "New classifier", "Turns those numbers into a score for each of the 5 grades"
            output_text = "5 grade scores"
        else:
            group_name, meaning = meaning_lookup.get(stage_row["Stage"], ("Layers", ""))
            channels, height, width = parse_feature_shape(stage_row["Output shape"])
            output_text = f"{channels} feature maps, each {height} x {width}"
        table_rows.append({
            "Part of the network": group_name,
            "What it typically learns": meaning,
            "What it outputs": output_text,
            "Building blocks": stage_row["What it contains"],
            "Learned weights": f"{stage_row['Parameters']:,}",
        })
    return pd.DataFrame(table_rows)


def build_training_timeline_markup():
    rows = []
    for architecture_name, run in TRAINING_RUNS.items():
        phase_one_width = PHASE_ONE_EPOCHS / MAXIMUM_EPOCHS * 100
        phase_two_width = (run["epochs_trained"] - PHASE_ONE_EPOCHS) / MAXIMUM_EPOCHS * 100
        marker_left = (run["kept_epoch"] - 0.5) / MAXIMUM_EPOCHS * 100
        rows.append(
            f'<div class="dr-timeline-row"><div class="dr-timeline-name">{escape(ARCHITECTURE_DISPLAY_NAMES.get(architecture_name, architecture_name))}</div>'
            f'<div class="dr-timeline-track">'
            f'<div class="dr-timeline-segment" style="left:0;width:{phase_one_width:.2f}%;background:#7C8595;border-radius:6px 0 0 6px"></div>'
            f'<div class="dr-timeline-segment" style="left:{phase_one_width:.2f}%;width:{phase_two_width:.2f}%;background:#E07A2F;opacity:0.85"></div>'
            f'<div class="dr-timeline-marker" style="left:calc({marker_left:.2f}% - 2px)"></div></div>'
            f'<div class="dr-timeline-result">Kept epoch {run["kept_epoch"]}, stopped at {run["epochs_trained"]}<br>'
            f'best validation QWK {run["best_validation_qwk"]:.3f}</div></div>'
        )
    legend = (
        '<div class="dr-timeline-legend">'
        '<span><span class="dr-legend-swatch" style="background:#7C8595"></span>Phase 1: head only (epochs 1 to 5)</span>'
        '<span><span class="dr-legend-swatch" style="background:#E07A2F"></span>Phase 2: whole network fine-tuned</span>'
        '<span><span class="dr-legend-swatch" style="background:#16213E"></span>Version kept (best validation QWK)</span>'
        f'<span><span class="dr-legend-swatch" style="background:#F1F3F7;border:1px solid #D6DEE6"></span>Epochs not needed (limit {MAXIMUM_EPOCHS})</span>'
        '</div>'
    )
    return f'<div class="dr-timeline">{"".join(rows)}{legend}</div>'


# ---------------------------------------------------------------------------
# How the model works tab: CNN architecture and training strategy, in summary
# ---------------------------------------------------------------------------
# Epoch-by-epoch numbers copied from the training logs printed in notebook cells 46 to 48.
# Each row is (epoch, training loss, validation loss, training accuracy, validation accuracy).
# A few mid-run epochs were not in the printed log, so the lines run straight across those gaps.
TRAINING_HISTORY = {
    "efficientnet_b3": [
        (1, 1.1214, 1.0212, 0.6512, 0.7086),
        (2, 0.9657, 0.9593, 0.7175, 0.725),
        (6, 0.8571, 0.8177, 0.7764, 0.7687),
        (7, 0.7737, 0.7831, 0.8158, 0.7905),
        (8, 0.7384, 0.7832, 0.8373, 0.8069),
        (9, 0.7152, 0.7696, 0.8416, 0.816),
        (10, 0.6753, 0.7888, 0.865, 0.7887),
        (11, 0.6557, 0.7716, 0.8798, 0.8051),
        (12, 0.6217, 0.7637, 0.8982, 0.8197),
        (13, 0.5957, 0.772, 0.9103, 0.8142),
        (14, 0.5789, 0.7736, 0.9208, 0.8179),
        (15, 0.5577, 0.774, 0.9309, 0.8215),
        (16, 0.5489, 0.7793, 0.936, 0.8197),
        (17, 0.5297, 0.7778, 0.9446, 0.8288),
        (18, 0.5169, 0.7905, 0.9532, 0.8106),
    ],
    "efficientnet_b0": [
        (1, 1.1068, 0.9837, 0.6551, 0.7213),
        (2, 0.9724, 0.9335, 0.7136, 0.7286),
        (3, 0.9323, 0.9211, 0.7382, 0.745),
        (4, 0.9159, 0.9104, 0.7347, 0.745),
        (5, 0.9207, 0.9168, 0.7405, 0.7596),
        (6, 0.8569, 0.8345, 0.7706, 0.7887),
        (7, 0.7923, 0.8107, 0.8049, 0.796),
        (8, 0.7366, 0.8107, 0.8357, 0.7978),
        (9, 0.7018, 0.7787, 0.8607, 0.8106),
        (10, 0.6682, 0.7891, 0.8677, 0.8051),
        (11, 0.656, 0.7684, 0.8712, 0.827),
        (12, 0.6383, 0.7661, 0.8814, 0.8179),
        (13, 0.6177, 0.7669, 0.8982, 0.8342),
        (16, 0.5669, 0.7816, 0.929, 0.8306),
        (17, 0.5507, 0.8265, 0.9352, 0.8179),
        (18, 0.5387, 0.8095, 0.9368, 0.816),
        (19, 0.533, 0.8053, 0.9458, 0.8215),
        (20, 0.5228, 0.7973, 0.9501, 0.816),
        (21, 0.5165, 0.8065, 0.9559, 0.8233),
        (22, 0.5105, 0.8077, 0.9571, 0.8142),
    ],
    "resnet50": [
        (1, 1.1064, 1.0448, 0.6621, 0.6958),
        (2, 0.9584, 0.9719, 0.7195, 0.7359),
        (3, 0.9132, 0.9372, 0.7452, 0.745),
        (4, 0.8848, 0.9234, 0.7577, 0.7505),
        (5, 0.8731, 0.9099, 0.7702, 0.7486),
        (6, 0.8155, 0.8312, 0.7983, 0.7723),
        (7, 0.7546, 0.7818, 0.8338, 0.8233),
        (8, 0.7219, 0.7958, 0.8478, 0.7978),
        (9, 0.6888, 0.729, 0.8584, 0.8215),
        (10, 0.6553, 0.7478, 0.8783, 0.8051),
        (11, 0.6384, 0.7395, 0.8841, 0.8106),
        (12, 0.6051, 0.7897, 0.9021, 0.8051),
        (13, 0.5919, 0.7223, 0.9091, 0.8361),
        (14, 0.5688, 0.7636, 0.9227, 0.827),
        (15, 0.5518, 0.7824, 0.9348, 0.8251),
        (16, 0.5339, 0.7919, 0.9423, 0.8051),
        (19, 0.492, 0.7755, 0.9641, 0.8288),
    ],
}

NETWORK_ROLES = {
    "efficientnet_b3": "The larger EfficientNet. It scales depth, width and input size together, so it gets strong accuracy from relatively few weights.",
    "efficientnet_b0": "The smallest and fastest. With only 2,563 training photos, fewer weights leaves less room to memorise them.",
    "resnet50": "A different design built on shortcut connections, so it makes different mistakes from the EfficientNets, which is what makes voting worthwhile. "
                "It was the best single model on validation, so it also draws the heat map.",
}
# Test-set scores (notebook cells 53 and 54) and CPU time for one photo (cell 52).
TEST_OUTCOMES = {"efficientnet_b3": (0.8715, 0.8055), "efficientnet_b0": (0.8771, 0.8218), "resnet50": (0.8735, 0.8073)}
CPU_MILLISECONDS_PER_PHOTO = {"efficientnet_b3": 124.5, "efficientnet_b0": 62.5, "resnet50": 180.4, "ensemble": 367.4}

# The six controlled experiments (notebook cells 39 to 45): what was tested, what won and why.
TRAINING_DECISIONS = [
    {"Decision": "1. Preprocessing", "Options tested": "Resize only, CLAHE, Ben Graham", "Winner": "CLAHE", "Validation QWK": 0.8741,
     "Why it won": "Best validation QWK and the best Severe recall."},
    {"Decision": "2. Augmentation", "Options tested": "None, flips + rotation, full, full + zoom out", "Winner": "Flips + rotation", "Validation QWK": 0.8718,
     "Why it won": "Within 0.005 of the full policy, so the lighter one won."},
    {"Decision": "3. Class balance", "Options tested": "Plain loss, class-weighted loss", "Winner": "Plain loss", "Validation QWK": 0.8799,
     "Why it won": "Higher validation QWK."},
    {"Decision": "4. Freezing", "Options tested": "Head only, last half, all layers in two phases, all layers from the start", "Winner": "Two phases, unfreeze all",
     "Validation QWK": 0.8799, "Why it won": "Within 0.005 of fine-tuning everything at once, but trains fewer weights early."},
    {"Decision": "5. Learning rate", "Options tested": "3e-5, 1e-4, 3e-4 for phase 2", "Winner": "1e-4", "Validation QWK": 0.8799,
     "Why it won": "Within 0.005 of 3e-4, so the default value won."},
    {"Decision": "6. Regularisation", "Options tested": "None, or label smoothing + weight decay + dropout + cosine schedule", "Winner": "Regularised",
     "Validation QWK": 0.8826, "Why it won": "Highest QWK and a smaller train to validation gap."},
]
FREEZING_EXPERIMENT_RESULTS = [
    {"Strategy": "Head only (rest frozen)", "Mean validation QWK": 0.7977, "Lowest seed": 0.7963, "Highest seed": 0.7992, "Severe recall": 0.276},
    {"Strategy": "Two phases, unfreeze last half", "Mean validation QWK": 0.8754, "Lowest seed": 0.8743, "Highest seed": 0.8765, "Severe recall": 0.466},
    {"Strategy": "Two phases, unfreeze all", "Mean validation QWK": 0.8799, "Lowest seed": 0.8784, "Highest seed": 0.8813, "Severe recall": 0.379},
    {"Strategy": "Fine-tune all from the start", "Mean validation QWK": 0.8826, "Lowest seed": 0.8786, "Highest seed": 0.8866, "Severe recall": 0.431},
]


def build_training_curve_chart(architecture_name, measure_name):
    """Training vs validation for one network. The grey band is phase 1 (only the new head learning)
    and the dashed line is the epoch that was kept, chosen by best validation QWK."""
    column_index = 1 if measure_name == "Loss" else 3
    curve_rows = []
    for record in TRAINING_HISTORY[architecture_name]:
        curve_rows.append({"Epoch": record[0], "Set": "Training", measure_name: record[column_index]})
        curve_rows.append({"Epoch": record[0], "Set": "Validation", measure_name: record[column_index + 1]})
    curve_data = pd.DataFrame(curve_rows)
    last_epoch = int(curve_data["Epoch"].max())
    kept_epoch = TRAINING_RUNS[architecture_name]["kept_epoch"]
    phase_one_band = alt.Chart(pd.DataFrame({"start": [0.5], "end": [PHASE_ONE_EPOCHS + 0.5]})).mark_rect(color="#E9ECF0", opacity=0.9).encode(x=alt.X("start:Q", title="Epoch"), x2="end:Q")
    curves = alt.Chart(curve_data).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Epoch:Q", title="Epoch", scale=alt.Scale(domain=[0.5, last_epoch + 0.5]), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{measure_name}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Set:N", scale=alt.Scale(domain=["Training", "Validation"], range=["#3346A8", "#E07A2F"]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["Epoch", "Set", alt.Tooltip(f"{measure_name}:Q", format=".3f")],
    )
    kept_line = alt.Chart(pd.DataFrame({"Epoch": [kept_epoch]})).mark_rule(strokeDash=[5, 3], color="#16213E", strokeWidth=2).encode(x=alt.X("Epoch:Q", title="Epoch"))
    return (phase_one_band + curves + kept_line).properties(height=240)


def build_outcome_table(architecture_summaries, test_performance):
    outcome_rows = []
    for architecture_name, run in TRAINING_RUNS.items():
        outcome_rows.append({
            "Model": ARCHITECTURE_DISPLAY_NAMES.get(architecture_name, architecture_name),
            "Learned weights": f"{architecture_summaries[architecture_name]['total_parameters'] / 1e6:.1f} M",
            "Best validation QWK": f"{run['best_validation_qwk']:.3f}",
            "Test QWK": f"{TEST_OUTCOMES[architecture_name][0]:.3f}",
            "Test accuracy": f"{TEST_OUTCOMES[architecture_name][1]:.1%}",
            "CPU time per photo": f"{CPU_MILLISECONDS_PER_PHOTO[architecture_name]:.0f} ms",
        })
    outcome_rows.append({
        "Model": "Ensemble (average of all three)", "Learned weights": "all three", "Best validation QWK": "not applicable",
        "Test QWK": f"{test_performance.get('ensemble_qwk', 0):.3f}", "Test accuracy": f"{test_performance.get('ensemble_accuracy', 0):.1%}",
        "CPU time per photo": f"{CPU_MILLISECONDS_PER_PHOTO['ensemble']:.0f} ms",
    })
    return pd.DataFrame(outcome_rows)


def render_architecture_page(configuration, loaded_models):
    image_size = configuration["target_image_size"]
    architecture_summaries = load_architecture_summaries(loaded_models, image_size)
    active_case = find_case_by_id(st.session_state.get("active_case_id"))
    grad_cam_name = ARCHITECTURE_DISPLAY_NAMES.get(configuration["grad_cam_architecture"], configuration["grad_cam_architecture"])
    test_performance = configuration.get("test_set_performance", {})

    # ---- Part 1: the CNN architecture -------------------------------------------------------
    st.markdown('<div class="dr-section-title">1. The CNN architecture</div>', unsafe_allow_html=True)
    st.write(
        "Three convolutional neural networks (CNNs), each pretrained on ImageNet, read every photo. The left part of each network finds patterns, "
        "starting with simple edges and building up to whole-eye patterns, while the feature maps get smaller but more numerous. The right part turns "
        "those patterns into a probability for each grade, and the app averages the three answers (soft voting)."
    )
    chips = "".join(
        f'<span class="dr-ensemble-chip" style="background:{ARCHITECTURE_COLOURS.get(name, "#3346A8")}">{escape(ARCHITECTURE_DISPLAY_NAMES.get(name, name))}</span>'
        + ('<span class="dr-ensemble-sign">+</span>' if position < len(architecture_summaries) - 1 else "")
        for position, name in enumerate(architecture_summaries)
    )
    st.markdown(
        f'<div class="dr-ensemble-strip">{chips}<span class="dr-ensemble-sign">average</span>'
        f'<span class="dr-ensemble-result">Final grade and confidence</span></div>',
        unsafe_allow_html=True,
    )
    model_column, view_column = st.columns([1, 1.3])
    with model_column:
        chosen_architecture = st.radio(
            "Model", options=list(architecture_summaries.keys()), horizontal=True,
            format_func=lambda name: ARCHITECTURE_DISPLAY_NAMES.get(name, name), key="architecture_choice",
        )
    with view_column:
        training_view = st.radio(
            "View", options=list(TRAINING_VIEW_LABELS.keys()), horizontal=True,
            format_func=TRAINING_VIEW_LABELS.get, key="training_view_choice",
        )
    chosen_summary = architecture_summaries[chosen_architecture]
    input_image = active_case["analysis"]["display_images"]["resized"] if active_case else None
    model_probabilities = active_case["analysis"]["per_model_probabilities"][chosen_architecture] if active_case else None
    uploaded_dimensions = active_case["analysis"]["image_dimensions"] if active_case else None
    diagram_svg = build_cnn_diagram_svg(chosen_architecture, chosen_summary, configuration["stage_names"], input_image, model_probabilities,
                                        ARCHITECTURE_COLOURS.get(chosen_architecture, "#3346A8"), image_size, uploaded_dimensions, training_view)
    with st.container(border=True):
        st.image(diagram_svg, width="stretch")
    view_explanations = {
        "structure": "Each stack is a set of feature maps: more sheets means more patterns, a smaller square means a coarser view.",
        "phase_1": "Grey means frozen: the ImageNet weights were locked. Orange means learning. Only the new classifier learned in phase 1.",
        "phase_2": "Everything is orange: every layer was unlocked, with a learning rate 10 times smaller than in phase 1, so the pretrained knowledge was refined, not overwritten.",
    }
    st.caption(
        ("Showing the photo you analysed and this model's own answer for it. " if active_case else "")
        + f"Every photo is preprocessed to {image_size} x {image_size} pixels with 3 colour channels before it reaches the network. "
        + view_explanations[training_view] + f" {chosen_summary['total_parameters']:,} learned weights in total."
    )
    network_rows = [{
        "Network": ARCHITECTURE_DISPLAY_NAMES.get(name, name),
        "Learned weights": f"{summary['total_parameters'] / 1e6:.1f} M",
        "Why it is in the ensemble": NETWORK_ROLES[name],
    } for name, summary in architecture_summaries.items()]
    st.dataframe(pd.DataFrame(network_rows), hide_index=True, width="stretch")
    with st.expander("Layer by layer for the selected network, in plain words"):
        st.write(ARCHITECTURE_DESCRIPTIONS.get(chosen_architecture, ""))
        st.dataframe(build_plain_stage_table(chosen_architecture, chosen_summary), hide_index=True, width="stretch")
        st.caption("Early layers pick up edges, deeper layers combine them into larger structures. This is the typical pattern for CNNs.")
    with st.expander("What do these terms mean?"):
        for term, explanation in GLOSSARY_ITEMS + [
            ("Epoch", "One full pass through all the training photos."),
            ("Learning rate", "How big a step the weights take each time they are adjusted. Smaller steps change a network more carefully."),
            ("Frozen layer", "A layer whose weights are locked, so training does not change it."),
            ("Fine-tuning", "Continuing to train a pretrained network on new data, usually with a small learning rate."),
            ("QWK", "Quadratic weighted kappa: agreement with the graders, where far-off mistakes cost much more than near misses. 1.0 is perfect."),
        ]:
            st.markdown(f"**{term}.** {explanation}")

    # ---- Part 2: the training strategy -------------------------------------------------------
    st.markdown('<div class="dr-section-title">2. The training strategy</div>', unsafe_allow_html=True)
    st.markdown("**How transfer learning was done**")
    step_classes = ["", "", " dr-tl-phase-one", " dr-tl-phase-two"]
    st.markdown(
        '<div class="dr-tl-steps">' + "".join(
            f'<div class="dr-tl-step{step_classes[step_index]}"><div class="dr-tl-number">Step {step_index + 1}</div>'
            f'<div class="dr-tl-title">{escape(title)}</div><div class="dr-tl-text">{escape(text)}</div></div>'
            for step_index, (title, text) in enumerate(TRANSFER_LEARNING_STEPS)
        ) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**What was tested, and what won**")
    st.caption("Six controlled experiments (26 training runs, two random seeds each, EfficientNet-B0 on a short schedule). If two options were within 0.005 validation QWK, the simpler one won.")
    st.dataframe(pd.DataFrame(TRAINING_DECISIONS), hide_index=True, width="stretch",
                 column_config={"Validation QWK": st.column_config.NumberColumn(format="%.4f")})
    freezing_chart_column, freezing_text_column = st.columns([1.3, 1], gap="large")
    with freezing_chart_column:
        st.markdown("**Experiment 4: how much of the network should learn?**")
        st.altair_chart(build_option_comparison_chart(FREEZING_EXPERIMENT_RESULTS, "Strategy", "Two phases, unfreeze all"), width="stretch")
    with freezing_text_column:
        st.markdown("**What this shows**")
        st.write(
            "Training only the new head reached a validation QWK of 0.798. Unlocking the whole network in phase 2 reached 0.880. "
            "ImageNet features alone are not enough for retinal lesions: the pretrained layers had to adapt to them."
        )

    st.markdown("**How each network trained**")
    st.markdown(build_training_timeline_markup(), unsafe_allow_html=True)
    st.caption(
        "Each bar is one training run, measured in epochs (full passes through the 2,563 training photos). The black line marks the epoch that scored "
        "best on the validation photos, which is the version the app uses. Training stopped after 6 epochs without improvement."
    )
    curve_choice_column, _ = st.columns([2, 1])
    with curve_choice_column:
        curve_architecture = st.radio("Training curves for", options=list(TRAINING_RUNS.keys()), horizontal=True,
                                      format_func=lambda name: ARCHITECTURE_DISPLAY_NAMES.get(name, name), key="curve_architecture_choice")
    loss_column, accuracy_column = st.columns(2, gap="large")
    with loss_column:
        st.markdown("**Loss** (lower is better)")
        st.altair_chart(build_training_curve_chart(curve_architecture, "Loss"), width="stretch")
    with accuracy_column:
        st.markdown("**Accuracy**")
        st.altair_chart(build_training_curve_chart(curve_architecture, "Accuracy"), width="stretch")
    st.caption(
        "The grey band is phase 1 and the dashed line is the kept epoch. Training loss keeps falling but validation loss flattens, which is why training "
        "stopped early and the kept version was used. Points are the epochs recorded in the notebook's printed log."
    )

    st.markdown("**What stopped the models from memorising**")
    st.markdown(
        '<div class="dr-ingredient-grid">' + "".join(
            f'<div class="dr-ingredient"><div class="dr-ingredient-title">{escape(title)}</div><div class="dr-ingredient-text">{escape(text)}</div></div>'
            for title, text in TRAINING_INGREDIENTS
        ) + "</div>",
        unsafe_allow_html=True,
    )

    # ---- Part 3: outcomes --------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">3. What came out of it</div>', unsafe_allow_html=True)
    st.dataframe(build_outcome_table(architecture_summaries, test_performance), hide_index=True, width="stretch")
    st.write(
        "The test photos were opened once, after every choice above was fixed. A small drop from validation to test is normal, because the best epoch was "
        "picked on validation. The ensemble scored higher than any single network, and McNemar's test confirms the gain is real: on the 550 test photos it "
        "fixed 28 of ResNet50's mistakes and introduced only 9 new ones (p = 0.0031). The price is about twice the CPU time, 0.37 seconds instead of 0.18 "
        f"per photo, which is still fast enough for a clinic. The heat map comes from {grad_cam_name}."
    )


# ---------------------------------------------------------------------------
# Dataset exploration tab
# ---------------------------------------------------------------------------
# These numbers were copied from the notebook outputs (cells 7, 9 and 23), so the
# app shows the same figures as the report without needing the dataset itself.

DATASET_GRADE_COUNTS = [1805, 370, 999, 193, 295]
DATASET_SPLIT_COUNTS = {
    "Training": [1263, 259, 699, 135, 207],
    "Validation": [271, 55, 150, 29, 44],
    "Test": [271, 56, 150, 29, 44],
}
SAMPLE_PHOTOS_FILE = Path(__file__).resolve().parent / "assets" / "sample_photos_by_grade.jpg"


def build_grade_count_chart(stage_names):
    """Horizontal bar chart of photos per grade, coloured with the same severity colours as the results."""
    total_photos = sum(DATASET_GRADE_COUNTS)
    chart_data = pd.DataFrame({
        "Grade": stage_names,
        "Photos": DATASET_GRADE_COUNTS,
        "Share": [count / total_photos for count in DATASET_GRADE_COUNTS],
    })
    chart_data["Label"] = chart_data.apply(lambda row: f"{row['Photos']:,} ({row['Share']:.0%})", axis=1)
    bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Photos:Q", title="Number of photos"),
        y=alt.Y("Grade:N", sort=stage_names, title=None),
        color=alt.Color("Grade:N", scale=alt.Scale(domain=stage_names, range=SEVERITY_COLOURS), legend=None),
        tooltip=["Grade", "Photos", alt.Tooltip("Share:Q", format=".1%")],
    )
    labels = bars.mark_text(align="left", dx=5, color="#16213E").encode(text="Label:N", color=alt.value("#16213E"))
    return (bars + labels).properties(height=220)


def render_dataset_tab(configuration):
    stage_names = configuration["stage_names"]
    st.markdown('<div class="dr-section-title">The APTOS 2019 dataset at a glance</div>', unsafe_allow_html=True)
    st.write(
        "The models learned from the APTOS 2019 Blindness Detection dataset on Kaggle: colour photos of the back of the eye "
        "taken in Indian eye clinics, each graded 0 to 4 on the international (ICDR) scale by clinicians."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Labelled photos", f"{sum(DATASET_GRADE_COUNTS):,}")
    summary_columns[1].metric("Grades", "5")
    summary_columns[2].metric("Biggest to smallest grade", "9.4 : 1")
    summary_columns[3].metric("Different photo sizes", "17")

    chart_column, text_column = st.columns([1.4, 1], gap="large")
    with chart_column:
        st.markdown("**Photos in each grade**")
        st.altair_chart(build_grade_count_chart(stage_names), width="stretch")
    with text_column:
        st.markdown("**What this means**")
        st.write(
            "Almost half the photos are healthy eyes, and Severe has only 193. A model could look accurate by mostly "
            "answering No DR, so the project judged models on quadratic weighted kappa (QWK) and on how well each grade is found, "
            "not on accuracy alone."
        )
        st.write(
            "Photo sizes range from 474 x 358 to 4288 x 2848 pixels, from different cameras, which is why every upload is "
            "cleaned and resized the same way before grading (see Image Preparation)."
        )

    st.markdown("**How the photos were split**")
    split_table = pd.DataFrame(DATASET_SPLIT_COUNTS, index=stage_names)
    split_table.loc["Total"] = split_table.sum()
    st.dataframe(split_table, width="stretch")
    st.caption(
        "70% of photos trained the models, 15% (validation) picked the best version and settled every experiment, and 15% (test) "
        "was used once at the very end. The split is stratified, so every part keeps the same mix of grades."
    )

    if SAMPLE_PHOTOS_FILE.exists():
        st.markdown("**Example photos from each grade**")
        st.image(str(SAMPLE_PHOTOS_FILE), caption="Raw photos before any cleaning (notebook cell 8).", width="stretch")


# ---------------------------------------------------------------------------
# Image Preparation tab: the uploaded photo step by step, and how the steps were chosen
# ---------------------------------------------------------------------------
# Experiment and measurement results copied from the notebook (cells 17, 18 to 20 and 39).

PREPROCESSING_EXPERIMENT_RESULTS = [
    {"Method": "Resize only", "Mean validation QWK": 0.8670, "Lowest seed": 0.8630, "Highest seed": 0.8709, "Severe recall": 0.638},
    {"Method": "CLAHE", "Mean validation QWK": 0.8741, "Lowest seed": 0.8612, "Highest seed": 0.8871, "Severe recall": 0.690},
    {"Method": "Ben Graham", "Mean validation QWK": 0.8719, "Lowest seed": 0.8651, "Highest seed": 0.8787, "Severe recall": 0.586},
]
LOCAL_CONTRAST_RESULTS = [
    {"Method": "Resize only", "Local contrast": 5.458},
    {"Method": "CLAHE", "Local contrast": 10.432},
    {"Method": "Ben Graham", "Local contrast": 17.729},
]
IMAGE_SIZE_RESULTS = [
    {"Image size": 224, "Detail kept (SSIM)": 0.9630, "Minutes to train all three models": 23.3, "Fits in GPU memory": "Yes"},
    {"Image size": 320, "Detail kept (SSIM)": 0.9732, "Minutes to train all three models": 47.5, "Fits in GPU memory": "Yes"},
    {"Image size": 384, "Detail kept (SSIM)": 0.9774, "Minutes to train all three models": 65.2, "Fits in GPU memory": "Yes (chosen)"},
    {"Image size": 448, "Detail kept (SSIM)": 0.9814, "Minutes to train all three models": 90.7, "Fits in GPU memory": "Yes"},
    {"Image size": 512, "Detail kept (SSIM)": 0.9843, "Minutes to train all three models": 117.0, "Fits in GPU memory": "No"},
]
STEP_REASONS = [
    ("Crop the black border", "The black space around the eye carries no information and wastes pixels once the photo is shrunk."),
    ("Pad to a square", "Stretching a widescreen photo into a square would squash the round eye and distort lesion shapes."),
    ("Circular mask", "Blanks glare, camera text and edges outside the eye, so the model cannot learn from them."),
    ("CLAHE contrast", "Evens out lighting in small tiles on the brightness channel only, so tiny red dots stand out without changing colours."),
    ("Resize to 384", "Every model needs one fixed input size, and 384 keeps almost all the detail at a manageable training cost."),
]


def build_option_comparison_chart(option_rows, option_column, chosen_option):
    """Each option as a dot (mean validation QWK of two seeds) with a line from its lowest to highest seed.
    The option the notebook chose is drawn in blue, the others in grey. Used for several experiments."""
    chart_data = pd.DataFrame(option_rows)
    option_order = [row[option_column] for row in option_rows]
    colour = alt.condition(alt.datum[option_column] == chosen_option, alt.value("#3346A8"), alt.value("#9AA8C7"))
    seed_range = alt.Chart(chart_data).mark_rule(strokeWidth=3).encode(
        x=alt.X("Lowest seed:Q", title="Validation QWK (dot = mean of 2 seeds, line = lowest to highest)", scale=alt.Scale(zero=False)),
        x2="Highest seed:Q", y=alt.Y(f"{option_column}:N", title=None, sort=option_order), color=colour,
    )
    mean_dot = alt.Chart(chart_data).mark_circle(size=160, opacity=1).encode(
        x="Mean validation QWK:Q", y=alt.Y(f"{option_column}:N", sort=option_order), color=colour,
        tooltip=[option_column, alt.Tooltip("Mean validation QWK:Q", format=".4f"), "Lowest seed", "Highest seed", "Severe recall"],
    )
    return (seed_range + mean_dot).properties(height=max(130, 56 * len(option_rows)))


def build_method_comparison_chart():
    return build_option_comparison_chart(PREPROCESSING_EXPERIMENT_RESULTS, "Method", "CLAHE")


def build_local_contrast_chart():
    chart_data = pd.DataFrame(LOCAL_CONTRAST_RESULTS)
    return alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Local contrast:Q", title="Average local contrast inside the eye"),
        y=alt.Y("Method:N", title=None, sort=[row["Method"] for row in LOCAL_CONTRAST_RESULTS]),
        color=alt.condition(alt.datum.Method == "CLAHE", alt.value("#3346A8"), alt.value("#9AA8C7")),
        tooltip=["Method", alt.Tooltip("Local contrast:Q", format=".2f")],
    ).properties(height=170)


def render_image_preparation_tab(active_case):
    st.markdown('<div class="dr-section-title">Your photo, step by step</div>', unsafe_allow_html=True)
    if active_case is None:
        st.info("Analyse a photo on the Screening tab and every preparation step for it will appear here.")
    else:
        st.caption(f"Showing {active_case['file_name']} for {display_patient_name(active_case['patient_details'])}, report {active_case['report_id']}.")
        render_preprocessing_tab(active_case)

    st.markdown('<div class="dr-section-title">Why each step is there</div>', unsafe_allow_html=True)
    for step_title, step_reason in STEP_REASONS:
        st.markdown(f"**{step_title}.** {step_reason}")

    st.markdown('<div class="dr-section-title">How the contrast method was decided</div>', unsafe_allow_html=True)
    st.write(
        "Three ways of enhancing the photo were compared under the same training recipe, with two random seeds each "
        "(EfficientNet-B0, notebook experiment 1). The one that graded best on the validation photos was kept."
    )
    experiment_column, contrast_column = st.columns(2, gap="large")
    with experiment_column:
        st.markdown("**Grading quality on validation photos**")
        st.altair_chart(build_method_comparison_chart(), width="stretch")
    with contrast_column:
        st.markdown("**How much each method sharpens small details**")
        st.altair_chart(build_local_contrast_chart(), width="stretch")
    st.write(
        "CLAHE had the highest average validation QWK (0.874), the best recall on the rare Severe grade (0.69) and the smallest gap "
        "between training and validation scores. Ben Graham sharpened details the most but graded slightly worse, which shows "
        "that more contrast alone does not mean better grading."
    )

    st.markdown('<div class="dr-section-title">How the image size was decided</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(IMAGE_SIZE_RESULTS), hide_index=True, width="stretch")
    st.caption(
        "Detail kept is the structural similarity (SSIM) to a 1000 pixel original, where 1.0 means nothing was lost. "
        "384 keeps 97.7% of the structure. Going to 512 adds very little, makes training about 1.8 times slower and no longer fits "
        "all three models in the Kaggle T4 GPU's memory."
    )


# ---------------------------------------------------------------------------
# Augmentation & Balancing tab: training variations of the uploaded photo, and how the policy was decided
# ---------------------------------------------------------------------------
# Experiment results copied from the notebook (cells 40 and 41). Every option was trained twice (seeds 42 and 7).

AUGMENTATION_EXPERIMENT_RESULTS = [
    {"Policy": "No augmentation", "What changes each epoch": "Nothing, the photo is used as it is",
     "Mean validation QWK": 0.8614, "Lowest seed": 0.8554, "Highest seed": 0.8673, "Train minus validation gap": 0.0731, "Severe recall": 0.483},
    {"Policy": "Flips + rotation", "What changes each epoch": "Left-right flip 50%, upside-down flip 50%, rotation up to 25 degrees either way",
     "Mean validation QWK": 0.8718, "Lowest seed": 0.8604, "Highest seed": 0.8832, "Train minus validation gap": 0.0240, "Severe recall": 0.603},
    {"Policy": "Full", "What changes each epoch": "As above, plus zoom in up to 10% and small brightness, contrast and saturation changes",
     "Mean validation QWK": 0.8741, "Lowest seed": 0.8612, "Highest seed": 0.8871, "Train minus validation gap": 0.0174, "Severe recall": 0.690},
    {"Policy": "Full + zoom out", "What changes each epoch": "As Full, but the zoom can also go out by 10%",
     "Mean validation QWK": 0.8670, "Lowest seed": 0.8536, "Highest seed": 0.8804, "Train minus validation gap": 0.0213, "Severe recall": 0.586},
]
CLASS_BALANCE_EXPERIMENT_RESULTS = [
    {"Loss": "Plain loss", "Mean validation QWK": 0.8799, "Lowest seed": 0.8784, "Highest seed": 0.8813, "Train minus validation gap": 0.0590, "Severe recall": 0.379},
    {"Loss": "Class-weighted loss", "Mean validation QWK": 0.8718, "Lowest seed": 0.8604, "Highest seed": 0.8832, "Train minus validation gap": 0.0240, "Severe recall": 0.603},
]
TRAINING_PHOTOS_PER_GRADE = [1263, 259, 699, 135, 207]
CLASS_WEIGHTS_IF_USED = [0.406, 1.979, 0.733, 3.797, 2.476]


def make_random_training_version(cleaned_photo_rgb, random_generator):
    """One random version of the cleaned photo, using the same three changes as the chosen
    'flips_and_rotation' policy in the notebook (cell 25): left-right flip, upside-down flip,
    and a rotation of up to 25 degrees either way. Returns the new photo and a short description."""
    new_version = cleaned_photo_rgb
    changes_made = []
    if random_generator.random() < 0.5:
        new_version = cv2.flip(new_version, 1)
        changes_made.append("mirrored")
    if random_generator.random() < 0.5:
        new_version = cv2.flip(new_version, 0)
        changes_made.append("upside down")
    rotation_degrees = float(random_generator.uniform(-25, 25))
    image_height, image_width = new_version.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), rotation_degrees, 1.0)
    new_version = cv2.warpAffine(new_version, rotation_matrix, (image_width, image_height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    changes_made.append(f"rotated {rotation_degrees:+.0f} degrees")
    return new_version, ", ".join(changes_made)


def roll_new_augmentation_versions():
    st.session_state["augmentation_roll"] = st.session_state.get("augmentation_roll", 0) + 1


def render_augmentation_tab(active_case, stage_names):
    st.markdown('<div class="dr-section-title">Your photo, with training variations</div>', unsafe_allow_html=True)
    if active_case is None:
        st.info("Analyse a photo on the Screening tab and random training versions of it will appear here.")
    else:
        st.caption(f"Showing {active_case['file_name']} for {display_patient_name(active_case['patient_details'])}, report {active_case['report_id']}.")
        cleaned_photo = active_case["analysis"]["display_images"]["resized"]
        # The seed mixes the case ID with a counter, so a case always shows the same versions until you press the button.
        random_generator = np.random.default_rng([zlib.crc32(active_case["report_id"].encode()), st.session_state.get("augmentation_roll", 0)])
        image_columns = st.columns(6)
        image_columns[0].image(cleaned_photo, caption="Cleaned photo (what the models see when grading)", width="stretch")
        for version_number, image_column in enumerate(image_columns[1:], start=1):
            augmented_photo, change_description = make_random_training_version(cleaned_photo, random_generator)
            image_column.image(augmented_photo, caption=f"Version {version_number}: {change_description}", width="stretch")
        st.button("Show new random versions", on_click=roll_new_augmentation_versions)
    st.info(
        "These versions are synthetic. During training, each photo is randomly changed every epoch, in memory only. Nothing is saved, and validation and "
        "test photos are never augmented, so every reported score comes from real, untouched photos. The photo is still the same eye with the same grade, "
        "only mirrored or turned, which changes nothing a grader would look at."
    )

    st.markdown('<div class="dr-section-title">How the augmentation policy was decided</div>', unsafe_allow_html=True)
    st.write(
        "Four policies were compared under the same training recipe, with two random seeds each (EfficientNet-B0, notebook experiment 2). "
        "Overfitting is measured as the gap between training and validation QWK, so a smaller gap means the model memorised less."
    )
    chart_column, text_column = st.columns([1.2, 1], gap="large")
    with chart_column:
        st.markdown("**Grading quality on validation photos**")
        st.altair_chart(build_option_comparison_chart(AUGMENTATION_EXPERIMENT_RESULTS, "Policy", "Flips + rotation"), width="stretch")
    with text_column:
        st.markdown("**What this shows**")
        st.write(
            "Without augmentation the training to validation gap was 0.073, about three times the 0.024 with flips and rotation, so augmentation clearly "
            "reduced memorising. The full policy scored 0.002 higher on QWK, which is inside the noise between seeds, so the lighter flips and rotation policy "
            "was chosen because it changes the photos less."
        )
    st.markdown("**What each policy changes, and what it achieved**")
    st.dataframe(
        pd.DataFrame(AUGMENTATION_EXPERIMENT_RESULTS)[["Policy", "What changes each epoch", "Mean validation QWK", "Train minus validation gap", "Severe recall"]],
        hide_index=True, width="stretch",
        column_config={"What changes each epoch": st.column_config.TextColumn(width="large"),
                       "Mean validation QWK": st.column_config.NumberColumn(format="%.4f"), "Train minus validation gap": st.column_config.NumberColumn(format="%.3f"),
                       "Severe recall": st.column_config.NumberColumn(format="%.2f")},
    )
    st.write(
        "Each change is realistic: a mirrored right eye looks like a left eye, and patients tilt their heads, so lesions mean the same thing wherever they sit. "
        "Zoom and colour changes were tested and dropped because they changed the photos more without grading better."
    )

    st.markdown('<div class="dr-section-title">Class balance</div>', unsafe_allow_html=True)
    st.write(
        "Grades are very uneven (see Dataset exploration), so the project tested whether making mistakes on rare grades cost more during training "
        "(class-weighted loss, experiment 3). The weights below were worked out from the training split only."
    )
    balance_table_column, balance_chart_column = st.columns([1, 1.2], gap="large")
    with balance_table_column:
        st.markdown("**Loss weight each grade would get**")
        st.dataframe(pd.DataFrame({"Grade": stage_names, "Training photos": TRAINING_PHOTOS_PER_GRADE, "Loss weight if used": CLASS_WEIGHTS_IF_USED}),
                     hide_index=True, width="stretch")
    with balance_chart_column:
        st.markdown("**Grading quality on validation photos**")
        st.altair_chart(build_option_comparison_chart(CLASS_BALANCE_EXPERIMENT_RESULTS, "Loss", "Plain loss"), width="stretch")
    st.write(
        "Plain loss scored higher (0.880 against 0.872), so class weights were switched off for the final models. The cost is real: Severe recall on the "
        "validation photos was 0.60 with weights and 0.38 without. Two things soften this. The stratified split keeps Severe in every set, and on the test set "
        "all 20 Severe photos the ensemble missed were called Moderate or Proliferative, which still leads to a referral."
    )


# ---------------------------------------------------------------------------
# Evaluation tab: how the ensemble did on the 550 test photos it had never seen
# ---------------------------------------------------------------------------
# Numbers copied from the notebook (cells 53 to 65). The test photos were opened once, after every choice was fixed.
# Rows are the true grade (expert label), columns are the grade the ensemble predicted (cell 56).
EVALUATION_CONFUSION_MATRIX = np.array([
    [265, 5, 1, 0, 0],
    [4, 34, 16, 0, 2],
    [2, 12, 129, 2, 5],
    [0, 0, 8, 9, 12],
    [0, 2, 13, 3, 26],
])
# Headline scores with 95% bootstrap confidence intervals for QWK (cell 54).
EVALUATION_MODEL_ROWS = [
    {"Model": "EfficientNet-B3", "QWK": 0.8715, "QWK low": 0.8390, "QWK high": 0.8981, "Accuracy": 0.8055, "Macro F1": 0.6226},
    {"Model": "EfficientNet-B0", "QWK": 0.8771, "QWK low": 0.8422, "QWK high": 0.9086, "Accuracy": 0.8218, "Macro F1": 0.6498},
    {"Model": "ResNet50", "QWK": 0.8735, "QWK low": 0.8381, "QWK high": 0.9019, "Accuracy": 0.8073, "Macro F1": 0.6310},
    {"Model": "Ensemble (soft vote)", "QWK": 0.8998, "QWK low": 0.8728, "QWK high": 0.9253, "Accuracy": 0.8418, "Macro F1": 0.6837},
]
MODEL_COLOUR_BY_NAME = {"EfficientNet-B3": "#3346A8", "EfficientNet-B0": "#2F7FC1", "ResNet50": "#7A4FB5", "Ensemble (soft vote)": "#16213E"}
# The two yes or no questions a clinic asks (cell 57): sensitivity, specificity, positive and negative predictive value, ROC AUC.
EVALUATION_SCREENING_ROWS = {
    "referable": [
        ("EfficientNet-B3", 0.8969, 0.9358, 0.9050, 0.9301, 0.9829), ("EfficientNet-B0", 0.9058, 0.9388, 0.9099, 0.9360, 0.9805),
        ("ResNet50", 0.9103, 0.9174, 0.8826, 0.9375, 0.9810), ("Ensemble (soft vote)", 0.9283, 0.9419, 0.9159, 0.9506, 0.9851),
    ],
    "any_dr": [
        ("EfficientNet-B3", 0.9606, 0.9742, 0.9745, 0.9600, 0.9974), ("EfficientNet-B0", 0.9785, 0.9779, 0.9785, 0.9779, 0.9980),
        ("ResNet50", 0.9749, 0.9742, 0.9749, 0.9742, 0.9977), ("Ensemble (soft vote)", 0.9785, 0.9779, 0.9785, 0.9779, 0.9985),
    ],
}
# Accuracy of the ensemble inside each confidence band (cell 65).
CONFIDENCE_BAND_ROWS = [
    ("under 50%", 54, 0.481), ("50 to 60%", 52, 0.712), ("60 to 70%", 54, 0.630),
    ("70 to 80%", 59, 0.763), ("80 to 90%", 86, 0.907), ("90% or more", 245, 0.992),
]


def compute_per_grade_scores(confusion_matrix):
    """Precision, recall and F1 for each grade, worked out from the confusion matrix itself."""
    correct_counts = np.diag(confusion_matrix).astype(float)
    precision = correct_counts / confusion_matrix.sum(axis=0)
    recall = correct_counts / confusion_matrix.sum(axis=1)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def build_confusion_heatmap(confusion_matrix, stage_names, show_percent):
    """Each square is shaded by the share of that true grade, so the rare grades stay readable.
    The number printed in it is either the photo count or that share."""
    heatmap_rows = []
    for true_index, row_counts in enumerate(confusion_matrix):
        for predicted_index, photo_count in enumerate(row_counts):
            share = photo_count / row_counts.sum()
            heatmap_rows.append({
                "True grade": stage_names[true_index], "Predicted grade": stage_names[predicted_index], "Photos": int(photo_count),
                "Share of true grade": float(share), "Label": f"{share:.0%}" if show_percent else str(int(photo_count)),
            })
    heatmap_data = pd.DataFrame(heatmap_rows)
    squares = alt.Chart(heatmap_data).mark_rect(stroke="#FFFFFF", strokeWidth=3, cornerRadius=6).encode(
        x=alt.X("Predicted grade:N", sort=stage_names, title="Grade the ensemble predicted", axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("True grade:N", sort=stage_names, title="True grade (expert label)"),
        color=alt.Color("Share of true grade:Q", scale=alt.Scale(domain=[0, 1], range=["#F1F4FB", "#3346A8"]), legend=None),
        tooltip=["True grade", "Predicted grade", "Photos", alt.Tooltip("Share of true grade:Q", format=".1%")],
    )
    numbers = alt.Chart(heatmap_data).mark_text(fontSize=15, fontWeight=500).encode(
        x=alt.X("Predicted grade:N", sort=stage_names), y=alt.Y("True grade:N", sort=stage_names), text="Label:N",
        color=alt.condition(alt.datum["Share of true grade"] > 0.5, alt.value("#FFFFFF"), alt.value("#16213E")),
    )
    return (squares + numbers).properties(height=340)


def build_grade_destination_chart(confusion_matrix, true_index, stage_names):
    """Where the photos of one true grade ended up, with the same severity colours used for results."""
    row_counts = confusion_matrix[true_index]
    destination_data = pd.DataFrame({
        "Predicted grade": stage_names, "Photos": row_counts.astype(int),
        "Label": [f"{count} ({count / row_counts.sum():.0%})" for count in row_counts],
    })
    bars = alt.Chart(destination_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Photos:Q", title="Number of test photos"),
        y=alt.Y("Predicted grade:N", sort=stage_names, title="Ensemble predicted"),
        color=alt.Color("Predicted grade:N", scale=alt.Scale(domain=stage_names, range=SEVERITY_COLOURS), legend=None),
        tooltip=["Predicted grade", "Photos"],
    )
    labels = bars.mark_text(align="left", dx=5, color="#16213E").encode(text="Label:N", color=alt.value("#16213E"))
    return (bars + labels).properties(height=230)


def describe_grade_destination(confusion_matrix, true_index, stage_names):
    """One plain sentence about what happened to the photos of the chosen true grade."""
    row_counts = confusion_matrix[true_index]
    total_photos = int(row_counts.sum())
    exact_count = int(row_counts[true_index])
    sentence = f"Experts graded {total_photos} test photos as {stage_names[true_index]}. The ensemble named {exact_count} of them exactly ({exact_count / total_photos:.0%}). "
    if true_index >= 2:
        referred_count = int(row_counts[2:].sum())
        sentence += (f"{referred_count} were still sent for referral (grade 2 or worse) and {total_photos - referred_count} were called grade 0 or 1, "
                     "which would have been missed.")
    else:
        cleared_count = int(row_counts[:2].sum())
        sentence += (f"{cleared_count} were correctly left as not referable and {total_photos - cleared_count} were sent for referral that was not needed.")
    return sentence


def build_per_grade_chart(precision, recall, f1_score, stage_names):
    score_rows = []
    for grade_index, stage_name in enumerate(stage_names):
        score_rows += [
            {"Grade": stage_name, "Score": "Precision", "Value": float(precision[grade_index])},
            {"Grade": stage_name, "Score": "Recall", "Value": float(recall[grade_index])},
            {"Grade": stage_name, "Score": "F1-score", "Value": float(f1_score[grade_index])},
        ]
    return alt.Chart(pd.DataFrame(score_rows)).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Grade:N", sort=stage_names, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Score:N", sort=["Precision", "Recall", "F1-score"]),
        y=alt.Y("Value:Q", title="Score (1.0 is perfect)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Score:N", sort=["Precision", "Recall", "F1-score"], scale=alt.Scale(range=["#16213E", "#3346A8", "#E07A2F"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Grade", "Score", alt.Tooltip("Value:Q", format=".3f")],
    ).properties(height=280)


def build_model_qwk_chart():
    """Each model's test QWK as a dot with its 95% confidence interval as a line."""
    qwk_data = pd.DataFrame(EVALUATION_MODEL_ROWS)
    model_order = list(qwk_data["Model"])
    colour = alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLOUR_BY_NAME), range=list(MODEL_COLOUR_BY_NAME.values())), legend=None)
    interval = alt.Chart(qwk_data).mark_rule(strokeWidth=3).encode(
        x=alt.X("QWK low:Q", title="Test QWK (dot) with 95% confidence interval (line)", scale=alt.Scale(domain=[0.83, 0.93])),
        x2="QWK high:Q", y=alt.Y("Model:N", sort=model_order, title=None, axis=alt.Axis(labelLimit=220)), color=colour)
    dots = alt.Chart(qwk_data).mark_circle(size=190, opacity=1).encode(
        x="QWK:Q", y=alt.Y("Model:N", sort=model_order), color=colour,
        tooltip=["Model", alt.Tooltip("QWK:Q", format=".4f"), alt.Tooltip("QWK low:Q", format=".4f"), alt.Tooltip("QWK high:Q", format=".4f"),
                 alt.Tooltip("Accuracy:Q", format=".3f"), alt.Tooltip("Macro F1:Q", format=".3f")])
    labels = alt.Chart(qwk_data).mark_text(dx=14, align="left", color="#16213E").encode(
        x="QWK high:Q", y=alt.Y("Model:N", sort=model_order), text=alt.Text("QWK:Q", format=".3f"))
    return (interval + dots + labels).properties(height=210)


def build_screening_chart(question_key):
    screening_data = pd.DataFrame(EVALUATION_SCREENING_ROWS[question_key], columns=["Model", "Sensitivity", "Specificity", "PPV", "NPV", "ROC AUC"])
    long_data = screening_data.melt(id_vars="Model", value_vars=["Sensitivity", "Specificity", "ROC AUC"], var_name="Measure", value_name="Score")
    # Dots, not bars: the axis starts at 0.85 so the differences are visible, and a bar on a cut axis would mislead.
    return alt.Chart(long_data).mark_circle(size=170, opacity=1).encode(
        x=alt.X("Measure:N", sort=["Sensitivity", "Specificity", "ROC AUC"], title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset=alt.XOffset("Model:N", sort=list(MODEL_COLOUR_BY_NAME)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.85, 1.0]), title="Score (axis starts at 0.85)"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLOUR_BY_NAME), range=list(MODEL_COLOUR_BY_NAME.values())), legend=alt.Legend(title=None, orient="top")),
        tooltip=["Model", "Measure", alt.Tooltip("Score:Q", format=".4f")],
    ).properties(height=270)


def build_confidence_chart():
    """Accuracy inside each confidence band. Bands below the app's 60% review line are drawn in warm colours."""
    band_data = pd.DataFrame(CONFIDENCE_BAND_ROWS, columns=["Confidence", "Photos", "Accuracy"])
    band_data["Colour"] = band_data["Confidence"].map({"under 50%": SEVERITY_COLOURS[3], "50 to 60%": SEVERITY_COLOURS[2]}).fillna("#3346A8")
    bars = alt.Chart(band_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Confidence:N", sort=list(band_data["Confidence"]), title="How sure the ensemble was", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Share of photos graded correctly"),
        color=alt.Color("Colour:N", scale=None), tooltip=["Confidence", "Photos", alt.Tooltip("Accuracy:Q", format=".1%")],
    )
    photo_counts = bars.mark_text(dy=-8, color="#16213E").encode(text=alt.Text("Photos:Q", format="d"), color=alt.value("#16213E"))
    return (bars + photo_counts).properties(height=260)


def render_evaluation_tab(configuration):
    stage_names = configuration["stage_names"]
    confusion_matrix = EVALUATION_CONFUSION_MATRIX
    precision, recall, f1_score = compute_per_grade_scores(confusion_matrix)
    test_performance = configuration.get("test_set_performance", {})

    # ---- Headline ---------------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">How well does the ensemble grade?</div>', unsafe_allow_html=True)
    st.write(
        "Everything on this tab comes from the 550 test photos, which the models never saw during training or tuning and which were opened once, "
        "after every choice was fixed. Accuracy and loss curves are on the How the model works tab."
    )
    headline_columns = st.columns(5)
    headline_columns[0].metric("Agreement (QWK)", f"{test_performance.get('ensemble_qwk', 0):.3f}", help="Agreement with the expert graders. 1.0 is perfect. Far-off mistakes cost much more than near misses.")
    headline_columns[1].metric("Exact grade accuracy", f"{test_performance.get('ensemble_accuracy', 0):.1%}")
    headline_columns[2].metric("Macro F1", f"{test_performance.get('ensemble_macro_f1', 0):.3f}", help="F1 averaged with every grade counting equally. Lower than accuracy because the rare grades are harder.")
    headline_columns[3].metric("Referable caught", f"{test_performance.get('referable_dr_sensitivity', 0):.1%}")
    headline_columns[4].metric("Non-referable cleared", f"{test_performance.get('referable_dr_specificity', 0):.1%}")
    st.caption("QWK 0.900 has a 95% confidence interval of 0.873 to 0.925, from resampling the test set. Above 0.80 is usually read as very strong agreement.")

    # ---- Confusion matrix ----------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">Where does it get things right and wrong?</div>', unsafe_allow_html=True)
    matrix_column, reading_column = st.columns([1.35, 1], gap="large")
    with matrix_column:
        show_as = st.radio("Show each square as", ["Photo count", "Percent of the true grade"], horizontal=True, key="evaluation_matrix_view")
        st.altair_chart(build_confusion_heatmap(confusion_matrix, stage_names, show_as != "Photo count"), width="stretch")
    with reading_column:
        st.markdown("**How to read it**")
        st.write(
            "Rows are what the expert graders said, columns are what the ensemble said, so the diagonal is every correct answer. "
            "Almost all mistakes sit right next to the diagonal: the ensemble is usually one grade off, rarely far off."
        )
        st.write(
            "Only 1 of 271 healthy eyes was called referable, and no Severe or Proliferative eye was called healthy. "
            "The weak spot is Severe: of 29 photos, 9 were named exactly and 20 were called Moderate or Proliferative."
        )

    st.markdown("**Explore one grade**")
    chosen_grade = st.radio("Pick a true grade and see where its photos went", options=list(range(len(stage_names))), horizontal=True,
                            format_func=lambda grade_index: stage_names[grade_index], key="evaluation_grade_explorer")
    destination_column, sentence_column = st.columns([1.4, 1], gap="large")
    with destination_column:
        st.altair_chart(build_grade_destination_chart(confusion_matrix, chosen_grade, stage_names), width="stretch")
    with sentence_column:
        st.markdown("**What happened**")
        st.write(describe_grade_destination(confusion_matrix, chosen_grade, stage_names))

    # ---- Precision, recall, F1 -------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">Precision, recall and F1 for every grade</div>', unsafe_allow_html=True)
    chart_column, table_column = st.columns([1.4, 1], gap="large")
    with chart_column:
        st.altair_chart(build_per_grade_chart(precision, recall, f1_score, stage_names), width="stretch")
    with table_column:
        st.dataframe(pd.DataFrame({
            "Grade": stage_names, "Precision": precision.round(3), "Recall": recall.round(3), "F1-score": f1_score.round(3),
            "Test photos": confusion_matrix.sum(axis=1),
        }), hide_index=True, width="stretch")
    st.write(
        "Precision asks: when the model names this grade, how often is it right? Recall asks: of all photos truly in this grade, how many did it find? "
        "F1 balances the two. No DR is almost perfect (0.98) and Moderate is strong (0.81). Severe has decent precision (0.64) but low recall (0.31): it is the "
        "rarest grade and it sits between two neighbours. Weighting the loss towards it helped Severe recall in the experiments, but lowered overall agreement, so it was not used."
    )

    # ---- Screening view ---------------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">The question a clinic actually asks</div>', unsafe_allow_html=True)
    question_key = st.radio("Question", options=["referable", "any_dr"], horizontal=True, key="evaluation_screening_question",
                            format_func={"referable": "Does this patient need a specialist? (grade 2 or worse)", "any_dr": "Is there any DR at all? (grade 1 or worse)"}.get)
    screening_chart_column, plain_numbers_column = st.columns([1.4, 1], gap="large")
    with screening_chart_column:
        st.altair_chart(build_screening_chart(question_key), width="stretch")
    with plain_numbers_column:
        st.markdown("**In plain numbers (ensemble)**")
        if question_key == "referable":
            referable_total, referable_caught = int(confusion_matrix[2:].sum()), int(confusion_matrix[2:, 2:].sum())
            other_total, other_cleared = int(confusion_matrix[:2].sum()), int(confusion_matrix[:2, :2].sum())
            st.write(
                f"Of {referable_total} patients who needed a specialist, {referable_caught} were flagged and {referable_total - referable_caught} were missed. "
                f"Of {other_total} who did not, {other_cleared} were correctly cleared and {other_total - other_cleared} were sent for a referral that was not needed."
            )
        else:
            st.write("Separating healthy eyes from any sign of DR is the easiest question: the ensemble scores 0.98 on both sensitivity and specificity.")
        st.caption("Sensitivity is the share of patients with the condition that it finds. Specificity is the share without it that it correctly leaves alone.")
    st.write(
        "Missing a patient who needs a specialist is the costlier mistake, which is why the app also flags borderline and low-confidence photos for a person to check. "
        "Every model scores above 0.98 ROC AUC for referable DR, so the ensemble's advantage here is small but consistent."
    )

    # ---- Models against the ensemble ------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">Is the ensemble really better than one network?</div>', unsafe_allow_html=True)
    comparison_column, test_column = st.columns([1.3, 1], gap="large")
    with comparison_column:
        st.altair_chart(build_model_qwk_chart(), width="stretch")
    with test_column:
        st.markdown("**McNemar's test**")
        mcnemar_columns = st.columns(3)
        mcnemar_columns[0].metric("Ensemble fixed", "28")
        mcnemar_columns[1].metric("Ensemble broke", "9")
        mcnemar_columns[2].metric("p-value", "0.0031")
    st.write(
        "The confidence intervals overlap, so they cannot prove the ensemble is better on their own. McNemar's test looks only at the photos where the ensemble and "
        "ResNet50 (the best single model on validation, picked before the test set was opened) disagree: the ensemble fixed 28 of its mistakes and introduced only 9 new ones. "
        "A split that uneven would happen by chance about 3 times in 1,000, so the gain is real."
    )

    # ---- Does it know when it is unsure -----------------------------------------------------------------
    st.markdown('<div class="dr-section-title">Does it know when it is unsure?</div>', unsafe_allow_html=True)
    confidence_column, confidence_text_column = st.columns([1.3, 1], gap="large")
    with confidence_column:
        st.altair_chart(build_confidence_chart(), width="stretch")
    with confidence_text_column:
        st.markdown("**Why the app flags some photos**")
        st.write(
            "When the ensemble is at least 90% sure (245 photos), it is right 99% of the time. Under 60% sure, accuracy drops to about 60%. "
            "The app flags a photo when confidence is under 60% or the three networks disagree. That flags 19% of photos and catches 53% of all mistakes."
        )
        st.caption("When all three networks agree (84% of photos) the ensemble is right 88% of the time. When they disagree, 64%. Numbers above the bars are test photos in each band.")

    # ---- Honest summary ---------------------------------------------------------------------------------
    st.markdown('<div class="dr-section-title">The honest summary</div>', unsafe_allow_html=True)
    strengths_column, limits_column = st.columns(2, gap="large")
    with strengths_column:
        st.markdown("**Strong**")
        st.markdown(
            "- Very strong agreement with expert graders (QWK 0.90).\n"
            "- Healthy eyes are recognised almost perfectly, and none of the worst cases were called healthy.\n"
            "- It catches 93% of patients who need a specialist and clears 94% who do not.\n"
            "- Its confidence is meaningful, so flagged photos really are the riskier ones."
        )
    with limits_column:
        st.markdown("**Limits**")
        st.markdown(
            "- Severe is often named as Moderate or Proliferative, so the exact grade is unreliable there.\n"
            "- Mild is often called Moderate (16 of 56 photos), the most common single mix-up.\n"
            "- One dataset, one country, 550 test photos, so intervals are wide and other cameras are untested.\n"
            "- Not clinically validated, so every result needs a qualified professional."
        )


def main():
    initialise_session_storage()
    st.markdown(INTERFACE_STYLES, unsafe_allow_html=True)
    try:
        configuration, loaded_models = load_configuration_and_models()
    except (FileNotFoundError, KeyError) as loading_error:
        st.error(f"The models could not be loaded. {loading_error}")
        st.stop()

    # "About this tool" lives in a pop-up at the top left, so it is one click away on every tab without taking space.
    with st.popover("About this tool", icon=":material/info:"):
        render_about_content(configuration)
    render_hero(configuration)
    screening_tab, dataset_tab, preparation_tab, augmentation_tab, architecture_tab, evaluation_tab = st.tabs(
        ["Screening", "Dataset exploration", "Image Preparation", "Augmentation & Balancing", "How the model works", "Evaluation"])
    with screening_tab:
        render_patient_and_session_panel(configuration)
        render_upload_area(configuration, loaded_models)
        render_session_history()
        active_case = find_case_by_id(st.session_state.get("active_case_id"))
        if active_case is not None:
            st.divider()
            render_case(active_case, configuration)
        else:
            render_landing_sections(configuration)
    with dataset_tab:
        render_dataset_tab(configuration)
    with preparation_tab:
        render_image_preparation_tab(find_case_by_id(st.session_state.get("active_case_id")))
    with augmentation_tab:
        render_augmentation_tab(find_case_by_id(st.session_state.get("active_case_id")), configuration["stage_names"])
    with architecture_tab:
        render_architecture_page(configuration, loaded_models)
    with evaluation_tab:
        render_evaluation_tab(configuration)


main()