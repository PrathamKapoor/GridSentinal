"""
GridSentinal Presentation Generator — Enhanced Version
Yuva Yodha Energy Tech Hackathon 2026 — Schneider Electric
Creates 11 professional, competition-ready widescreen slides.
Incorporates native tables, vector energy-flow diagrams, recolored assets, and zero text overflow.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- COLOR PALETTE (Schneider Electric & Industrial Command Center) ---
BG_COLOR = RGBColor(0x06, 0x0A, 0x0F)         # Deep midnight graphite
CARD_BG = RGBColor(0x0D, 0x15, 0x20)          # Tech card fill
CARD_BORDER = RGBColor(0x1E, 0x2D, 0x3D)      # Card border subtle
CARD_BG_ALT = RGBColor(0x09, 0x12, 0x1C)      # Slightly darker card

TEXT_PRIMARY = RGBColor(0xF8, 0xFA, 0xFC)     # Crisp white
TEXT_SECONDARY = RGBColor(0x94, 0xA3, 0xB8)   # Slate gray
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)       # Dark slate

ACCENT_CYAN = RGBColor(0x00, 0xE5, 0xFF)      # Electric cyan
ACCENT_GREEN = RGBColor(0x00, 0xD3, 0x82)     # Schneider energy green
ACCENT_AMBER = RGBColor(0xF5, 0x9E, 0x0B)     # Warning amber
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)       # Negative / reject red
ACCENT_BLUE = RGBColor(0x38, 0xBD, 0xF8)      # Sky blue

TABLE_HEADER_BG = RGBColor(0x11, 0x24, 0x33)  # Table header
TABLE_ROW_ALT = RGBColor(0x0A, 0x16, 0x22)    # Table row alt

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"
FONT_MONO = "Consolas"

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, category, title, subtitle):
        # Top category label
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(6.0), Inches(0.32))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.name = FONT_MONO
        p_cat.font.size = Pt(9)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_GREEN

        # Top right Hackathon tag
        tag_box = slide.shapes.add_textbox(Inches(7.0), Inches(0.36), Inches(5.533), Inches(0.32))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_right = tf_tag.margin_top = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.alignment = PP_ALIGN.RIGHT
        p_tag.text = "YUVA YODHA ENERGY TECH 2026  ◆  SCHNEIDER ELECTRIC"
        p_tag.font.name = FONT_MONO
        p_tag.font.size = Pt(8.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = ACCENT_CYAN

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.70), Inches(11.733), Inches(0.60))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_PRIMARY

        # Subtitle
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.30), Inches(11.733), Inches(0.40))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_right = tf_sub.margin_top = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(11.5)
        p_sub.font.color.rgb = TEXT_SECONDARY

    def add_footer(slide, slide_num, total_slides=11):
        # Divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.012))
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.fill.background()

        # Footer text left
        fb_l = slide.shapes.add_textbox(Inches(0.8), Inches(7.12), Inches(5.0), Inches(0.28))
        tf_l = fb_l.text_frame
        tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0
        p_l = tf_l.paragraphs[0]
        p_l.text = "GRIDSENTINAL  |  Adaptive, Self-Verifying Energy Intelligence"
        p_l.font.name = FONT_MONO
        p_l.font.size = Pt(8.5)
        p_l.font.color.rgb = TEXT_MUTED

        # Footer text center
        fb_c = slide.shapes.add_textbox(Inches(4.8), Inches(7.12), Inches(5.5), Inches(0.28))
        tf_c = fb_c.text_frame
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
        p_c = tf_c.paragraphs[0]
        p_c.alignment = PP_ALIGN.CENTER
        p_c.text = "Challenge: Grid Reliability  |  The Unsupervised Suspects"
        p_c.font.name = FONT_MONO
        p_c.font.size = Pt(8.5)
        p_c.font.color.rgb = TEXT_MUTED

        # Footer text right (slide number)
        fb_r = slide.shapes.add_textbox(Inches(10.533), Inches(7.12), Inches(2.0), Inches(0.28))
        tf_r = fb_r.text_frame
        tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = 0
        p_r = tf_r.paragraphs[0]
        p_r.alignment = PP_ALIGN.RIGHT
        p_r.text = f"‹ {slide_num:02d} / {total_slides:02d} ›"
        p_r.font.name = FONT_MONO
        p_r.font.size = Pt(8.5)
        p_r.font.bold = True
        p_r.font.color.rgb = ACCENT_GREEN

    def add_card(slide, left, top, width, height, bg=CARD_BG, border=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg
        card.line.color.rgb = border
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: THE GRID PROBLEM
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1)
    add_header(s1, "01 / The Context & Problem", "THE GRID IS BECOMING LESS PREDICTABLE.",
               "Renewables fluctuate. Demand moves. Flexibility is uncertain. Every intervention has consequences.")
    add_footer(s1, 1)

    # Left Card: Detailed Breakdown + Visual Flow Diagram
    add_card(s1, Inches(0.8), Inches(1.85), Inches(5.8), Inches(5.0))
    
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.98), Inches(5.4), Inches(2.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "THE INTERACTION OF FORCES"
    p.font.name = FONT_MONO
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    p = tf1.add_paragraph()
    p.space_before = Pt(6)
    p.text = "Modern distribution networks operate under simultaneous non-stationary dynamics:"
    p.font.name = FONT_BODY
    p.font.size = Pt(9.5)
    p.font.color.rgb = TEXT_SECONDARY
    
    forces = [
        ("Renewable Intermittency", "Feeder solar & wind swing by MWs within minutes. Without weather forecasts, 24h PV prediction error spikes to 74.3 kW (R²=0.62)."),
        ("Variable Demand", "1,871 residential & commercial loads shift concurrently. Peak evening ramps concentrate 18.5x more prediction error than calm hours."),
        ("Uncertain Flexibility", "Operators balance grids using assumed flexibility, yet historical consumption variation carries zero physical control guarantees.")
    ]
    for f_name, f_desc in forces:
        p = tf1.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {f_name}: {f_desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.8)
        p.font.color.rgb = TEXT_PRIMARY

    # Visual Flow Box inside Left Card (Prompt Section 6 Requirement)
    add_card(s1, Inches(1.0), Inches(4.35), Inches(5.4), Inches(2.35), bg=CARD_BG_ALT, border=ACCENT_CYAN)
    tb_diag = s1.shapes.add_textbox(Inches(1.1), Inches(4.42), Inches(5.2), Inches(2.2))
    tf_diag = tb_diag.text_frame
    tf_diag.word_wrap = True
    
    p = tf_diag.paragraphs[0]
    p.text = "ENERGY SYSTEM VULNERABILITY COUPLING:"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    # Mini Flow boxes inside the diagram container
    flow_steps = [
        (Inches(1.15), Inches(4.75), Inches(2.2), Inches(0.38), "RENEWABLES (Intermittent)", ACCENT_GREEN),
        (Inches(1.15), Inches(5.25), Inches(2.2), Inches(0.38), "DEMAND (Volatile Shifts)", ACCENT_CYAN),
        (Inches(1.15), Inches(5.75), Inches(2.2), Inches(0.38), "DISTRIBUTED ASSETS (DERs)", ACCENT_AMBER),
        (Inches(3.75), Inches(4.85), Inches(2.4), Inches(0.48), "GRID STATE & UNCERTAINTY", ACCENT_BLUE),
        (Inches(3.75), Inches(5.55), Inches(2.4), Inches(0.48), "DECISION & CONSEQUENCE", ACCENT_RED),
    ]
    for fx, fy, fw, fh, ftxt, fcol in flow_steps:
        bx = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, fx, fy, fw, fh)
        bx.fill.solid()
        bx.fill.fore_color.rgb = RGBColor(0x13, 0x1F, 0x2C)
        bx.line.color.rgb = fcol
        bx.line.width = Pt(1)
        tf_b = bx.text_frame
        tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_right = tf_b.margin_top = tf_b.margin_bottom = 0
        p = tf_b.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = ftxt
        p.font.name = FONT_MONO
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = TEXT_PRIMARY

    # Connector arrow text
    ar1 = s1.shapes.add_textbox(Inches(3.4), Inches(5.15), Inches(0.35), Inches(0.5))
    tf_ar1 = ar1.text_frame
    p = tf_ar1.paragraphs[0]
    p.text = "➔"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.color.rgb = ACCENT_CYAN

    # Right side: 3 cards showing Real Feeder Scale & Consequences
    # Card 1: The Modelled Reality
    add_card(s1, Inches(6.8), Inches(1.85), Inches(5.733), Inches(1.5))
    tb_r1 = s1.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.333), Inches(1.3))
    tf_r1 = tb_r1.text_frame
    tf_r1.word_wrap = True
    p = tf_r1.paragraphs[0]
    p.text = "REALITY BENCHMARK: SMART-DS V1.0 FEEDER"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p = tf_r1.add_paragraph()
    p.space_before = Pt(4)
    p.text = "5,196 Nodes  |  1,871 Customer Loads  |  1,216 PV Arrays  |  93 Batteries"
    p.font.name = FONT_MONO
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = TEXT_PRIMARY
    p = tf_r1.add_paragraph()
    p.space_before = Pt(4)
    p.text = "Evaluated on 35,040 native 15-minute intervals across real geographic distribution topology. No synthetic data, no artificial smoothing."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_SECONDARY

    # Card 2: Technical Line Loss
    add_card(s1, Inches(6.8), Inches(3.5), Inches(5.733), Inches(1.4))
    tb_r2 = s1.shapes.add_textbox(Inches(7.0), Inches(3.6), Inches(5.333), Inches(1.2))
    tf_r2 = tb_r2.text_frame
    tf_r2.word_wrap = True
    p = tf_r2.paragraphs[0]
    p.text = "UNMODELLED PHYSICS: 3.22% TECHNICAL LOSSES"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p = tf_r2.add_paragraph()
    p.space_before = Pt(4)
    p.text = "Real feeders lose 501.67 kW across lines at 15,090.72 kW load. Naive balance algorithms that ignore physics fail to close by >500 kW."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_SECONDARY

    # Card 3: The Danger of Bad Interventions
    add_card(s1, Inches(6.8), Inches(5.05), Inches(5.733), Inches(1.8))
    tb_r3 = s1.shapes.add_textbox(Inches(7.0), Inches(5.15), Inches(5.333), Inches(1.6))
    tf_r3 = tb_r3.text_frame
    tf_r3.word_wrap = True
    p = tf_r3.paragraphs[0]
    p.text = "THE IMPACT ON DISCOMS & RELIABILITY"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p = tf_r3.add_paragraph()
    p.space_before = Pt(4)
    p.text = "When automated dispatch acts blindly on point forecasts, incorrect commands trigger line overloads, voltage collapse, and severe DSM deviation penalties."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_SECONDARY
    p = tf_r3.add_paragraph()
    p.space_before = Pt(4)
    p.text = "GridSentinal Core Objective: Turn raw predictions into verified, risk-aware energy decisions."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.bold = True
    p.font.color.rgb = TEXT_PRIMARY

    # =========================================================================
    # SLIDE 2: WHY EXISTING APPROACHES FALL SHORT
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2)
    add_header(s2, "02 / The Critical Industry Gap", "PREDICTION IS ONLY THE FIRST PROBLEM.",
               "Why naive automation fails: four interacting challenges that make unverified AI unsafe.")
    add_footer(s2, 2)

    p_w, p_h = Inches(5.7), Inches(2.05)
    coords = [
        (Inches(0.8), Inches(1.85), "01 / VARIABLE DEMAND",
         "Non-Stationary Consumer Loads",
         "Customer demand profiles change by day of week, seasonal temperature, and lifestyle shifts.\n\n"
         "• Peak error concentrates in high-demand & high-ramp periods (18.5x error ratio vs baseline).\n"
         "• Standard regression models optimize average loss, collapsing exactly during critical stress hours."),

        (Inches(6.8), Inches(1.85), "02 / RENEWABLE INTERMITTENCY",
         "Weather-Dependent Generation Volatility",
         "Solar ramps occur with high frequency and zero advance notice from standard SCADA.\n\n"
         "• At 24h horizon without local weather forecasts, PV forecasting MAE jumps from 8.57 kW to 74.3 kW.\n"
         "• Percentage metrics (MAPE) fail completely (reading 126%) because 52.2% of solar data is zero."),

        (Inches(0.8), Inches(4.05), "03 / UNCERTAIN FLEXIBILITY",
         "Consumption History != Controllable Capacity",
         "Standard smart grid tools mistake past load variation for available demand response.\n\n"
         "• Real feeder audit reveals 0 of 8 flexibility dimensions have physical control interfaces.\n"
         "• Batteries often lack real-time state-of-charge telemetry; dispatching assumed flexibility causes blackout risk."),

        (Inches(6.8), Inches(4.05), "04 / UNCHECKED DECISION RISK",
         "Execution Without Adversarial Verification",
         "Point forecasts feed directly into optimization solvers without verifying failure modes.\n\n"
         "• Naive automation acts on confidence without knowing if the forecast is in an uncalibrated regime.\n"
         "• No red-teaming exists to ask: 'What happens if this dispatch command coincides with a solar drop?'")
    ]

    for left, top, num_tag, heading, body in coords:
        add_card(s2, left, top, p_w, p_h)
        tb = s2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.12), p_w - Inches(0.4), p_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = num_tag
        p.font.name = FONT_MONO
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        
        p = tf.add_paragraph()
        p.space_before = Pt(2)
        p.text = heading
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_PRIMARY
        
        p = tf.add_paragraph()
        p.space_before = Pt(3)
        p.text = body
        p.font.name = FONT_BODY
        p.font.size = Pt(8.8)
        p.font.color.rgb = TEXT_SECONDARY

    # Bottom summary callout banner
    add_card(s2, Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.65), bg=CARD_BG_ALT, border=ACCENT_RED)
    tb_b = s2.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.333), Inches(0.55))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "THE VULNERABILITY FORMULA:"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED
    p = tf_b.add_paragraph()
    p.space_before = Pt(2)
    p.text = "[ Forecast Error ]  +  [ Hidden Uncertainty ]  +  [ Unknown Capability ]  +  [ Physical Consequences ]  =  Naive Automation is Unsafe"
    p.font.name = FONT_MONO
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_PRIMARY

    # =========================================================================
    # SLIDE 3: INTRODUCING GRIDSENTINAL
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3)
    add_header(s3, "03 / Proposed Solution", "MEET GRIDSENTINAL.",
               "An adaptive, self-verifying energy intelligence system for renewable-integrated distributed networks.")
    add_footer(s3, 3)

    # Left: Core Definition & The Closed Loop
    add_card(s3, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0))
    tb_s3_l = s3.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.7))
    tf_s3_l = tb_s3_l.text_frame
    tf_s3_l.word_wrap = True

    p = tf_s3_l.paragraphs[0]
    p.text = "THE CORE PHILOSOPHY"
    p.font.name = FONT_MONO
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf_s3_l.add_paragraph()
    p.space_before = Pt(6)
    p.text = "Don't just predict the grid.\nVerify what happens next."
    p.font.name = FONT_HEADING
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = TEXT_PRIMARY

    p = tf_s3_l.add_paragraph()
    p.space_before = Pt(8)
    p.text = "GridSentinal connects predictive intelligence directly to verified action. Instead of treating forecasting as the end product, it wraps every prediction in calibrated uncertainty, audits available flexibility, and challenges candidate decisions before they touch physical hardware."
    p.font.name = FONT_BODY
    p.font.size = Pt(9.5)
    p.font.color.rgb = TEXT_SECONDARY

    p = tf_s3_l.add_paragraph()
    p.space_before = Pt(12)
    p.text = "THE 11-STAGE INTELLIGENCE LIFECYCLE:"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    stages = [
        ("01-04  OBSERVE & PREDICT", "Ingest telemetry, normalize power flows, run multi-expert ensemble, calibrate prediction intervals [OPERATIONAL]"),
        ("05-07  CHALLENGE & SIMULATE", "Formulate candidate dispatch, subject to adversarial red-teaming, run digital twin power-flow [DESIGNED]"),
        ("08-11  ASSURE, ACT & LEARN", "Decision assurance gate, verified execution, outcome tracking, energy-aware MLOps loop [ROADMAP]")
    ]
    for st_title, st_desc in stages:
        p = tf_s3_l.add_paragraph()
        p.space_before = Pt(5)
        p.text = f"• {st_title}:\n   {st_desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_PRIMARY

    # Right: Embedded UI Screenshot of the Feeder Console Card
    add_card(s3, Inches(6.6), Inches(1.85), Inches(5.933), Inches(5.0))
    if os.path.exists(r"presentation\assets\ui_feeder_card.png"):
        s3.shapes.add_picture(r"presentation\assets\ui_feeder_card.png",
                              Inches(6.75), Inches(2.05), width=Inches(5.633))
    
    # Caption under image (cleanly wrapped)
    tb_cap = s3.shapes.add_textbox(Inches(6.75), Inches(6.15), Inches(5.633), Inches(0.65))
    tf_cap = tb_cap.text_frame
    tf_cap.word_wrap = True
    p = tf_cap.paragraphs[0]
    p.text = "LIVE OPERATOR CONSOLE: Feeder p1uhs0_1247 (5,196 nodes).\n" \
             "Shaded 90% conformal interval bounding 1h demand forecast with operational verification pipeline."
    p.font.name = FONT_MONO
    p.font.size = Pt(8.2)
    p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 4: HOW IT WORKS — HORIZONTAL ARCHITECTURE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4)
    add_header(s4, "04 / System Architecture", "FROM TELEMETRY TO TRUSTED DECISION.",
               "A layered, evidence-based pipeline with explicit operational boundaries and zero fake mocks.")
    add_footer(s4, 4)

    col_w = Inches(2.8)
    gap = Inches(0.18)
    left_start = Inches(0.8)

    cols_data = [
        ("LAYER 1: DATA & TELEMETRY", ACCENT_GREEN, [
            ("SMART-DS v1.0 Feeder", "OPERATIONAL", "AUS/P1U 2018 base timeseries; 359 files SHA-256 verified; 35,040 native 15-min points."),
            ("Data Quality & Contract", "OPERATIONAL", "Strict Pydantic domain models; 3.22% physical line loss detected and accounted for; no synthetic filling.")
        ]),
        ("LAYER 2: PREDICTION & ERROR", ACCENT_CYAN, [
            ("Heterogeneous Ensemble", "OPERATIONAL", "Fixed ensemble (Persistence, Hist-GBM, Temporal TCN) beats learned router across all horizons (0.398 kW at h=1)."),
            ("Calibrated Uncertainty", "OPERATIONAL", "Split-conformal intervals on CAL_CONF; empirical coverage within ±0.01 at h=1, h=4, h=96 on sealed test.")
        ]),
        ("LAYER 3: FLEXIBILITY & PROXY", ACCENT_AMBER, [
            ("Flexibility Capability Audit", "OPERATIONAL", "0/8 dimensions physically supported in SMART-DS; battery dispatch unavailable (G-01); all gaps documented."),
            ("Behavioural Envelope", "STATISTICAL PROXY", "Empirical variation bounds (±2.73 kW at h=1 to ±14.12 kW at h=96); scheduler control strictly forbidden.")
        ]),
        ("LAYER 4: DECISION & ASSURANCE", ACCENT_BLUE, [
            ("Optimization & Red Team", "DESIGNED", "Multi-objective candidate actions attacked under stress scenarios (solar ramp, EV spike, telemetry delay)."),
            ("Digital Twin & Assurance", "ROADMAP", "OpenDSS power flow validation; assurance gate approves/rejects before physical execution touches hardware.")
        ])
    ]

    for idx, (col_title, col_accent, items) in enumerate(cols_data):
        c_x = left_start + idx * (col_w + gap)
        add_card(s4, c_x, Inches(1.85), col_w, Inches(4.35))
        
        tb = s4.shapes.add_textbox(c_x + Inches(0.12), Inches(1.95), col_w - Inches(0.24), Inches(0.35))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = col_title
        p.font.name = FONT_MONO
        p.font.size = Pt(8.2)
        p.font.bold = True
        p.font.color.rgb = col_accent

        for it_idx, (it_title, it_status, it_desc) in enumerate(items):
            it_y = Inches(2.35) + it_idx * Inches(1.85)
            add_card(s4, c_x + Inches(0.12), it_y, col_w - Inches(0.24), Inches(1.75), bg=CARD_BG_ALT)
            
            tb_it = s4.shapes.add_textbox(c_x + Inches(0.2), it_y + Inches(0.08), col_w - Inches(0.4), Inches(1.6))
            tf_it = tb_it.text_frame
            tf_it.word_wrap = True
            
            p = tf_it.paragraphs[0]
            p.text = it_title
            p.font.name = FONT_HEADING
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = TEXT_PRIMARY
            
            p = tf_it.add_paragraph()
            p.space_before = Pt(2)
            p.text = f"[{it_status}]"
            p.font.name = FONT_MONO
            p.font.size = Pt(7.8)
            p.font.bold = True
            if it_status == "OPERATIONAL":
                p.font.color.rgb = ACCENT_GREEN
            elif it_status == "STATISTICAL PROXY":
                p.font.color.rgb = ACCENT_AMBER
            else:
                p.font.color.rgb = ACCENT_CYAN
            
            p = tf_it.add_paragraph()
            p.space_before = Pt(3)
            p.text = it_desc
            p.font.name = FONT_BODY
            p.font.size = Pt(8.2)
            p.font.color.rgb = TEXT_SECONDARY

        # Add connector arrow between columns
        if idx < 3:
            ar_box = s4.shapes.add_textbox(c_x + col_w, Inches(3.8), gap, Inches(0.4))
            tf_ar = ar_box.text_frame
            tf_ar.margin_left = tf_ar.margin_right = tf_ar.margin_top = tf_ar.margin_bottom = 0
            p = tf_ar.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.text = "➔"
            p.font.name = FONT_HEADING
            p.font.size = Pt(12)
            p.font.color.rgb = ACCENT_GREEN

    # Bottom Architecture Rule Card (Word wrap, height adjusted)
    add_card(s4, Inches(0.8), Inches(6.30), Inches(11.733), Inches(0.60), bg=CARD_BG_ALT, border=ACCENT_CYAN)
    tb_ab = s4.shapes.add_textbox(Inches(1.0), Inches(6.34), Inches(11.333), Inches(0.52))
    tf_ab = tb_ab.text_frame
    tf_ab.word_wrap = True
    p = tf_ab.paragraphs[0]
    p.text = "ARCHITECTURAL INTEGRITY COMMITMENT: Learned models (Forecaster, Uncertainty) are separated from orchestrating agents (Red Team, Drift). Subpackages are created only when implemented — no empty stubs or simulated mocks."
    p.font.name = FONT_MONO
    p.font.size = Pt(8.2)
    p.font.color.rgb = TEXT_PRIMARY

    # =========================================================================
    # SLIDE 5: FORECASTING + UNCERTAINTY INTELLIGENCE (Native Table)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5)
    add_header(s5, "05 / Probabilistic Intelligence", "A FORECAST WITHOUT UNCERTAINTY IS HALF A FORECAST.",
               "Point predictions fail silently; calibrated intervals identify exactly which rows must not be trusted.")
    add_footer(s5, 5)

    # Left Card: Native Formatted Table + Quantitative Insights
    add_card(s5, Inches(0.8), Inches(1.85), Inches(5.8), Inches(5.0))
    tb_s5_l = s5.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.4), Inches(0.7))
    tf_s5_l = tb_s5_l.text_frame
    tf_s5_l.word_wrap = True

    p = tf_s5_l.paragraphs[0]
    p.text = "PHASE 8 EMPIRICAL EVIDENCE (SEALED TEST, 179,520 ROWS)"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf_s5_l.add_paragraph()
    p.space_before = Pt(3)
    p.text = "Evaluated on sealed chronological test split without data leakage:"
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_SECONDARY

    # Add Native PowerPoint Table
    tbl_shape = s5.shapes.add_table(4, 5, Inches(1.0), Inches(2.65), Inches(5.4), Inches(1.75))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.1)
    tbl.columns[1].width = Inches(1.5)
    tbl.columns[2].width = Inches(0.9)
    tbl.columns[3].width = Inches(0.8)
    tbl.columns[4].width = Inches(1.1)

    headers = ["Horizon", "Published Method", "Cov @90%", "Corr ρ", "Top Decile"]
    row_data = [
        ["h = 1 (15m)", "conformal_state", "90.20%", "+0.709", "4.61× err"],
        ["h = 4 (1h)", "conformal_state", "89.73%", "+0.673", "4.21× err"],
        ["h = 96 (24h)", "conformal_dispersion", "89.34%", "+0.658", "4.80× err"]
    ]

    for col_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_HEADER_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.name = FONT_MONO
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    for r_idx, r_vals in enumerate(row_data):
        for c_idx, val in enumerate(r_vals):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = TABLE_ROW_ALT if r_idx % 2 == 1 else CARD_BG_ALT
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_MONO
            p.font.size = Pt(8)
            p.font.color.rgb = TEXT_PRIMARY if c_idx != 2 else ACCENT_GREEN
            p.font.bold = (c_idx == 2 or c_idx == 0)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Key Quantitative Insights below table
    tb_s5_bot = s5.shapes.add_textbox(Inches(1.0), Inches(4.55), Inches(5.4), Inches(2.2))
    tf_s5_bot = tb_s5_bot.text_frame
    tf_s5_bot.word_wrap = True

    p = tf_s5_bot.paragraphs[0]
    p.text = "KEY QUANTITATIVE TAKEAWAYS:"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.8)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER

    p = tf_s5_bot.add_paragraph()
    p.space_before = Pt(4)
    p.text = "• Error-Width Coupling: Spearman correlation between interval width and realized error is +0.66 to +0.71. The interval reliably expands when the grid is unpredictable."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.5)
    p.font.color.rgb = TEXT_PRIMARY

    p = tf_s5_bot.add_paragraph()
    p.space_before = Pt(4)
    p.text = "• Risk Containment: The widest 10% of prediction intervals carries 4.2x to 4.8x the mean error — enabling automated decision suppression during volatile periods."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.5)
    p.font.color.rgb = TEXT_PRIMARY

    # Right: Methodological Honesty (Negative Finding) + UI Asset
    add_card(s5, Inches(6.8), Inches(1.85), Inches(5.733), Inches(5.0))
    tb_s5_r = s5.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.333), Inches(2.1))
    tf_s5_r = tb_s5_r.text_frame
    tf_s5_r.word_wrap = True

    p = tf_s5_r.paragraphs[0]
    p.text = "METHODOLOGICAL HONESTY: THE ±1σ NEGATIVE RESULT"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED

    p = tf_s5_r.add_paragraph()
    p.space_before = Pt(4)
    p.text = "A constant-width ±1σ interval (the default rule most practitioners use) fails 11 of 12 coverage cells on this dataset.\n\n" \
             "• At h=1 it over-covers (0.984 vs nominal 0.95); at h=96 it under-covers (0.907).\n" \
             "• The root cause: Point forecast error differs by +8.6% (h=1) and -11.0% (h=96) across the year. A static width calibrated on one half does not transfer to another.\n" \
             "• Conformal calibration guarantees are NOT claimed: time series residuals violate exchangeability. We report empirical held-out coverage, not marketing fiction."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.5)
    p.font.color.rgb = TEXT_SECONDARY

    # Embed UI screenshot of uncertainty boxes
    if os.path.exists(r"presentation\assets\ui_uncertainty_boxes.png"):
        s5.shapes.add_picture(r"presentation\assets\ui_uncertainty_boxes.png",
                              Inches(7.0), Inches(4.35), width=Inches(5.333))

    # =========================================================================
    # SLIDE 6: FLEXIBILITY INTELLIGENCE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6)
    add_header(s6, "06 / Flexibility Capability Audit", "FLEXIBILITY WITHOUT FICTION.",
               "Historical consumption behavior does not automatically imply dispatchable capacity.")
    add_footer(s6, 6)

    t_w = Inches(3.75)
    t_h = Inches(4.25)
    tiers = [
        ("PHYSICAL FLEXIBILITY", "UNKNOWN ON RAW DATA", ACCENT_RED, [
            ("0 of 8 Dimensions Supported", "SMART-DS carries no physical limits or control interfaces for customer loads."),
            ("Battery Telemetry Blindspot", "93/93 storage units report State=IDLING with nameplate scalar only; no SOC series (G-01)."),
            ("Missing DER Programs", "Zero demand-response enrolment, HVAC setpoint, or EV departure series exist (G-07 through G-10)."),
            ("Rigorous Verdict: NO", "Refusing to invent controllable capacity is a core safety feature, not an omission.")
        ]),
        ("STATISTICAL PROXY", "PARTIALLY ESTIMABLE", ACCENT_AMBER, [
            ("Behavioural Envelope", "Measures how far demand historically deviated from expected profile without intervention."),
            ("Calibrated Widths", "Bounded at ±2.73 kW (h=1), ±5.57 kW (h=4), ±14.12 kW (h=96)."),
            ("6x Wider Than Point MAE", "90% two-sided coverage requires wide bands due to heavy-tailed load variation."),
            ("Strict Interpretation", "'We were surprised outside this band 10% of the time' — NOT 'We can move 6 kW.'")
        ]),
        ("ASSUMED FLEXIBILITY", "SCENARIO ONLY", ACCENT_CYAN, [
            ("Advisory Authority Only", "Assumed values are allowed strictly for offline scenario stress-testing and simulation."),
            ("Enforced Domain Contract", "Code constructor raises DomainValidationError if a statistical proxy claims scheduleable control."),
            ("Zero Unverified Commands", "The decision engine is architecturally barred from commanding unverified capacity."),
            ("DISCOM Value", "Protects distribution utilities from severe under-delivery imbalance penalties.")
        ])
    ]

    for idx, (t_title, t_badge, t_color, t_bullets) in enumerate(tiers):
        t_x = Inches(0.8) + idx * (t_w + Inches(0.24))
        add_card(s6, t_x, Inches(1.85), t_w, t_h)
        
        tb = s6.shapes.add_textbox(t_x + Inches(0.15), Inches(1.95), t_w - Inches(0.3), Inches(0.65))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = t_title
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_PRIMARY
        
        p = tf.add_paragraph()
        p.space_before = Pt(2)
        p.text = f"[{t_badge}]"
        p.font.name = FONT_MONO
        p.font.size = Pt(8.2)
        p.font.bold = True
        p.font.color.rgb = t_color

        tb_b = s6.shapes.add_textbox(t_x + Inches(0.15), Inches(2.65), t_w - Inches(0.3), t_h - Inches(0.85))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        for b_title, b_desc in t_bullets:
            p = tf_b.add_paragraph() if tf_b.paragraphs[0].text else tf_b.paragraphs[0]
            p.space_before = Pt(5)
            p.text = f"◆ {b_title}"
            p.font.name = FONT_HEADING
            p.font.size = Pt(9.2)
            p.font.bold = True
            p.font.color.rgb = TEXT_PRIMARY
            
            p = tf_b.add_paragraph()
            p.space_before = Pt(1)
            p.text = b_desc
            p.font.name = FONT_BODY
            p.font.size = Pt(8.2)
            p.font.color.rgb = TEXT_SECONDARY

    # Bottom Callout: The Withheld Uncertainty Discount
    add_card(s6, Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.65), bg=CARD_BG_ALT, border=ACCENT_GREEN)
    tb_s6b = s6.shapes.add_textbox(Inches(1.0), Inches(6.28), Inches(11.333), Inches(0.58))
    tf_s6b = tb_s6b.text_frame
    tf_s6b.word_wrap = True
    p = tf_s6b.paragraphs[0]
    p.text = "SCIENTIFIC INTEGRITY: UNCERTAINTY PREDICTS FLEXIBILITY, YET DISCOUNT WAS WITHHELD"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.8)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p = tf_s6b.add_paragraph()
    p.space_before = Pt(2)
    p.text = "Phase 8 uncertainty correlated with realized flexibility deviations (+0.393 / +0.246 / +0.312 at h=1/4/96). Because this was observed post-hoc on the sealed split, GridSentinal deliberately refused to apply an uncertainty discount to avoid test-leakage contamination."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.2)
    p.font.color.rgb = TEXT_SECONDARY

    # =========================================================================
    # SLIDE 7: SELF-VERIFYING DECISION ARCHITECTURE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7)
    add_header(s7, "07 / Decision Assurance Engine", "BEFORE IT ACTS, IT TRIES TO BREAK THE DECISION.",
               "Instead of asking only 'What should the system do?', GridSentinal asks 'What could go wrong if it does?'")
    add_footer(s7, 7)

    step_w = Inches(2.2)
    step_gap = Inches(0.18)
    step_top = Inches(1.90)
    step_h = Inches(4.25)

    steps_data = [
        ("01 / PROPOSE", "Optimizer Engine", "DESIGNED (Phase 10)", ACCENT_CYAN,
         "Converts forecasts, calibrated uncertainty envelopes, and contractual flexibility into candidate dispatch actions.\n\n"
         "Formulates multi-objective cost vs reliability trade-offs under operational constraints."),

        ("02 / RED TEAM", "Adversarial Engine", "DESIGNED (Phase 12)", ACCENT_RED,
         "Subjects candidate action to deliberate extreme stress scenarios:\n\n"
         "• Sudden 80% solar collapse\n"
         "• Correlated EV cluster spike\n"
         "• Communication latency & dropped telemetry\n"
         "• Unmodelled 3.22% line losses"),

        ("03 / SIMULATE", "Digital Twin", "DESIGNED (Phase 11)", ACCENT_AMBER,
         "Independent power-flow simulation (OpenDSS integration) computes actual physical outcomes:\n\n"
         "• Feeder voltage deviations\n"
         "• Line thermal overload limits\n"
         "• Substation back-feed risks"),

        ("04 / ASSURE", "Assurance Gate", "DESIGNED (Phase 13)", ACCENT_GREEN,
         "Decision Assurance evaluates candidate against physics & policy:\n\n"
         "• Does action survive Red Team?\n"
         "• Is uncertainty within safe bounds?\n"
         "• Action: Approve, Reject, or Re-optimize"),

        ("05 / EXECUTE", "Adaptive Autonomy", "ROADMAP (Phase 16)", ACCENT_BLUE,
         "Closed-loop actuation:\n\n"
         "• Autonomous execution for verified high-confidence actions\n"
         "• Fallback to Human Supervisor for edge-case alerts\n"
         "• Real-world outcome telemetry loops into MLOps")
    ]

    for idx, (st_num, st_name, st_stat, st_color, st_text) in enumerate(steps_data):
        x = Inches(0.8) + idx * (step_w + step_gap)
        add_card(s7, x, step_top, step_w, step_h)
        
        tb = s7.shapes.add_textbox(x + Inches(0.12), step_top + Inches(0.12), step_w - Inches(0.24), Inches(0.75))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = st_num
        p.font.name = FONT_MONO
        p.font.size = Pt(8.2)
        p.font.bold = True
        p.font.color.rgb = st_color
        
        p = tf.add_paragraph()
        p.space_before = Pt(1)
        p.text = st_name
        p.font.name = FONT_HEADING
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_PRIMARY
        
        p = tf.add_paragraph()
        p.space_before = Pt(2)
        p.text = f"[{st_stat}]"
        p.font.name = FONT_MONO
        p.font.size = Pt(7.2)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        tb_body = s7.shapes.add_textbox(x + Inches(0.12), step_top + Inches(0.9), step_w - Inches(0.24), step_h - Inches(1.0))
        tf_body = tb_body.text_frame
        tf_body.word_wrap = True
        p = tf_body.paragraphs[0]
        p.text = st_text
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_SECONDARY

        # Connector arrow
        if idx < 4:
            ar_box = s7.shapes.add_textbox(x + step_w, step_top + Inches(1.8), step_gap, Inches(0.4))
            tf_ar = ar_box.text_frame
            tf_ar.margin_left = tf_ar.margin_right = tf_ar.margin_top = tf_ar.margin_bottom = 0
            p = tf_ar.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.text = "➔"
            p.font.name = FONT_HEADING
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_GREEN

    # Bottom Core Differentiator Banner (no clipping, wrapped)
    add_card(s7, Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.65), bg=CARD_BG_ALT, border=ACCENT_GREEN)
    tb_s7b = s7.shapes.add_textbox(Inches(1.0), Inches(6.28), Inches(11.333), Inches(0.58))
    tf_s7b = tb_s7b.text_frame
    tf_s7b.word_wrap = True
    p = tf_s7b.paragraphs[0]
    p.text = "THE CORE DIFFERENTIATOR:"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p = tf_s7b.add_paragraph()
    p.space_before = Pt(2)
    p.text = "Trust is not earned by high model confidence. Trust is earned when candidate actions survive deliberate attempts to break them before touching physical grid hardware."
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_PRIMARY

    # =========================================================================
    # SLIDE 8: TECHNICAL APPROACH & RESEARCH EVIDENCE
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8)
    add_header(s8, "08 / Research & Experimental Evidence", "WE TESTED THE IDEA. WE DIDN'T JUST DEMO IT.",
               "1,235 tests passing, 29 registered experiments, and rigorous negative result reporting.")
    add_footer(s8, 8)

    # Left: Rigorous Stack & Experimental Discipline
    add_card(s8, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0))
    tb_s8_l = s8.shapes.add_textbox(Inches(1.0), Inches(1.98), Inches(5.2), Inches(2.7))
    tf_s8_l = tb_s8_l.text_frame
    tf_s8_l.word_wrap = True

    p = tf_s8_l.paragraphs[0]
    p.text = "TECHNICAL STACK & PROTOCOL DISCIPLINE"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    stack_points = [
        ("Runtime & Tooling", "Python 3.12, PyTorch CPU-deterministic, scikit-learn, OpenDSS parser. pandas was deliberately rejected across the entire core pipeline."),
        ("Leakage Protection", "Chronological train/validation/test partitions. Validation split into CAL_FIT (50%) and CAL_CONF (50%). Sealed test split read exactly once."),
        ("Pre-Registered Bars", "Success thresholds registered before execution (e.g. Qwen required to beat 1.5648 kW at 24h). Misses recorded rather than tuned around."),
        ("Verification Integrity", "1,235 automated tests passing; 29 committed experiment runs in registry.jsonl; D-001 through D-109 architectural decision records.")
    ]
    for sp_title, sp_desc in stack_points:
        p = tf_s8_l.add_paragraph()
        p.space_before = Pt(6)
        p.text = f"◆ {sp_title}: {sp_desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_PRIMARY

    # Embedded UI operational cards
    if os.path.exists(r"presentation\assets\ui_operational_cards.png"):
        s8.shapes.add_picture(r"presentation\assets\ui_operational_cards.png",
                              Inches(1.0), Inches(4.75), width=Inches(5.2))

    # Right: 5 Rigorous Experimental Findings
    add_card(s8, Inches(6.6), Inches(1.85), Inches(5.933), Inches(5.0))
    tb_s8_r = s8.shapes.add_textbox(Inches(6.8), Inches(1.95), Inches(5.533), Inches(4.8))
    tf_s8_r = tb_s8_r.text_frame
    tf_s8_r.word_wrap = True

    p = tf_s8_r.paragraphs[0]
    p.text = "FIVE COMPLETED RESEARCH FINDINGS"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    findings = [
        ("Phase 6: Language-Model Specialization", "NEGATIVE RESULT", ACCENT_RED,
         "A frozen Qwen3-1.7B LLM trunk lost to classical GBM at every horizon (2.74 vs 1.56 kW at 24h). Readout representations collapsed to participation ratio 2.1 vs 10.0 for random controls."),
        ("Phase 7: Learned Expert Router (MoE)", "NEGATIVE RESULT", ACCENT_RED,
         "While oracle routing proved 30-38% theoretical headroom existed, a learned neural router lost to the fixed weighted ensemble at 0 of 3 horizons (0.398 vs 0.407 kW at h=1). The fixed ensemble shipped."),
        ("Phase 8: Calibrated Prediction Intervals", "VERIFIED YES", ACCENT_GREEN,
         "Conformal state intervals achieved nominal 90% coverage (89.3%-90.2%) across all horizons. Disproved static ±1σ assumption (failed 11/12 cells)."),
        ("Phase 9: Flexibility Capability Audit", "MEASURED VERDICT: NO", ACCENT_AMBER,
         "Disproved the industry myth of free demand response. 0 of 8 dimensions possessed physical control interfaces; statistical proxy bounded at ±2.73 to ±14.12 kW."),
        ("Phase 10: Controlled Feature Ablation", "COMPLETE EVIDENCE", ACCENT_CYAN,
         "Feature contribution is target-dependent. PV improved -27.40% under autoregressive lag features (A->B), while customer load preferred simple calendar features (Set A).")
    ]

    for f_title, f_badge, f_color, f_desc in findings:
        p = tf_s8_r.add_paragraph()
        p.space_before = Pt(5)
        p.text = f"{f_title}  [{f_badge}]"
        p.font.name = FONT_HEADING
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = f_color
        
        p = tf_s8_r.add_paragraph()
        p.space_before = Pt(1)
        p.text = f_desc
        p.font.name = FONT_BODY
        p.font.size = Pt(8)
        p.font.color.rgb = TEXT_SECONDARY

    # =========================================================================
    # SLIDE 9: INNOVATION & GRID RELIABILITY ALIGNMENT
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9)
    add_header(s9, "09 / Innovation & Challenge Alignment", "WHAT ACTUALLY MAKES GRIDSENTINAL DIFFERENT?",
               "Yuva Yodha Energy Tech Hackathon: Direct causal mapping to Grid Reliability challenge needs.")
    add_footer(s9, 9)

    row_w = Inches(11.733)
    row_h = Inches(1.02)
    row_top_start = Inches(1.85)
    row_gap = Inches(0.12)

    alignments = [
        ("CHALLENGE NEED: RENEWABLE INTERMITTENCY",
         "Feeder solar ramps cause voltage spikes, reverse power flows, and sudden generation deficits.",
         "GRIDSENTINAL CAPABILITY",
         "15-minute multi-expert nowcasting + calibrated 24h day-ahead horizon with conformal error envelopes.",
         "EXPECTED VALUE",
         "DISCOMs gain quantified ramp foresight, preventing blind generation shortfalls."),

        ("CHALLENGE NEED: DER & STORAGE COORDINATION",
         "Uncoordinated distributed solar and batteries can destabilize local distribution networks.",
         "GRIDSENTINAL CAPABILITY",
         "Rigorous Flexibility Capability Audit + Behavioural Envelopes; refuses unverified dispatch commands.",
         "EXPECTED VALUE",
         "Prevents over-commitment penalties; commands only verified, physically supported assets."),

        ("CHALLENGE NEED: TECHNICAL LOSSES & PHYSICS",
         "Distribution lines exhibit 3.22% technical losses (501.7 kW on feeder), causing power balance failure.",
         "GRIDSENTINAL CAPABILITY",
         "Formal Energy Contract with loss modeling; OpenDSS digital twin physical power-flow validation.",
         "EXPECTED VALUE",
         "Ensures dispatch actions close physically and prevent transformer thermal stress."),

        ("CHALLENGE NEED: AUTONOMOUS GRID SAFETY",
         "Direct execution of unverified AI decisions risks catastrophic grid events and utility liability.",
         "GRIDSENTINAL CAPABILITY",
         "Adversarial Red Team stress-testing + Decision Assurance Gate with human-in-the-loop fallback.",
         "EXPECTED VALUE",
         "Zero unchecked automated decisions; verifiable operational safety for industrial deployment.")
    ]

    for idx, (c_need_h, c_need_t, cap_h, cap_t, val_h, val_t) in enumerate(alignments):
        y = row_top_start + idx * (row_h + row_gap)
        add_card(s9, Inches(0.8), y, row_w, row_h)
        
        # Column 1: Need
        tb1 = s9.shapes.add_textbox(Inches(0.95), y + Inches(0.06), Inches(3.4), row_h - Inches(0.12))
        tf1 = tb1.text_frame
        tf1.word_wrap = True
        p = tf1.paragraphs[0]
        p.text = c_need_h
        p.font.name = FONT_MONO
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = ACCENT_AMBER
        p = tf1.add_paragraph()
        p.space_before = Pt(2)
        p.text = c_need_t
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_SECONDARY

        # Arrow 1
        ar1 = s9.shapes.add_textbox(Inches(4.38), y + Inches(0.28), Inches(0.3), Inches(0.4))
        p = ar1.text_frame.paragraphs[0]
        p.text = "➔"
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN

        # Column 2: Capability
        tb2 = s9.shapes.add_textbox(Inches(4.7), y + Inches(0.06), Inches(4.0), row_h - Inches(0.12))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        p = tf2.paragraphs[0]
        p.text = cap_h
        p.font.name = FONT_MONO
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p = tf2.add_paragraph()
        p.space_before = Pt(2)
        p.text = cap_t
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_PRIMARY

        # Arrow 2
        ar2 = s9.shapes.add_textbox(Inches(8.75), y + Inches(0.28), Inches(0.3), Inches(0.4))
        p = ar2.text_frame.paragraphs[0]
        p.text = "➔"
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_GREEN

        # Column 3: Value
        tb3 = s9.shapes.add_textbox(Inches(9.05), y + Inches(0.06), Inches(3.35), row_h - Inches(0.12))
        tf3 = tb3.text_frame
        tf3.word_wrap = True
        p = tf3.paragraphs[0]
        p.text = val_h
        p.font.name = FONT_MONO
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = ACCENT_GREEN
        p = tf3.add_paragraph()
        p.space_before = Pt(2)
        p.text = val_t
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_PRIMARY

    # Bottom Innovation Summary Box (Word-wrap clean)
    add_card(s9, Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.65), bg=CARD_BG_ALT, border=ACCENT_GREEN)
    tb_s9b = s9.shapes.add_textbox(Inches(1.0), Inches(6.28), Inches(11.333), Inches(0.58))
    tf_s9b = tb_s9b.text_frame
    tf_s9b.word_wrap = True
    p = tf_s9b.paragraphs[0]
    p.text = "THE SYSTEM-LEVEL INNOVATION:"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p = tf_s9b.add_paragraph()
    p.space_before = Pt(2)
    p.text = "GridSentinal treats uncertainty, capability limits, and decision verification as first-class citizens of energy intelligence, rather than cosmetic reporting dashboards."
    p.font.name = FONT_BODY
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_PRIMARY

    # =========================================================================
    # SLIDE 10: EXPECTED IMPACT & IMPLEMENTATION ROADMAP
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10)
    add_header(s10, "10 / Value Creation & Roadmap", "FROM LAB EVIDENCE TO INDUSTRIAL DEPLOYMENT.",
               "Clear distinction between completed research foundation and future industrial scaling.")
    add_footer(s10, 10)

    # Left: Expected Impact Categories (No Fabricated Stats)
    add_card(s10, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0))
    tb_s10_l = s10.shapes.add_textbox(Inches(1.0), Inches(1.98), Inches(5.2), Inches(4.7))
    tf_s10_l = tb_s10_l.text_frame
    tf_s10_l.word_wrap = True

    p = tf_s10_l.paragraphs[0]
    p.text = "EXPECTED SYSTEM IMPACT (SCIENTIFICALLY FRAMED)"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    impacts = [
        ("Enhanced Grid Reliability", "Pre-emptive mitigation of solar ramps & evening demand peaks through calibrated forecast bounds; eliminates blind tripping."),
        ("DISCOM Financial Integrity", "Separating statistical proxies from physical flexibility prevents costly DSM deviation penalties and battery degradation."),
        ("Operational Safety & Assurance", "Adversarial stress-testing ensures zero unverified autonomous dispatch commands reach critical distribution assets."),
        ("Physics-Grounded Optimization", "Incorporating explicit 3.22% technical losses guarantees power balance closure across feeder nodes."),
        ("Outcome-Aware MLOps", "Model degradation is evaluated by physical consequence (voltage violations, line stress) rather than offline validation loss.")
    ]
    for im_title, im_desc in impacts:
        p = tf_s10_l.add_paragraph()
        p.space_before = Pt(7)
        p.text = f"◆ {im_title}"
        p.font.name = FONT_HEADING
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_PRIMARY
        p = tf_s10_l.add_paragraph()
        p.space_before = Pt(1)
        p.text = im_desc
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_SECONDARY

    # Right: Implementation Roadmap (Phases)
    add_card(s10, Inches(6.6), Inches(1.85), Inches(5.933), Inches(5.0))
    tb_s10_r = s10.shapes.add_textbox(Inches(6.8), Inches(1.98), Inches(5.533), Inches(4.7))
    tf_s10_r = tb_s10_r.text_frame
    tf_s10_r.word_wrap = True

    p = tf_s10_r.paragraphs[0]
    p.text = "PHASED IMPLEMENTATION ROADMAP"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    phases = [
        ("PHASE I: PROVEN FOUNDATION", "NOW (COMPLETE)", ACCENT_GREEN,
         "• SMART-DS v1.0 data ingestion & reality validation (5,196 nodes)\n"
         "• Heterogeneous multi-expert forecasting ensemble\n"
         "• Calibrated split-conformal uncertainty intervals (89.7% cov)\n"
         "• Flexibility capability audit & Phase 10 feature ablation"),

        ("PHASE II: DECISION & VERIFICATION", "Q1-Q2 2026 (NEXT)", ACCENT_AMBER,
         "• Pyomo/MPC constrained candidate action optimizer (Phase 10)\n"
         "• OpenDSS digital twin power-flow integration (Phase 11)\n"
         "• Adversarial Red-Team scenario attack engine (Phase 12)\n"
         "• Decision assurance scoring gate with operator console (Phase 13)"),

        ("PHASE III: INDUSTRIAL SCALING", "Q3-Q4 2026 (ROADMAP)", ACCENT_BLUE,
         "• Schneider Electric EcoStruxure ADMS / DERMS API bridge\n"
         "• Hardware-in-the-loop (HIL) microgrid testbed integration\n"
         "• Real-world DISCOM pilot deployment with adaptive autonomy\n"
         "• Outcome-aware MLOps agent orchestration with automated drift recovery")
    ]

    for ph_title, ph_time, ph_color, ph_desc in phases:
        p = tf_s10_r.add_paragraph()
        p.space_before = Pt(7)
        p.text = f"{ph_title}  [{ph_time}]"
        p.font.name = FONT_HEADING
        p.font.size = Pt(9.2)
        p.font.bold = True
        p.font.color.rgb = ph_color
        p = tf_s10_r.add_paragraph()
        p.space_before = Pt(2)
        p.text = ph_desc
        p.font.name = FONT_BODY
        p.font.size = Pt(8)
        p.font.color.rgb = TEXT_SECONDARY

    # =========================================================================
    # SLIDE 11: TEAM & CLOSING STATEMENT
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11)
    add_header(s11, "11 / Team & Conclusion", "BUILDING THE GRID'S INTELLIGENCE LAYER.",
               "Yuva Yodha Energy Tech Hackathon 2026 — Schneider Electric.")
    add_footer(s11, 11)

    # Left: Team Information & Competencies (Using recolored high-contrast logo)
    add_card(s11, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0))
    
    # Team Logo (high-contrast light on dark)
    if os.path.exists(r"presentation\assets\team_logo_light.png"):
        s11.shapes.add_picture(r"presentation\assets\team_logo_light.png",
                               Inches(1.0), Inches(2.05), width=Inches(1.3))

    tb_s11_l = s11.shapes.add_textbox(Inches(2.45), Inches(2.05), Inches(3.8), Inches(1.35))
    tf_s11_l = tb_s11_l.text_frame
    tf_s11_l.word_wrap = True
    p = tf_s11_l.paragraphs[0]
    p.text = "THE UNSUPERVISED SUSPECTS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_PRIMARY
    
    p = tf_s11_l.add_paragraph()
    p.space_before = Pt(2)
    p.text = "Team Lead & Developer: Pratham Kapoor\nEmail: prathamkapoor027@gmail.com"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.5)
    p.font.color.rgb = ACCENT_CYAN

    # Team details text
    tb_td = s11.shapes.add_textbox(Inches(1.0), Inches(3.45), Inches(5.2), Inches(3.2))
    tf_td = tb_td.text_frame
    tf_td.word_wrap = True
    
    p = tf_td.paragraphs[0]
    p.text = "CORE EXPERTISE & RESEARCH FOCUS:"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    team_points = [
        ("Probabilistic Energy AI", "Split-conformal prediction, quantile regression, heterogeneous ensemble modeling."),
        ("Energy Systems Modeling", "OpenDSS distribution power-flow, SMART-DS v1.0 architecture, feeder loss accounting."),
        ("Adversarial Verification", "Red-teaming decision pipelines, digital twin safety constraints, adaptive autonomy gates."),
        ("Enterprise MLOps", "Zero-leakage data pipelines, verifiable experiment registries, automated audit traces.")
    ]
    for tp_title, tp_desc in team_points:
        p = tf_td.add_paragraph()
        p.space_before = Pt(5)
        p.text = f"◆ {tp_title}: {tp_desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_SECONDARY

    # Right: Standards, Repository & Grand Closing Callout
    add_card(s11, Inches(6.6), Inches(1.85), Inches(5.933), Inches(5.0))
    tb_s11_r = s11.shapes.add_textbox(Inches(6.85), Inches(2.05), Inches(5.4), Inches(4.6))
    tf_s11_r = tb_s11_r.text_frame
    tf_s11_r.word_wrap = True

    p = tf_s11_r.paragraphs[0]
    p.text = "ENGINEERING REFERENCES & STANDARDS"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    refs = [
        ("NREL SMART-DS v1.0", "Synthetic Models for Advanced, Realistic, Reliable Grid Scenarios"),
        ("IEEE 1547-2018", "Standard for Interconnection and Interoperability of Distributed Energy Resources"),
        ("Conformal Prediction", "Algorithmic Learning in a Random World (Vovk, Gammerman, Shafer)"),
        ("EPRI OpenDSS", "Electric Power Research Institute Open Distribution System Simulator")
    ]
    for r_title, r_desc in refs:
        p = tf_s11_r.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {r_title}: {r_desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_SECONDARY

    p = tf_s11_r.add_paragraph()
    p.space_before = Pt(12)
    p.text = "OPEN SOURCE REPOSITORY & PROVENANCE:"
    p.font.name = FONT_MONO
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf_s11_r.add_paragraph()
    p.space_before = Pt(3)
    p.text = "GitHub: https://github.com/PrathamKapoor/GridSentinal"
    p.font.name = FONT_MONO
    p.font.size = Pt(8.8)
    p.font.color.rgb = TEXT_PRIMARY

    # Final Bold Statement
    p = tf_s11_r.add_paragraph()
    p.space_before = Pt(18)
    p.text = "DON'T JUST AUTOMATE THE GRID."
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    p = tf_s11_r.add_paragraph()
    p.space_before = Pt(2)
    p.text = "MAKE IT THINK BEFORE IT ACTS."
    p.font.name = FONT_HEADING
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    output_path = r"presentation\GridSentinal_Hackathon_2026.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_deck()
