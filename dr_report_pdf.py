"""Builds the one-page, A4 landscape screening report as PDF bytes in memory."""

import io
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from dr_inference import ARCHITECTURE_DISPLAY_NAMES

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
PAGE_MARGIN = 30
INK_COLOUR = colors.HexColor("#1B2A3A")
SLATE_COLOUR = colors.HexColor("#5A6878")
RULE_COLOUR = colors.HexColor("#C9D2DB")
PANEL_COLOUR = colors.HexColor("#F3F6F8")
SEVERITY_COLOURS = ["#2F7D5B", "#7C8B2A", "#C18A1A", "#C0582A", "#9E2A34"]
PREPROCESSING_CAPTION_NAMES = {"clahe": "CLAHE", "ben_graham": "Ben Graham", "resize_only": "no enhancement"}

BODY_STYLE = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.5, leading=12, textColor=INK_COLOUR)
NOTE_STYLE = ParagraphStyle("note", fontName="Times-Roman", fontSize=9.5, leading=11, textColor=INK_COLOUR)
DISCLAIMER_TOP_Y = PAGE_MARGIN + 26
SMALL_STYLE = ParagraphStyle("small", fontName="Times-Roman", fontSize=8.5, leading=10.5, textColor=SLATE_COLOUR)
DISCLAIMER_STYLE = ParagraphStyle("disclaimer", fontName="Times-Italic", fontSize=8, leading=10, textColor=SLATE_COLOUR)


def draw_wrapped_text(pdf_canvas, text, style, left_x, top_y, available_width):
    paragraph = Paragraph(text, style)
    _, paragraph_height = paragraph.wrap(available_width, PAGE_HEIGHT)
    paragraph.drawOn(pdf_canvas, left_x, top_y - paragraph_height)
    return top_y - paragraph_height


def draw_section_label(pdf_canvas, label_text, left_x, baseline_y):
    pdf_canvas.setFont("Times-Bold", 10)
    pdf_canvas.setFillColor(INK_COLOUR)
    pdf_canvas.drawString(left_x, baseline_y, label_text)


def draw_image_with_caption(pdf_canvas, image_rgb, left_x, top_y, box_size, caption_text):
    pil_image = Image.fromarray(image_rgb)
    pil_image.thumbnail((600, 600))
    image_width, image_height = pil_image.size
    scale = min(box_size / image_width, box_size / image_height)
    drawn_width, drawn_height = image_width * scale, image_height * scale
    pdf_canvas.setFillColor(colors.black)
    pdf_canvas.rect(left_x, top_y - box_size, box_size, box_size, stroke=0, fill=1)
    pdf_canvas.drawImage(
        ImageReader(pil_image), left_x + (box_size - drawn_width) / 2, top_y - box_size + (box_size - drawn_height) / 2,
        width=drawn_width, height=drawn_height,
    )
    draw_wrapped_text(pdf_canvas, escape(caption_text), SMALL_STYLE, left_x, top_y - box_size - 4, box_size)


def draw_header(pdf_canvas, case_record):
    top_y = PAGE_HEIGHT - PAGE_MARGIN
    pdf_canvas.setFillColor(INK_COLOUR)
    pdf_canvas.setFont("Times-Bold", 16)
    pdf_canvas.drawString(PAGE_MARGIN, top_y - 12, case_record["facility_name"] or "Diabetic eye screening service")
    pdf_canvas.setFont("Times-Roman", 11.5)
    pdf_canvas.setFillColor(SLATE_COLOUR)
    pdf_canvas.drawString(PAGE_MARGIN, top_y - 28, "Diabetic Retinopathy Screening Report")

    pdf_canvas.setFont("Times-Roman", 9.5)
    right_x = PAGE_WIDTH - PAGE_MARGIN
    pdf_canvas.drawRightString(right_x, top_y - 12, f"Report ID: {case_record['report_id']}")
    pdf_canvas.drawRightString(right_x, top_y - 25, f"Generated: {case_record['analysed_at_display']}")
    pdf_canvas.drawRightString(right_x, top_y - 38, "Page 1 of 1")

    pdf_canvas.setStrokeColor(INK_COLOUR)
    pdf_canvas.setLineWidth(1.2)
    pdf_canvas.line(PAGE_MARGIN, top_y - 46, PAGE_WIDTH - PAGE_MARGIN, top_y - 46)
    return top_y - 46


def draw_patient_details(pdf_canvas, case_record, top_y):
    patient_details = case_record["patient_details"]
    detail_fields = [
        ("Patient name", patient_details.get("patient_name")),
        ("Date of birth", patient_details.get("date_of_birth")),
        ("Sex", patient_details.get("sex")),
        ("Referring doctor", patient_details.get("referring_clinician")),
        ("Image file", case_record["file_name"]),
    ]
    column_width = (PAGE_WIDTH - 2 * PAGE_MARGIN) / len(detail_fields)
    for field_index, (field_label, field_value) in enumerate(detail_fields):
        column_x = PAGE_MARGIN + field_index * column_width
        pdf_canvas.setFont("Times-Italic", 8.5)
        pdf_canvas.setFillColor(SLATE_COLOUR)
        pdf_canvas.drawString(column_x, top_y - 12, field_label)
        pdf_canvas.setFont("Times-Roman", 10.5)
        pdf_canvas.setFillColor(INK_COLOUR)
        value_text = str(field_value).strip() if field_value else "Not recorded"
        while pdf_canvas.stringWidth(value_text, "Times-Roman", 10.5) > column_width - 10 and len(value_text) > 4:
            value_text = value_text[:-4] + "..."
        pdf_canvas.drawString(column_x, top_y - 24, value_text)

    bottom_y = top_y - 34
    pdf_canvas.setStrokeColor(RULE_COLOUR)
    pdf_canvas.setLineWidth(0.6)
    pdf_canvas.line(PAGE_MARGIN, bottom_y, PAGE_WIDTH - PAGE_MARGIN, bottom_y)
    return bottom_y


def draw_severity_scale(pdf_canvas, stage_names, averaged_probabilities, predicted_grade, left_x, top_y, total_width):
    draw_section_label(pdf_canvas, "Where this result sits on the ICDR severity scale", left_x, top_y - 10)
    segment_gap = 4
    segment_width = (total_width - segment_gap * (len(stage_names) - 1)) / len(stage_names)
    segment_top_y = top_y - 18
    segment_height = 16
    for grade_index, stage_name in enumerate(stage_names):
        segment_x = left_x + grade_index * (segment_width + segment_gap)
        severity_colour = colors.HexColor(SEVERITY_COLOURS[grade_index])
        is_predicted_grade = grade_index == predicted_grade
        pdf_canvas.setFillColor(severity_colour if is_predicted_grade else colors.HexColor("#E4E9EE"))
        pdf_canvas.setStrokeColor(severity_colour)
        pdf_canvas.setLineWidth(1.4 if is_predicted_grade else 0.6)
        pdf_canvas.rect(segment_x, segment_top_y - segment_height, segment_width, segment_height, stroke=1, fill=1)
        pdf_canvas.setFillColor(colors.white if is_predicted_grade else INK_COLOUR)
        pdf_canvas.setFont("Times-Bold" if is_predicted_grade else "Times-Roman", 9)
        pdf_canvas.drawCentredString(segment_x + segment_width / 2, segment_top_y - 11.5, f"{grade_index}  {stage_name}")
        pdf_canvas.setFillColor(SLATE_COLOUR)
        pdf_canvas.setFont("Times-Roman", 8.5)
        pdf_canvas.drawCentredString(segment_x + segment_width / 2, segment_top_y - segment_height - 10, f"{averaged_probabilities[grade_index]:.0%}")
    return segment_top_y - segment_height - 14


def draw_result_panel(pdf_canvas, case_record, left_x, top_y, panel_width):
    analysis = case_record["analysis"]
    predicted_grade = analysis["predicted_grade"]
    severity_colour = colors.HexColor(SEVERITY_COLOURS[predicted_grade])

    draw_section_label(pdf_canvas, "Model result", left_x, top_y - 10)
    pdf_canvas.setFont("Times-Bold", 22)
    pdf_canvas.setFillColor(severity_colour)
    pdf_canvas.drawString(left_x, top_y - 34, analysis["predicted_stage_name"])
    pdf_canvas.setFont("Times-Roman", 10)
    pdf_canvas.setFillColor(SLATE_COLOUR)
    pdf_canvas.drawString(left_x, top_y - 48, f"Grade {predicted_grade} of 4 on the international (ICDR) scale")

    metric_top_y = top_y - 64
    for metric_index, (metric_label, metric_value) in enumerate([
        ("Confidence in this grade", f"{analysis['confidence']:.0%}"),
        ("Chance of referable DR (grade 2+)", f"{analysis['referable_probability']:.0%}"),
    ]):
        metric_x = left_x + metric_index * (panel_width / 2)
        pdf_canvas.setFont("Times-Italic", 8.5)
        pdf_canvas.setFillColor(SLATE_COLOUR)
        pdf_canvas.drawString(metric_x, metric_top_y, metric_label)
        pdf_canvas.setFont("Times-Bold", 15)
        pdf_canvas.setFillColor(INK_COLOUR)
        pdf_canvas.drawString(metric_x, metric_top_y - 16, metric_value)

    recommendation_top_y = metric_top_y - 26
    guidance = analysis["guidance"]
    recommendation_text = f"<b>Suggested action:</b> {escape(guidance['action'])}<br/>{escape(guidance['finding'])}"
    recommendation_paragraph = Paragraph(recommendation_text, BODY_STYLE)
    _, recommendation_height = recommendation_paragraph.wrap(panel_width - 18, PAGE_HEIGHT)
    box_height = recommendation_height + 12
    pdf_canvas.setFillColor(PANEL_COLOUR)
    pdf_canvas.rect(left_x, recommendation_top_y - box_height, panel_width, box_height, stroke=0, fill=1)
    pdf_canvas.setFillColor(severity_colour)
    pdf_canvas.rect(left_x, recommendation_top_y - box_height, 4, box_height, stroke=0, fill=1)
    recommendation_paragraph.drawOn(pdf_canvas, left_x + 12, recommendation_top_y - 6 - recommendation_height)

    table_top_y = recommendation_top_y - box_height - 16
    draw_section_label(pdf_canvas, "Probability of each grade (average of three models)", left_x, table_top_y)
    bar_left_x = left_x + 92
    bar_full_width = panel_width - 92 - 34
    for grade_index, stage_name in enumerate(case_record["stage_names"]):
        row_y = table_top_y - 15 - grade_index * 13.5
        probability = float(analysis["averaged_probabilities"][grade_index])
        pdf_canvas.setFont("Times-Bold" if grade_index == predicted_grade else "Times-Roman", 9.5)
        pdf_canvas.setFillColor(INK_COLOUR)
        pdf_canvas.drawString(left_x, row_y, f"{grade_index}  {stage_name}")
        pdf_canvas.setFillColor(colors.HexColor("#E4E9EE"))
        pdf_canvas.rect(bar_left_x, row_y - 1, bar_full_width, 8, stroke=0, fill=1)
        pdf_canvas.setFillColor(colors.HexColor(SEVERITY_COLOURS[grade_index]))
        pdf_canvas.rect(bar_left_x, row_y - 1, max(bar_full_width * probability, 0.5), 8, stroke=0, fill=1)
        pdf_canvas.setFillColor(INK_COLOUR)
        pdf_canvas.setFont("Times-Roman", 9.5)
        pdf_canvas.drawRightString(left_x + panel_width, row_y, f"{probability:.1%}")

    agreement_top_y = table_top_y - 15 - 5 * 13.5 - 6
    draw_section_label(pdf_canvas, "Individual models", left_x, agreement_top_y)
    for model_index, (architecture_name, model_grade) in enumerate(analysis["per_model_grades"].items()):
        row_y = agreement_top_y - 14 - model_index * 12
        pdf_canvas.setFont("Times-Roman", 9.5)
        pdf_canvas.setFillColor(INK_COLOUR)
        pdf_canvas.drawString(left_x, row_y, ARCHITECTURE_DISPLAY_NAMES.get(architecture_name, architecture_name))
        model_confidence = float(analysis["per_model_probabilities"][architecture_name][model_grade])
        pdf_canvas.drawRightString(left_x + panel_width, row_y, f"{case_record['stage_names'][model_grade]} ({model_confidence:.0%})")
    agreement_text = "All three models agree." if analysis["all_models_agree"] else "The models do not all agree, see review notes."
    pdf_canvas.setFont("Times-Italic", 9)
    pdf_canvas.setFillColor(SLATE_COLOUR)
    final_y = agreement_top_y - 14 - 3 * 12
    pdf_canvas.drawString(left_x, final_y, agreement_text)
    return final_y


def draw_notes(pdf_canvas, case_record, left_x, top_y, available_width):
    analysis = case_record["analysis"]
    draw_section_label(pdf_canvas, "Image quality and review notes", left_x, top_y - 10)
    note_lines = [f"Image quality: {escape(warning)}" for warning in analysis["quality_warnings"]]
    note_lines += [f"Review: {escape(reason)}" for reason in analysis["review_reasons"]]
    if not note_lines:
        note_lines = ["No quality problems were detected and the models agreed with good confidence."]
    current_y = top_y - 16
    for note_line in note_lines[:5]:
        current_y = draw_wrapped_text(pdf_canvas, f"&#8226; {note_line}", NOTE_STYLE, left_x, current_y, available_width) - 1
    return current_y


def draw_sign_off(pdf_canvas, case_record, top_y):
    content_width = PAGE_WIDTH - 2 * PAGE_MARGIN
    draw_section_label(pdf_canvas, "Clinician comments", PAGE_MARGIN, top_y - 10)
    box_top_y = top_y - 15
    signature_row_height = 18
    box_height = max(26, min(50, box_top_y - signature_row_height - DISCLAIMER_TOP_Y - 6))
    pdf_canvas.setStrokeColor(RULE_COLOUR)
    pdf_canvas.setLineWidth(0.6)
    pdf_canvas.rect(PAGE_MARGIN, box_top_y - box_height, content_width, box_height, stroke=1, fill=0)
    comments_text = (case_record.get("clinician_comments") or "").strip()
    if comments_text:
        comment_paragraph = Paragraph(escape(comments_text).replace("\n", "<br/>"), NOTE_STYLE)
        _, comment_height = comment_paragraph.wrap(content_width - 12, box_height - 6)
        comment_paragraph.drawOn(pdf_canvas, PAGE_MARGIN + 6, box_top_y - 4 - min(comment_height, box_height - 6))

    signature_y = box_top_y - box_height - 14
    pdf_canvas.setFont("Times-Roman", 9.5)
    pdf_canvas.setFillColor(INK_COLOUR)
    for label_text, label_x, line_length in [("Reviewed by", PAGE_MARGIN, 210), ("Signature", PAGE_MARGIN + 300, 180), ("Date", PAGE_MARGIN + 560, 150)]:
        pdf_canvas.drawString(label_x, signature_y, label_text)
        label_width = pdf_canvas.stringWidth(label_text, "Times-Roman", 9.5)
        pdf_canvas.setStrokeColor(SLATE_COLOUR)
        pdf_canvas.line(label_x + label_width + 6, signature_y - 2, label_x + label_width + 6 + line_length, signature_y - 2)
    return signature_y - 10


def draw_disclaimer_and_footer(pdf_canvas, case_record):
    test_performance = case_record["test_set_performance"]
    disclaimer_text = (
        "Decision support only, not a diagnosis. Produced by three deep learning models trained on the public APTOS 2019 dataset "
        f"(on {test_performance.get('test_photos', 'held-out')} unseen test photos: quadratic weighted kappa {test_performance.get('ensemble_qwk', float('nan')):.2f}, "
        f"{test_performance.get('referable_dr_sensitivity', float('nan')):.0%} of referable cases caught). Not clinically validated. "
        "The heat map shows where one model looked, not confirmed disease. A qualified eye care professional must review this report before any clinical decision."
    )
    draw_wrapped_text(pdf_canvas, disclaimer_text, DISCLAIMER_STYLE, PAGE_MARGIN, DISCLAIMER_TOP_Y, PAGE_WIDTH - 2 * PAGE_MARGIN)
    pdf_canvas.setFont("Times-Roman", 7.5)
    pdf_canvas.setFillColor(SLATE_COLOUR)
    pdf_canvas.drawString(PAGE_MARGIN, PAGE_MARGIN - 8, f"Model version {case_record['model_version']}. Photo and results were processed in memory and not stored by this tool.")
    pdf_canvas.drawRightString(PAGE_WIDTH - PAGE_MARGIN, PAGE_MARGIN - 8, f"Report {case_record['report_id']}")


def build_screening_report_pdf(case_record):
    pdf_buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    pdf_canvas.setTitle(f"DR screening report {case_record['report_id']}")
    pdf_canvas.setAuthor(case_record["facility_name"] or "Diabetic eye screening service")

    header_bottom_y = draw_header(pdf_canvas, case_record)
    details_bottom_y = draw_patient_details(pdf_canvas, case_record, header_bottom_y)

    left_column_width = 470
    right_column_x = PAGE_MARGIN + left_column_width + 24
    right_column_width = PAGE_WIDTH - PAGE_MARGIN - right_column_x
    image_box_size = 140
    images_top_y = details_bottom_y - 10
    display_images = case_record["analysis"]["display_images"]
    grad_cam_name = ARCHITECTURE_DISPLAY_NAMES.get(case_record["analysis"]["grad_cam_architecture"], "best model")
    image_panels = [
        (display_images["original"], "Original photo as uploaded"),
        (display_images["resized"], f"After preprocessing: crop, pad to square, mask, {PREPROCESSING_CAPTION_NAMES.get(case_record['analysis'].get('preprocessing_method'), 'enhancement')}, {case_record['target_image_size']} px"),
        (display_images["grad_cam"], f"Grad-CAM heat map ({grad_cam_name}). Red areas influenced the grade most"),
    ]
    for panel_index, (panel_image, panel_caption) in enumerate(image_panels):
        draw_image_with_caption(pdf_canvas, panel_image, PAGE_MARGIN + panel_index * (left_column_width - image_box_size) / 2, images_top_y, image_box_size, panel_caption)

    scale_bottom_y = draw_severity_scale(
        pdf_canvas, case_record["stage_names"], case_record["analysis"]["averaged_probabilities"], case_record["analysis"]["predicted_grade"],
        PAGE_MARGIN, images_top_y - image_box_size - 28, left_column_width,
    )
    notes_bottom_y = draw_notes(pdf_canvas, case_record, PAGE_MARGIN, scale_bottom_y - 4, left_column_width)

    pdf_canvas.setStrokeColor(RULE_COLOUR)
    pdf_canvas.setLineWidth(0.6)
    pdf_canvas.line(right_column_x - 12, images_top_y, right_column_x - 12, min(notes_bottom_y, images_top_y - 290))
    result_bottom_y = draw_result_panel(pdf_canvas, case_record, right_column_x, images_top_y, right_column_width)

    sign_off_top_y = min(notes_bottom_y, result_bottom_y) - 6
    draw_sign_off(pdf_canvas, case_record, sign_off_top_y)
    draw_disclaimer_and_footer(pdf_canvas, case_record)

    pdf_canvas.showPage()
    pdf_canvas.save()
    return pdf_buffer.getvalue()