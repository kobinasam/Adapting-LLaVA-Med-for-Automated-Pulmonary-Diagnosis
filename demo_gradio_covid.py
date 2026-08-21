import gradio as gr
import torch
import torch.nn as nn
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer
from PIL import Image
import os
import time
import io
import tempfile
from datetime import datetime

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

disable_torch_init()

BASE_MODEL_PATH = "/scratch/bhwr/msam1/llava-med-checkpoints/stage2-covid"

print(f"Loading base model from {BASE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.cuda()

model.config.mm_use_im_start_end = False
model.config.mm_projector_type = "mlp2x_gelu"
model.config.mm_vision_select_layer = -2
model.config.mm_vision_tower = "openai/clip-vit-large-patch14-336"

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.float16)
image_processor = vision_tower.image_processor

print("Model loaded successfully!")


# ============================================================
# CORE INFERENCE
# ============================================================

def inference(image, question, temperature, max_tokens, history_state):
    """Run inference and append result to session history."""
    if history_state is None:
        history_state = []

    if image is None:
        return ("⚠️ Please upload a medical image first.", "—", "—", "—",
                history_state, format_history(history_state))

    if not question or question.strip() == "":
        return ("⚠️ Please enter a question.", "—", "—", "—",
                history_state, format_history(history_state))

    try:
        start_time = time.time()

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()

        conv = conv_templates["vicuna_v1"].copy()
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        question_with_image = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        do_sample = temperature > 0
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if "ASSISTANT:" in outputs:
            outputs = outputs.split("ASSISTANT:")[-1].strip()

        elapsed = time.time() - start_time
        num_tokens = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        history_state.append({
            "timestamp": datetime.now().isoformat(),
            "image": image.copy(),
            "question": question,
            "response": outputs,
            "elapsed": elapsed,
            "num_tokens": int(num_tokens),
            "temperature": temperature,
        })

        return (
            outputs,
            f"{elapsed:.1f}s",
            f"{num_tokens}",
            f"{tokens_per_sec:.1f} tok/s",
            history_state,
            format_history(history_state),
        )

    except Exception as e:
        import traceback
        return (
            f" Error: {str(e)}\n\n{traceback.format_exc()}",
            "—", "—", "—", history_state, format_history(history_state),
        )


def format_history(history_state):
    if not history_state:
        return "_No analyses yet. Upload an image and click 'Analyze Image' to start._"

    md = f"### 📚 Session History ({len(history_state)} analyses)\n\n"
    for i, entry in enumerate(history_state, 1):
        ts = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
        q_preview = entry["question"][:80] + ("..." if len(entry["question"]) > 80 else "")
        r_preview = entry["response"][:120] + ("..." if len(entry["response"]) > 120 else "")
        md += f"**#{i} · {ts}**\n\n"
        md += f"- **Q:** {q_preview}\n"
        md += f"- **A:** {r_preview}\n"
        md += f"-  {entry['elapsed']:.1f}s ·  {entry['num_tokens']} tokens\n\n"
        md += "---\n\n"
    return md


def clear_history():
    return [], format_history([])


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def _clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('\n', '<br/>')
    return text


def pil_to_reportlab(pil_image, max_width=2.6 * inch, max_height=2.6 * inch):
    img_buffer = io.BytesIO()
    rgb_image = pil_image.convert('RGB') if pil_image.mode != 'RGB' else pil_image
    rgb_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    w, h = pil_image.size
    aspect = w / h
    if aspect > max_width / max_height:
        new_w = max_width
        new_h = max_width / aspect
    else:
        new_h = max_height
        new_w = max_height * aspect

    return RLImage(img_buffer, width=new_w, height=new_h)


def generate_report(history_state, patient_id_field, clinician_field):
    if not history_state:
        return None, "No analyses to report. Run at least one analysis first."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"llava_med_report_{timestamp}.pdf"
    pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="LLaVA-Med Analysis Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontSize=22,
        textColor=HexColor('#2c5f8d'), spaceAfter=8,
        alignment=TA_LEFT, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=11,
        textColor=HexColor('#5a6c7d'), spaceAfter=4)
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'], fontSize=14,
        textColor=HexColor('#2c5f8d'), spaceBefore=14,
        spaceAfter=8, fontName='Helvetica-Bold')
    case_label_style = ParagraphStyle(
        'CaseLabel', parent=styles['Normal'], fontSize=10,
        textColor=HexColor('#00a6a6'), fontName='Helvetica-Bold',
        spaceAfter=2)
    q_label_style = ParagraphStyle(
        'QLabel', parent=styles['Normal'], fontSize=10,
        textColor=HexColor('#2c5f8d'), fontName='Helvetica-Bold',
        spaceBefore=6, spaceAfter=2)
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontSize=10, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=6)
    meta_style = ParagraphStyle(
        'Meta', parent=styles['Normal'], fontSize=8,
        textColor=HexColor('#718096'), fontName='Helvetica-Oblique')
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'], fontSize=9,
        textColor=HexColor('#78350f'), fontName='Helvetica-Oblique',
        alignment=TA_JUSTIFY, leading=12, spaceBefore=6, spaceAfter=6)

    story = []

    # Header
    story.append(Paragraph("LLaVA-Med Analysis Report", title_style))
    story.append(Paragraph(
        "Multimodal vision-language analysis of medical imaging", subtitle_style
    ))

    # Info table
    report_date = datetime.now().strftime("%B %d, %Y at %H:%M")
    pid = patient_id_field.strip() if patient_id_field else "—"
    clin = clinician_field.strip() if clinician_field else "—"
    info_data = [
        ['Report Generated:', report_date],
        ['Patient/Case ID:', pid],
        ['Clinician/Researcher:', clin],
        ['Total Analyses:', str(len(history_state))],
        ['Model:', 'LLaVA-Med (Vicuna-7B + CLIP-ViT-L/14-336), COVID-19 fine-tuned'],
    ]
    info_table = Table(info_data, colWidths=[1.7 * inch, 4.5 * inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#2c5f8d')),
        ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#1a202c')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f7fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, HexColor('#e2e8f0')),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.15 * inch))

    # Disclaimer
    disclaimer_table = Table(
        [[Paragraph(
            "<b>⚠ Research Use Only:</b> This report is generated by an AI model "
            "for research and demonstration purposes. It is NOT a substitute for "
            "professional medical diagnosis. All findings must be reviewed and "
            "validated by a qualified clinician.",
            disclaimer_style
        )]],
        colWidths=[6.2 * inch],
    )
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#fff8e6')),
        ('LINEBEFORE', (0, 0), (0, -1), 4, HexColor('#f59e0b')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 0.2 * inch))

    # Cases
    story.append(Paragraph("Analysis Findings", h2_style))

    for i, entry in enumerate(history_state, 1):
        case_elements = []
        ts = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")

        case_elements.append(Paragraph(
            f"CASE #{i} &middot; Captured at {ts}", case_label_style
        ))
        case_elements.append(Spacer(1, 0.05 * inch))

        try:
            rl_image = pil_to_reportlab(entry["image"])
        except Exception as e:
            rl_image = Paragraph(f"[Image could not be embedded: {e}]", body_style)

        content_paragraphs = [
            Paragraph("<b>Question:</b>", q_label_style),
            Paragraph(_clean_text(entry["question"]), body_style),
            Paragraph("<b>Model Response:</b>", q_label_style),
            Paragraph(_clean_text(entry["response"]), body_style),
        ]

        case_table = Table(
            [[rl_image, content_paragraphs]],
            colWidths=[2.8 * inch, 3.6 * inch],
        )
        case_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f7fafc')),
        ]))
        case_elements.append(case_table)

        meta = (
            f"Inference time: {entry['elapsed']:.1f}s &middot; "
            f"Tokens: {entry['num_tokens']} &middot; "
            f"Temperature: {entry['temperature']}"
        )
        case_elements.append(Spacer(1, 0.05 * inch))
        case_elements.append(Paragraph(meta, meta_style))
        case_elements.append(Spacer(1, 0.2 * inch))

        story.append(KeepTogether(case_elements))

    # Footer
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<i>Generated by LLaVA-Med COVID-19 Analysis Tool. "
        "Model: Vicuna-7B + CLIP ViT-L/14-336, fine-tuned on COVID-19 chest CT data. "
        "Eval metrics: BLEU-4 0.218, ROUGE-L 0.384, BERTScore F1 0.911.</i>",
        meta_style
    ))

    doc.build(story)
    return pdf_path, f"Report generated successfully ({len(history_state)} cases)."


# ============================================================
# UI
# ============================================================

PRESET_PROMPTS = [
    "Describe the image in detail.",
    "What abnormalities or findings can you identify in this scan?",
    "Is this CT scan consistent with COVID-19 pneumonia? Explain your reasoning.",
    "What anatomical structures are visible in this image?",
    "Are there ground-glass opacities or consolidations present?",
    "Describe the distribution and extent of any lung pathology.",
    "What is the most likely diagnosis based on this imaging?",
    "Compare the appearance of the lung fields bilaterally.",
]

custom_css = """
:root {
    --primary-color: #2c5f8d;
    --accent-color: #00a6a6;
    --card-shadow: 0 4px 16px rgba(0,0,0,0.06);
    --border-radius: 12px;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}

#header-block {
    background: linear-gradient(135deg, #2c5f8d 0%, #00a6a6 100%);
    border-radius: var(--border-radius);
    padding: 28px 32px;
    margin-bottom: 20px;
    color: white;
    box-shadow: var(--card-shadow);
}
#header-block h1 {
    color: white !important;
    font-size: 2.0em !important;
    margin: 0 0 6px 0 !important;
    font-weight: 700 !important;
}
#header-block p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.05em !important;
    margin: 0 !important;
}

.section-card {
    background: white;
    border-radius: var(--border-radius);
    padding: 20px;
    box-shadow: var(--card-shadow);
    margin-bottom: 16px;
    border: 1px solid #e8ecf1;
}

#response-output .wrap textarea {
    background: #ffffff !important;
    color: #1a202c !important;
    border: 1px solid #d1d9e0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    opacity: 1 !important;
}

#response-output .wrap textarea::placeholder {
    color: #a0aec0 !important;
    opacity: 1 !important;
}

.preset-button {
    background: white !important;
    border: 1px solid #d1d9e0 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    color: #4a5568 !important;
    transition: all 0.15s ease !important;
}
.preset-button:hover {
    background: var(--primary-color) !important;
    color: white !important;
    border-color: var(--primary-color) !important;
    transform: translateY(-1px);
}

.stat-card {
    background: white;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e8ecf1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.stat-label {
    font-size: 0.78em;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
    font-weight: 600;
}
.stat-value {
    font-size: 1.7em;
    color: var(--primary-color);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

#analyze-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--accent-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(44, 95, 141, 0.3) !important;
}
#analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(44, 95, 141, 0.4) !important;
}

#report-btn {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
}
#report-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(5, 150, 105, 0.4) !important;
}

#clear-btn, #clear-history-btn {
    background: white !important;
    color: #4a5568 !important;
    border: 1.5px solid #d1d9e0 !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-size: 15px !important;
}

.disclaimer {
    background: #fff8e6;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 8px;
    margin-top: 16px;
    font-size: 13px;
    color: #78350f;
}

footer { display: none !important; }
"""


with gr.Blocks(
    title="LLaVA-Med | COVID-19 CT Analysis",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    css=custom_css
) as demo:

    history_state = gr.State([])

    gr.HTML("""
    <div id="header-block">
        <h1> LLaVA-Med: COVID-19 CT Analysis</h1>
        <p>Multimodal vision-language model fine-tuned for chest CT interpretation and visual question answering.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # LEFT
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 📤 Upload Medical Image")
                image_input = gr.Image(type="pil", label="", show_label=False, height=420)

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Ask a Question")
                question_input = gr.Textbox(
                    label="",
                    placeholder="e.g., Describe the image in detail.",
                    lines=3,
                    value="Describe the image in detail.",
                    show_label=False,
                )

                gr.Markdown("**Quick prompts:**")
                preset_btns = []
                with gr.Row():
                    for p in PRESET_PROMPTS[:4]:
                        btn = gr.Button(
                            p[:35] + ("..." if len(p) > 35 else ""),
                            size="sm", elem_classes=["preset-button"]
                        )
                        preset_btns.append((btn, p))
                with gr.Row():
                    for p in PRESET_PROMPTS[4:8]:
                        btn = gr.Button(
                            p[:35] + ("..." if len(p) > 35 else ""),
                            size="sm", elem_classes=["preset-button"]
                        )
                        preset_btns.append((btn, p))

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                            label="Temperature",
                                            info="0 = deterministic, 1 = creative")
                    max_tokens = gr.Slider(64, 2048, value=512, step=64,
                                           label="Max Tokens")

                with gr.Row():
                    submit_btn = gr.Button("🔬 Analyze Image", variant="primary",
                                           scale=3, elem_id="analyze-btn")
                    clear_btn = gr.Button("Clear", scale=1, elem_id="clear-btn")

        # RIGHT
        with gr.Column(scale=5):
            with gr.Row():
                stat_time = gr.HTML("""<div class="stat-card"><div class="stat-label">Time</div><div class="stat-value">—</div></div>""")
                stat_tokens = gr.HTML("""<div class="stat-card"><div class="stat-label">Tokens</div><div class="stat-value">—</div></div>""")
                stat_speed = gr.HTML("""<div class="stat-card"><div class="stat-label">⚡ Speed</div><div class="stat-value">—</div></div>""")

            with gr.Group(elem_classes=["section-card"]):
                with gr.Tabs():
                    with gr.Tab("🩺 Model Analysis"):
                        output = gr.Textbox(
                            label="", lines=14, show_copy_button=True,
                            elem_id="response-output", show_label=False,
                            placeholder="Upload an image and click 'Analyze Image'..."
                        )
                    with gr.Tab("📚 Session History"):
                        history_display = gr.Markdown(
                            "_No analyses yet. Upload an image and click 'Analyze Image' to start._"
                        )
                        clear_hist_btn = gr.Button("🗑️ Clear Session History",
                                                   elem_id="clear-history-btn")
                    with gr.Tab("About"):
                        gr.Markdown("""
                        ### Model Details
                        - **Architecture:** LLaVA-Med (Vicuna-7B + CLIP-ViT-L/14-336)
                        - **Fine-tuning:** ~14K image-question-answer triples on COVID-19 CT data
                        - **Evaluation:** BLEU-4 0.218 · ROUGE-L 0.384 · BERTScore F1 0.911

                        ### How It Works
                        1. Upload a chest CT image
                        2. Ask a question or pick a quick prompt
                        3. Each analysis is automatically saved to your session
                        4. Click "Generate PDF Report" below to download a compiled report
                        """)

            gr.HTML("""
            <div class="disclaimer">
                <strong>Research Use Only:</strong> Not for clinical decision-making.
                Always consult a qualified radiologist.
            </div>
            """)

    # ==================== REPORT SECTION ====================
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### Generate PDF Report")
        gr.Markdown(
            "Compile all analyses from your current session into a downloadable PDF. "
            "Each case includes the uploaded image, question, model response, and metadata."
        )

        with gr.Row():
            patient_id = gr.Textbox(
                label="Patient/Case ID (optional)",
                placeholder="e.g., CT-2026-001",
                scale=1,
            )
            clinician_name = gr.Textbox(
                label="Clinician/Researcher (optional)",
                placeholder="e.g., Dr. Smith",
                scale=1,
            )

        generate_report_btn = gr.Button(
            "Generate PDF Report", variant="primary", elem_id="report-btn"
        )

        report_status = gr.Markdown("")
        report_file = gr.File(label="Download Report", interactive=False)

    # ==================== HANDLERS ====================

    def update_stats(response, t, tok, speed, hist, hist_md):
        time_html = f'<div class="stat-card"><div class="stat-label">Time</div><div class="stat-value">{t}</div></div>'
        tokens_html = f'<div class="stat-card"><div class="stat-label">Tokens</div><div class="stat-value">{tok}</div></div>'
        speed_html = f'<div class="stat-card"><div class="stat-label">⚡ Speed</div><div class="stat-value">{speed}</div></div>'
        return response, time_html, tokens_html, speed_html, hist, hist_md

    def run_inference(image, question, temp, max_tok, hist):
        response, t, tok, speed, new_hist, hist_md = inference(image, question, temp, max_tok, hist)
        return update_stats(response, t, tok, speed, new_hist, hist_md)

    submit_btn.click(
        fn=run_inference,
        inputs=[image_input, question_input, temperature, max_tokens, history_state],
        outputs=[output, stat_time, stat_tokens, stat_speed, history_state, history_display],
        show_progress="full",
    )

    for btn, prompt_text in preset_btns:
        btn.click(fn=lambda p=prompt_text: p, outputs=question_input)

    def clear_current():
        empty = lambda label: f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">—</div></div>'
        return (None, "Describe the image in detail.", "",
                empty("Time"), empty("Tokens"), empty("Speed"))

    clear_btn.click(
        fn=clear_current,
        outputs=[image_input, question_input, output, stat_time, stat_tokens, stat_speed],
    )

    clear_hist_btn.click(
        fn=clear_history,
        outputs=[history_state, history_display],
    )

    generate_report_btn.click(
        fn=generate_report,
        inputs=[history_state, patient_id, clinician_name],
        outputs=[report_file, report_status],
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting LLaVA-Med Gradio Server")
    print(f"   Base Model: {BASE_MODEL_PATH}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )