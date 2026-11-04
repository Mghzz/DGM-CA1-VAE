"""Build the complete, professionally formatted Word report for DGM HW1 Question 2."""
import os
import sys
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D1D5DB", sz="4"):
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)

def add_p(doc, text="", align=WD_ALIGN_PARAGRAPH.RIGHT, space_before=3, space_after=4, line_spacing=1.25):
    p = doc.add_paragraph()
    p.alignment = align
    pPr = p._p.get_or_add_pPr()
    pPr.append(OxmlElement('w:bidi'))
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        add_text(p, text)
    return p

def add_text(p, text, font_name="B Nazanin", size_pt=11, bold=False, italic=False, color_rgb=None):
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb
    rPr = run._r.get_or_add_rPr()
    rPr.append(OxmlElement('w:rtl'))
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:cs'), font_name)
    return run

def add_latin(p, text, font_name="Times New Roman", size_pt=10.5, bold=False, italic=False, color_rgb=None):
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb
    return run

def add_heading_1(doc, text):
    p = add_p(doc, space_before=16, space_after=8)
    add_text(p, text, font_name="B Nazanin", size_pt=15, bold=True, color_rgb=RGBColor(30, 58, 138))
    return p

def add_heading_2(doc, text):
    p = add_p(doc, space_before=11, space_after=6)
    add_text(p, text, font_name="B Nazanin", size_pt=13, bold=True, color_rgb=RGBColor(31, 41, 55))
    return p

def add_heading_3(doc, text):
    p = add_p(doc, space_before=8, space_after=4)
    add_text(p, text, font_name="B Nazanin", size_pt=11.5, bold=True, color_rgb=RGBColor(55, 65, 81))
    return p

def add_caption(doc, text):
    p = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=10)
    add_text(p, text, font_name="B Nazanin", size_pt=10, bold=True, color_rgb=RGBColor(75, 85, 99))
    return p

def add_image_centered(doc, image_path, width_inches=6.0, caption=""):
    if os.path.exists(image_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        run = p_img.add_run()
        run.add_picture(str(image_path), width=Inches(width_inches))
        if caption:
            add_caption(doc, caption)
    else:
        print(f"Warning: Image not found: {image_path}")

def build_complete_document():
    # Create document
    doc = docx.Document()
    
    # Page setup (A4, 23mm margins)
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.9)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    
    # -------------------------------------------------------------
    # Cover Page Header
    # -------------------------------------------------------------
    p0 = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=20, space_after=10)
    add_text(p0, "به نام خدا", font_name="B Nazanin", size_pt=14, bold=True)

    # University logo/name
    p_uni = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=5, space_after=4)
    add_text(p_uni, "دانشگاه تهران\nپردیس دانشکدگان فنی - دانشکده مهندسی برق و کامپیوتر", font_name="B Nazanin", size_pt=13, bold=True)

    p_course = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=15, space_after=6)
    add_text(p_course, "درس مدل‌های مولد عمیق (Deep Generative Models)", font_name="B Nazanin", size_pt=14, bold=True, color_rgb=RGBColor(30, 58, 138))

    p_title = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=10)
    add_text(p_title, "گزارش تمرین اول: پیاده‌سازی خودرمزگذار متغیر (VAE) و تحلیل تجربی توزیع‌های پیشین", font_name="B Nazanin", size_pt=16, bold=True, color_rgb=RGBColor(17, 24, 39))

    p_meta = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=15, space_after=25)
    add_text(p_meta, "استاد درس: دکتر مصطفی توسلی‌پور\nنیم‌سال اول سال تحصیلی ۱۴۰۵-۱۴۰۴", font_name="B Nazanin", size_pt=12)

    doc.add_page_break()

    # -------------------------------------------------------------
    # Table of Contents
    # -------------------------------------------------------------
    add_heading_2(doc, "فهرست مطالب")
    toc_items = [
        ("سوال اول: پرسش‌های تئوری و تحلیلی", "۲"),
        ("سوال دوم: پیاده‌سازی مدل خودرمزگذار متغیر (VAE) و تحلیل تجربی توزیع‌های پیشین", "۳"),
        ("    ۱. معماری شبکه، تابع هزینه و دینامیک آموزش", "۳"),
        ("    ۲. تحلیل کیفیت بازسازی و توزیع داده‌های تولیدی در فضای PCA", "۴"),
        ("    ۳. تحلیل هندسه فضای نهفته و میزان بهره‌برداری از مؤلفه‌های پیشین", "۵"),
        ("    ۴. ارزیابی کمی: معیارهای MMD، پوشش خوشه‌ها و پاسخ به سوالات ۱ تا ۳", "۶"),
        ("    ۵. تحلیل پژوهشی، مقایسه با VampPrior و جمع‌بندی", "۷"),
        ("مراجع و منابع علمی", "۸")
    ]
    for item, page in toc_items:
        p_toc = add_p(doc, space_before=2, space_after=2)
        add_text(p_toc, item, font_name="B Nazanin", size_pt=10.5)
        run_tab = p_toc.add_run(f"  ...........................................................................................  {page}")
        run_tab.font.name = "Times New Roman"
        run_tab.font.size = Pt(10)

    doc.add_page_break()

    # -------------------------------------------------------------
    # Question 1 (Placeholder for theory)
    # -------------------------------------------------------------
    add_heading_1(doc, "سوال اول: پرسش‌های تئوری و تحلیلی")
    p_q1 = add_p(doc)
    add_text(p_q1, "[پاسخ پرسش‌های بخش تئوری و محاسباتی تمرین اول در این قسمت قرار می‌گیرد.]")

    doc.add_page_break()

    # -------------------------------------------------------------
    # Question 2 (Full VAE Report)
    # -------------------------------------------------------------
    add_heading_1(doc, "سوال دوم: پیاده‌سازی مدل خودرمزگذار متغیر (VAE) و تحلیل تجربی توزیع‌های پیشین")
    
    p = add_p(doc)
    add_text(p, "در این پژوهش، مدل خودرمزگذار متغیر (")
    add_latin(p, "Variational Autoencoder - VAE")
    add_text(p, ") بر روی یک مجموعه داده ساخت‌یافته ۲۰‌بعدی با ۸ مد خوشه‌ای غیرخطی پیاده‌سازی و ارزیابی شده است. هدف بنیادین این آزمایش، بررسی و مقایسه اثرات هندسی و آماری چهار ساختار توزیع پیشین (")
    add_latin(p, "Prior Distribution")
    add_text(p, ") بر کیفیت بازسازی داده‌ها، وفاداری نمونه‌های تولیدی، انطباق با پسین تجمیعی و هندسه فضای نهفته دوبعدی است:")
    
    p_models = add_p(doc, space_before=2, space_after=4)
    add_text(p_models, "۱) ")
    add_latin(p_models, "Model A")
    add_text(p_models, ": توزیع پیشین گاوسی استاندارد همسانگرد ")
    add_latin(p_models, "N(0, I_2)")
    add_text(p_models, " با واگرایی KL تحلیلی بسته.\n۲) ")
    add_latin(p_models, "Model B")
    add_text(p_models, ": توزیع پیشین ترکیب گاوسی ۸‌تایی ثابت روی دایره‌ای به شعاع ۲ و انحراف معیار ۰.۴ با واگرایی KL مونت‌کارلو.\n۳) ")
    add_latin(p_models, "Model C")
    add_text(p_models, ": توزیع پیشین ترکیب گاوسی ۸‌تایی با میانگین‌ها و واریانس‌های قطری یادگیرنده.\n۴) ")
    add_latin(p_models, "Model D")
    add_text(p_models, ": توزیع پیشین ترکیب گاوسی ۴‌تایی با میانگین‌ها و واریانس‌های قطری یادگیرنده.")

    # 1. Architecture
    add_heading_2(doc, "۱. معماری شبکه، تابع هزینه و دینامیک آموزش")
    
    p = add_p(doc)
    add_text(p, "مجموعه داده شامل ۹۶۰۰ نمونه ۲۰‌بعدی است که از ۸ خوشه دایره‌ای با ۱۲۰۰ نمونه به ازای هر خوشه در فضای منبع دوبعدی منشأ می‌گیرد و از طریق نگاشت‌های غیرخطی مثلثاتی، چندجمله‌ای و اغتشاش نویز گاوسی با انحراف معیار ۰.۰۷ تولید شده است. داده‌ها به تفکیک ۷۰٪ آموزش (۶۷۲۰ نمونه)، ۱۵٪ اعتبارسنجی (۱۴۴۰ نمونه) و ۱۵٪ آزمون (۱۴۴۰ نمونه) افراز شده و بر مبنای آماره‌های بخش آموزش استانداردسازی شده‌اند. برچسب خوشه‌ها صرفاً برای ارزیابی و مصورسازی به کار رفته و به مدل داده نشده است.")

    p = add_p(doc)
    add_text(p, "ساختار شبکه‌های رمزگذار و رمزگشا مطابق مشخصات استاندارد طراحی شده است:\n• ")
    add_latin(p, "Encoder: 20 -> 128 (ReLU) -> 128 (ReLU) -> (mu: 2, log_var: 2)")
    add_text(p, "، با استفاده از ترفند بازپارامتری‌سازی.\n• ")
    add_latin(p, "Decoder: 2 -> 128 (ReLU) -> 128 (ReLU) -> 20 (mu_x)")
    add_text(p, "، با درست‌نمایی گاوسی همسانگرد با واریانس مشاهداتی ثابت ")
    add_latin(p, "sigma_x^2 = 0.09")
    add_text(p, ".\nتابع هزینه منفی لگاریتم درست‌نمایی (NLL) برای بعد ۲۰ به صورت تحلیلی به فرم ")
    add_latin(p, "L_recon = (1 / 0.18) * ||x - mu_x||^2 - 5.7006")
    add_text(p, " محاسبه می‌شود که ضریب ۵.۵۵۵۶ نقش ترازکننده بسیار حیاتی در برابر واگرایی KL دارد.")

    p = add_p(doc)
    add_text(p, "کلیه مدل‌ها با بهینه‌ساز Adam، نرخ یادگیری 0.001، اندازه دسته ۶۴، و شرط توقف زودهنگام با شکیبایی ۱۵ دوره روی اتلاف اعتبارسنجی آموزش داده شدند. روند همگرایی منحنی‌های اتلاف کل، اتلاف بازسازی و واگرایی KL در شکل ۱ ترسیم شده است. بهترین دوره‌های ثبت‌شده عبارتند از: مدل A دوره ۹۹، مدل B دوره ۹۷، مدل C دوره ۱۰۰، و مدل D دوره ۹۹.")

    add_image_centered(doc, "results/figures/learning_curves_all_models.png", width_inches=6.0, 
                       caption="شکل ۱- منحنی‌های آموزش و اعتبارسنجی اتلاف کل، اتلاف بازسازی و واگرایی KL در هر چهار مدل.")

    # 2. Recon and PCA
    add_heading_2(doc, "۲. تحلیل کیفیت بازسازی و توزیع داده‌های تولیدی در فضای PCA")

    p = add_p(doc)
    add_text(p, "ارزیابی خطای بازسازی (Reconstruction MSE) بر روی ۱۴۴۰ نمونه بخش آزمون با استفاده از میانگین تعیینی پسین نشان می‌دهد که مدل D با خطای هر عنصر ۰.۰۷۷۶۴ و مدل A با خطای ۰.۰۷۸۰۲ دارای کمترین خطای بازسازی هستند. در مقابل، مدل‌های چندمدی C با ۰.۰۸۲۲۱ و B با ۰.۰۸۲۴۶ خطای بازسازی اندکی بالاتر دارند. علت فیزیکی این تفاوت آن است که توزیع‌های پیشین تک‌مدی یا کم‌مؤلفه، انعطاف بیشتری به رمزگذار می‌دهند تا بازنمایی نهفته را برای حداقل‌سازی خطا تنظیم کند؛ در حالی که پیشین‌های ۸‌مؤلفه‌ای قید ساختاری سخت‌گیرانه‌تری بر پیوستگی فضای نهفته اعمال می‌کنند.")

    p = add_p(doc)
    add_text(p, "برای هر مدل، ۲۰۰۰ نمونه مستقل از پیشین نمونه‌برداری و بازسازی شدند. تصویر دوبعدی PCA این نمونه‌ها بر روی فضای فیت‌شده منحصربه‌فرد بخش آموزش (بدون نشت اطلاعات) در شکل ۲ نمایش داده شده است:")

    add_image_centered(doc, "results/figures/pca_test_vs_generated_all_grid.png", width_inches=6.2,
                       caption="شکل ۲- تصویر دوبعدی PCA از نمونه‌های واقعی آزمون (آبی) در برابر نمونه‌های تولیدی از پیشین (قرمز) در هر چهار مدل.")

    p = add_p(doc)
    add_text(p, "مشاهده می‌شود که در مدل A، به دلیل پیشین تک‌مدی پیوسته، نمونه‌های متعددی در حفره مرکزی (که در داده‌های واقعی تهی است) تولید شده‌اند (Mode Bridging). در مدل B، تطابق بی‌نظیری با ۸ مد واقعی مشاهده می‌شود. در مدل C، خوشه‌ها به درستی تفکیک شده‌اند و هیچ نمونه کاذبی در مرکز وجود ندارد. در مدل D، به دلیل وجود تنها ۴ مؤلفه برای ۸ مد، ادغام جفتی خوشه‌ها به وضوح رخ داده است.")

    # 3. Latent space
    add_heading_2(doc, "۳. تحلیل هندسه فضای نهفته و میزان بهره‌برداری از مؤلفه‌ها")

    p = add_p(doc)
    add_text(p, "شکل ۳ ساختار فضای نهفته دوبعدی شامل کدهای نهفته نمونه‌های آزمون به همراه خطوط تراز چگالی و مراکز مؤلفه‌های پیشین را به تصویر می‌کشد:")

    add_image_centered(doc, "results/figures/latent_space_all_grid.png", width_inches=5.8,
                       caption="شکل ۳- فضای نهفته دوبعدی: کدهای نهفته آزمون به همراه مراکز و کانتورهای بیضوی توزیع پیشین در مدل‌های A، B، C و D.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۴ (میزان استفاده از مؤلفه‌ها بر اساس چگالی بیشینه):")
    add_text(p, "ارزیابی میزان بهره‌برداری از مؤلفه‌های پیشین بر اساس تخصیص سخت بیشترین چگالی و میانگین مسئولیت پسین نشان می‌دهد:\n"
                "• مدل B (۸ مؤلفه ثابت): سهم تخصیص سخت بین ۱۱.۷۴٪ تا ۱۳.۶۸٪ و میانگین مسئولیت بین ۱۱.۷۶٪ تا ۱۳.۷۱٪ کاملاً متعادل است (مقدار ایده‌آل یکنواخت ۱۲.۵٪). هیچ مؤلفه‌ای دچار زوال نشده است.\n"
                "• مدل C (۸ مؤلفه یادگیرنده): تمامی ۸ مؤلفه فعال هستند (سهم تخصیص ۱۰.۰۰٪ تا ۱۷.۳۶٪ و میانگین مسئولیت ۱۰.۸۸٪ تا ۱۶.۸۵٪). مراکز یادگرفته‌شده در شعاع‌های مختلف حول مبدا قرار گرفته و ساختار چندخوشه‌ای را با موفقیت تفکیک کرده‌اند.\n"
                "• مدل D (۴ مؤلفه یادگیرنده): هر ۴ مؤلفه به طور متوازن (سهم ۲۳.۷۵٪ تا ۲۶.۶۷٪) مورد استفاده قرار گرفته‌اند و هر مؤلفه تقریب زننده حدود ۲ خوشه واقعی است.")

    # 4. Quantitative evaluation
    add_heading_2(doc, "۴. ارزیابی کمی: معیارهای MMD، پوشش خوشه‌ها و پاسخ به سوالات ۱ تا ۳")

    p = add_p(doc)
    add_text(p, "جدول ۱ نتایج کلیه معیارهای کمی را خلاصه می‌کند:")

    # Table 1
    t = doc.add_table(rows=5, cols=9)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t, color="9CA3AF", sz="6")

    headers = [
        "مدل", "نوع پیشین", "بهترین دوره", 
        "MSE هر عنصر", "MSE هر نمونه", 
        "MMD² داده", "MMD² نهان", 
        "نرخ اعتبار (%)", "پوشش خوشه‌ها"
    ]
    for col_idx, text in enumerate(headers):
        cell = t.cell(0, col_idx)
        set_cell_background(cell, "374151")
        set_cell_margins(cell, top=120, bottom=120, left=80, right=80)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p_cell = cell.paragraphs[0]
        p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pPr = p_cell._p.get_or_add_pPr()
        pPr.append(OxmlElement('w:bidi'))
        add_text(p_cell, text, font_name="B Nazanin", size_pt=9.5, bold=True, color_rgb=RGBColor(255, 255, 255))

    rows_data = [
        ["مدل A", "گاوسی استاندارد", "۹۹", "۰.۰۷۸۰۲", "۱.۵۶۰۴", "۰.۰۰۶۶۳۷", "۰.۰۱۱۱۶۱", "۸۰.۱۰٪", "۸ از ۸"],
        ["مدل B", "۸ مؤلفه ثابت", "۹۷", "۰.۰۸۲۴۶", "۱.۶۴۹۲", "۰.۰۰۰۹۴۳", "-۰.۰۰۰۳۷۲", "۹۵.۲۵٪", "۸ از ۸"],
        ["مدل C", "۸ مؤلفه یادگیرنده", "۱۰۰", "۰.۰۸۲۲۱", "۱.۶۴۴۲", "۰.۰۰۱۹۹۵", "۰.۰۰۳۲۱۴", "۹۵.۶۵٪", "۸ از ۸"],
        ["مدل D", "۴ مؤلفه یادگیرنده", "۹۹", "۰.۰۷۷۶۴", "۱.۵۵۲۸", "۰.۰۰۱۸۲۵", "۰.۰۰۰۵۸۸", "۹۳.۰۰٪", "۸ از ۸"]
    ]

    for row_idx, rdata in enumerate(rows_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(rdata):
            cell = t.cell(row_idx, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p_cell = cell.paragraphs[0]
            p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pPr = p_cell._p.get_or_add_pPr()
            pPr.append(OxmlElement('w:bidi'))
            bold = True if ("۰.۰۷۷۶۴" in val or "۰.۰۰۰۹۴۳" in val or "۹۵.۶۵٪" in val) else False
            color = RGBColor(17, 24, 39)
            add_text(p_cell, val, font_name="B Nazanin", size_pt=9.5, bold=bold, color_rgb=color)

    add_caption(doc, "جدول ۱- مقایسه جامع عملکرد چهار مدل VAE در معیارهای بازسازی، MMD²، اعتبار و پوشش خوشه‌ها.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۱ (رتبه‌بندی مدل‌ها و تحلیل بده‌بستان بازسازی-تولید):")
    add_text(p, "• از نظر کیفیت تولید داده: بر مبنای MMD² داده‌ها: B (0.00094) > D (0.00183) > C (0.00199) >> A (0.00664). بر مبنای نرخ اعتبار خوشه‌ای: C (95.65%) ≈ B (95.25%) > D (93.00%) >> A (80.10%). مدل‌های چندمدی با اختلاف فاحش برتر از گاوسی استاندارد هستند.\n"
                "• از نظر کیفیت بازسازی: رتبه‌بندی معکوس است: D (0.07764) ≈ A (0.07802) > C (0.08221) ≈ B (0.08246).\n"
                "• تحلیل مصالحه (Trade-off): نرمی فضای نهفته در مدل‌های تک‌مدی و کم‌مؤلفه منجر به بازسازی بهتر ولی تولید داده ضعیف‌تر می‌شود؛ در حالی که پیشین‌های غنی ۸‌مؤلفه‌ای کیفیت تولید را بیش از ۴ برابر بهبود می‌بخشند با افت جزئی ۶ درصدی در بازسازی.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۲ (مقایسه رتبه‌بندی MMD و پوشش خوشه‌ای):")
    add_text(p, "معیار MMD² فاصله‌ای در فضای هیلبرت با هسته بازتولیدی (RKHS) است که همخوانی توزیع تجمعی سراسری نمونه‌ها را در کل فضای ۲۰‌بعدی می‌سنجد. در مقابل، معیار پوشش خوشه‌ای ویژگی هندسی موضعی است که حضور نمونه‌ها درون ابربیضی صدک ۹۵ام خوشه‌های مرجع را می‌سنجد. مدل D در توزیع کلی سراسری خطای MMD کمی دارد اما به دلیل فشرده‌سازی ۸ خوشه به ۴ مؤلفه، نرخ اعتبار موضعی کمتری نسبت به مدل C با ۸ مؤلفه کسب می‌کند.")

    add_image_centered(doc, "results/figures/cluster_coverage_comparison.png", width_inches=5.6,
                       caption="شکل ۴- نرخ اعتبار کلی و پراکندگی نمونه‌های معتبر در هر یک از خوشه‌های ۸‌گانه در فضای ۲۰‌بعدی.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۳ (مقایسه مدل C با ۸ مؤلفه و مدل D با ۴ مؤلفه):")
    add_text(p, "مطابق شکل ۴، هر دو مدل شرط پوشش را با موفقیت کامل (۸ از ۸ خوشه) احراز کرده‌اند. با این وجود، مدل C به نرخ اعتبار ۹۵.۶۵٪ در برابر ۹۳.۰۰٪ مدل D دست یافته است؛ زیرا مدل C ساختار حقیقی ۸ خوشه‌ای داده را بدون فشرده‌سازی اجباری بازنمایی می‌کند.")

    # 5. Research questions
    add_heading_2(doc, "۵. تحلیل پژوهشی، مقایسه با VampPrior و جمع‌بندی")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۵ (اثر سوگیری استقرایی مدل B در برابر مدل A):")
    add_text(p, "مدل B با بهره‌گیری از دانش صریح مسئله (۸ مؤلفه متقارن دایره‌ای) سوگیری استقرایی قوی دارد که مانع از ایجاد چاله تهی مرکزی و ناهمخوانی پسین-پیشین می‌شود و بهترین MMD داده‌ها (۰.۰۰۰۹۴) و نهان را به ارمغان می‌آورد.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۶ (پیشین یادگیرنده مدل C در برابر پیشین ثابت مدل B):")
    add_text(p, "پیشین یادگیرنده انعطاف‌پذیری بالایی دارد و به مدل اجازه می‌دهد موقعیت و پراکندگی خوشه‌ها را بر اساس پویایی بهینه‌سازی کند که منجر به بالاترین نرخ اعتبار خوشه‌ای (۹۵.۶۵٪) شده است؛ هرچند آموزش توأم مراکز پیشین خطای MMD را اندکی نسبت به ساختار متقارن ایده‌آل B افزایش داده است.")

    p = add_p(doc)
    add_heading_3(doc, "پاسخ به سوال تحلیلی ۷ (چرا ترکیب گاوسی یادگیرنده یک VampPrior نیست؟):")
    add_text(p, "۱) در ترکیب گاوسی یادگیرنده، پارامترهای میانگین و واریانس مستقیماً در فضای نهفته دوبعدی قرار دارند و جدا از رمزگذار بهینه‌سازی می‌شوند. در مقابل، در روش VampPrior (تامچاک و ولینگ، ۲۰۱۸)، پیشین به عنوان ترکیبی از توزیع‌های شرطی رمزگذار روی مجموعه‌ای از شبه‌ورودی‌های یادگیرنده u_k در فضای داده‌ها تعریف می‌شود: ")
    add_latin(p, "p(z) = (1/K) * sum_{k=1}^K q_phi(z | u_k)")
    add_text(p, ".\n۲) هدف VampPrior تقریب مستقیم پسین تجمیعی (Aggregated Posterior) از طریق هدایت گرادیان‌ها به وزن‌های رمزگذار است، در حالی که پیشین ترکیب گاوسی معمولی فاقد اتصال دینامیک به نمایش‌های رمزگذار است.")

    p = add_p(doc)
    add_heading_3(doc, "محدودیت‌ها و ملاحظات آزمایشگاهی:")
    add_text(p, "۱) ارزیابی حاضر تک‌بذره است و تخمین واریانس نتایج نیازمند تکرار چندبذره است.\n"
                "۲) مقدار منفی MMD² در مدل B نتیجه ذاتی برآوردگر بدون‌سوگیری با واریانس محدود نمونه‌ای است و نباید به عنوان برابری قطعی توزیع‌ها تفسیر شود.\n"
                "۳) نرخ اعتبار وابسته به آستانه شعاع صدک ۹۵ام محاسبه‌شده از داده‌های اعتبارسنجی است.")

    p = add_p(doc)
    add_heading_3(doc, "جمع‌بندی:")
    add_text(p, "نتایج تجربی نشان داد که جایگزینی پیشین استاندارد گاوسی با پیشین‌های غنی ترکیب گاوسی، مشکل تولید نمونه‌های نامعتبر را برطرف ساخته و وفاداری تولید داده را به طرز چشمگیری ارتقا می‌دهد.")

    # References
    add_heading_2(doc, "مراجع")
    p_ref1 = add_p(doc, space_before=4, space_after=3)
    add_text(p_ref1, "[1] J. Tomczak and M. Welling, \"VAE with a VampPrior,\" in ")
    add_latin(p_ref1, "International Conference on Artificial Intelligence and Statistics (AISTATS)")
    add_text(p_ref1, ", PMLR 84, pp. 1214–1223, 2018.")

    p_ref2 = add_p(doc, space_before=2, space_after=3)
    add_text(p_ref2, "[2] D. P. Kingma and M. Welling, \"Auto-encoding variational Bayes,\" in ")
    add_latin(p_ref2, "International Conference on Learning Representations (ICLR)")
    add_text(p_ref2, ", 2014.")

    out_docx = Path("DGM_Report_Template.docx")
    doc.save(out_docx)
    print(f"Report successfully saved to: {out_docx.resolve()}")
    
    backup_docx = Path("DGM_Report_Question2_Complete.docx")
    doc.save(backup_docx)
    print(f"Backup report saved to: {backup_docx.resolve()}")

if __name__ == "__main__":
    build_complete_document()
