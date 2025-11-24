# pdf2md.py

import sys
from pathlib import Path
import hashlib
import yaml

# 导入 docling 相关的模块
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem, TextItem, SectionHeaderItem, TableItem
from docling.datamodel.pipeline_options import EasyOcrOptions

# 确保 Pillow 已安装
try:
    from PIL import Image
except ImportError:
    print("错误：Pillow 库未安装。请运行 'pip install Pillow' 进行安装。")
    sys.exit(1)


def convert_pdf_to_markdown_with_images(
        pdf_path: Path,
        model_dir: Path,
        output_dir: Path,
        use_ocr: bool = False,
        save_pages: bool = False,
        page_dpi: int = 200
) -> Path:  # <--- 修改：增加返回值
    """
    转换 PDF 的核心功能函数。现在返回生成的 Markdown 文件路径。
    """
    if not pdf_path.is_file():
        print(f"❌ 错误：找不到输入文件 '{pdf_path}'")
        return None

    # 准备路径和配置
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    absolute_model_path = model_dir.resolve()
    print(f"📦 模型将使用此路径: {absolute_model_path}")
    print(f"📂 输出内容将保存至: {output_dir.resolve()}")

    pipeline_options = PdfPipelineOptions(artifacts_path=str(absolute_model_path))
    pipeline_options.do_ocr = use_ocr
    if use_ocr:
        print("⚙️ 模式: 启用强制全页 OCR (EasyOCR)")
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options
    else:
        print("⚙️ 模式: 禁用 OCR")

    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = save_pages
    pipeline_options.images_scale = page_dpi / 72.0
    print(f"⚙️ 图片渲染 DPI 设置为: {page_dpi} (scale: {pipeline_options.images_scale:.2f})")

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    print(f"\n🚀 开始转换文件: {pdf_path.name}")

    try:
        doc = doc_converter.convert(str(pdf_path)).document
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        return None

    if save_pages:
        print(f"📸 正在保存分页图片...")
        page_output_dir = output_dir / "page"
        page_output_dir.mkdir(exist_ok=True)
        count = 0
        for page_no, page in doc.pages.items():
            if page.image and hasattr(page.image, 'pil_image'):
                page_filename = f"page_{page_no}.png"
                page_save_path = page_output_dir / page_filename
                page.image.pil_image.save(page_save_path, format="PNG")
                count += 1
        print(f"✅ 成功保存 {count} 张分页图片。")

    markdown_parts = []
    image_output_dir = output_dir / "image"
    image_output_dir.mkdir(exist_ok=True)
    print(f"✍️ 正在手动构建 Markdown 内容...")
    for element, level in doc.iterate_items():
        part_md = ""
        if isinstance(element, PictureItem):
            page_no = -1
            if element.prov and len(element.prov) > 0:
                page_no = element.prov[0].page_no
            image = element.get_image(doc)
            image_hash = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = f"page_{page_no}_{image_hash[:16]}.png"
            image_save_path = image_output_dir / image_filename
            image.save(image_save_path, format="PNG")
            part_md = f"![Image from page {page_no}](image/{image_filename})"
        elif isinstance(element, SectionHeaderItem):
            text = element.text.strip()
            hashes = '#' * (level + 2)
            part_md = f"{hashes} {text}"
        elif isinstance(element, TableItem):
            if hasattr(element, 'export_to_markdown'):
                part_md = element.export_to_markdown(doc=doc)
        elif hasattr(element, 'text'):
            part_md = element.text
        if part_md.strip():
            markdown_parts.append(part_md)
    final_markdown = "\n\n".join(markdown_parts)
    print(f"✅ 成功处理 {len(markdown_parts)} 个内容块。")

    md_output_path = output_dir / f"{pdf_path.stem}.md"
    md_output_path.write_text(final_markdown, encoding='utf-8')

    print(f"\n✅ 转换全部完成!")
    print(f"📄 Markdown 及相关图片已保存至: {output_dir.resolve()}")

    return md_output_path  # <--- 修改：返回最终的md文件路径


# --- 主处理函数，接收外部传入的输出目录 ---
def process_pdf(pdf_path: Path, output_dir: Path) -> Path:  # <--- 修改1：接收第二个参数 output_dir
    """
    处理单个 PDF 文件的主入口函数。
    它会读取配置、计算路径并调用核心转换函数。
    现在会返回生成的 Markdown 文件的路径。
    """
    print(f"-> Starting PDF processing for: {pdf_path.name}")

    config_path = Path("config.yaml")
    should_enable_ocr = False
    should_save_pages = False
    page_resolution_dpi = 200
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        pdf_processing_config = config.get('PDF processing', {})
        should_enable_ocr = pdf_processing_config.get('OCR', should_enable_ocr)
        should_save_pages = pdf_processing_config.get('Page save', should_save_pages)
        page_resolution_dpi = pdf_processing_config.get('Page dpi', page_resolution_dpi)
    except FileNotFoundError:
        print(f"ℹ️  配置文件 '{config_path}' 未找到。将使用默认设置。")
    except yaml.YAMLError as e:
        print(f"⚠️  配置文件 '{config_path}' 格式错误: {e}。将使用默认设置。")

    model_folder = "model"
    model_path = Path(model_folder)

    # --- 修改2：不再自己计算输出目录，而是直接使用传入的参数 ---
    unique_output_dir = output_dir

    md_file_path = convert_pdf_to_markdown_with_images(
        pdf_path=pdf_path,
        model_dir=model_path,
        output_dir=unique_output_dir,
        use_ocr=should_enable_ocr,
        save_pages=should_save_pages,
        page_dpi=page_resolution_dpi
    )

    # --- 修改3：返回核心函数生成的 Markdown 文件路径 ---
    return md_file_path


# --- 程序主入口 (简化为测试桩) ---
if __name__ == "__main__":
    print("--- Running pdf2md.py in standalone test mode ---")
    test_pdf_file = "TI-UC3611M.pdf"  # 在这里硬编码一个用于测试的 PDF 文件

    # 为了能独立运行，我们需要在这里手动创建一个输出目录
    test_output_dir = Path("rag_storage") / hashlib.md5(test_pdf_file.encode('utf-8')).hexdigest()

    # 直接调用本脚本的主处理函数
    process_pdf(Path(test_pdf_file), test_output_dir)