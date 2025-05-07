import fitz  # PyMuPDF
from collections import Counter

def extract_line_positions(page, num_lines=2):
    """Extract top and bottom N lines with their Y coordinates."""
    text_lines = page.get_text("dict")["blocks"]
    lines = []

    for block in text_lines:
        for line in block.get("lines", []):
            line_text = " ".join([span["text"] for span in line["spans"]])
            y0 = line["bbox"][1]
            y1 = line["bbox"][3]
            lines.append((line_text.strip(), y0, y1))

    lines_sorted = sorted(lines, key=lambda x: x[1])
    top_lines = lines_sorted[:num_lines]
    bottom_lines = lines_sorted[-num_lines:] if len(lines_sorted) >= num_lines else []
    return top_lines, bottom_lines

def detect_common_lines(pages_data, part='top'):
    """Detect repeated top/bottom lines."""
    line_texts = []
    for top, bottom in pages_data:
        lines = top if part == 'top' else bottom
        for line in lines:
            line_texts.append(line[0])
    counter = Counter(line_texts)
    common = [text for text, count in counter.items() if count > 1 and text]
    return common

def detect_common_image_positions(doc):
    """Detect repeated image positions (likely watermarks/logos)."""
    image_positions = []
    for page in doc:
        for img in page.get_images(full=True):
            bbox = page.get_image_bbox(img)
            image_positions.append((round(bbox.x0), round(bbox.y0), round(bbox.x1), round(bbox.y1)))
    counter = Counter(image_positions)
    return [pos for pos, count in counter.items() if count > 1]

def crop_fixed_header_footer_and_remove_images(input_path, output_path, num_lines=2):
    doc = fitz.open(input_path)
    pages_data = []

    for page in doc:
        top, bottom = extract_line_positions(page, num_lines)
        pages_data.append((top, bottom))

    # Detect common headers and footers
    common_headers = detect_common_lines(pages_data, part='top')
    common_footers = detect_common_lines(pages_data, part='bottom')
    common_images = detect_common_image_positions(doc)

    # Determine header and footer Y-bounds from first matching page
    header_y1 = None
    footer_y0 = None

    for top, bottom in pages_data:
        # **Header Logic**: Crop till the bottom-most matching header line
        for line_text, y0, y1 in top:
            if line_text in common_headers:
                if header_y1 is None or y1 > header_y1:
                    header_y1 = y1  # Update header_y1 to the bottom of the last matched header line

        # **Footer Logic**: Crop from the first matching footer line
        matching_footer_lines = [y0 for line_text, y0, y1 in bottom if line_text in common_footers or "page" in line_text.lower()]
        if matching_footer_lines:
            footer_y0 = min(matching_footer_lines)  # top of the first matching footer line

        if header_y1 is not None and footer_y0 is not None:
            break

    if header_y1 is None:
        header_y1 = 0  # No header detected
    if footer_y0 is None:
        footer_y0 = doc[0].rect.y1  # No footer detected

    # Apply uniform cropping and image removal
    for page in doc:
        for img in page.get_images(full=True):
            bbox = page.get_image_bbox(img)
            pos = (round(bbox.x0), round(bbox.y0), round(bbox.x1), round(bbox.y1))
            if pos in common_images:
                page.add_redact_annot(bbox, fill=(1, 1, 1))
        page.apply_redactions()
        rect = page.rect
        cropped_rect = fitz.Rect(rect.x0, header_y1, rect.x1, footer_y0)
        page.set_cropbox(cropped_rect)
    doc.save(output_path)
    doc.close()
    print(f"Cropped and cleaned PDF saved to: {output_path}")

#input_pdf = r"C:/Users/DELL/Desktop/Chatbot/Input_docs/machineLearning.pdf"
#output_pdf = r"C:\Users\DELL\Desktop\cleaned_output_srs.pdf"
#crop_fixed_header_footer_and_remove_images(input_pdf, output_pdf)

