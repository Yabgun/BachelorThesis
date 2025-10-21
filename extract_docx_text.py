import sys
import os
import zipfile
import xml.etree.ElementTree as ET

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
}


def extract_paragraph_text(document_xml_bytes):
    try:
        root = ET.fromstring(document_xml_bytes)
    except ET.ParseError as e:
        # Attempt to fix common XML issues
        text = document_xml_bytes.decode('utf-8', errors='ignore')
        # Remove invalid characters and try again
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')
        root = ET.fromstring(text)

    paragraphs = []
    for p in root.findall('.//w:p', NS):
        texts = []
        for t in p.findall('.//w:t', NS):
            if t.text:
                texts.append(t.text)
        # Handle line breaks inside runs
        # Add a newline if paragraph has explicit breaks
        has_br = p.find('.//w:br', NS) is not None
        p_text = ''.join(texts).strip()
        if p_text:
            paragraphs.append(p_text)
        elif has_br:
            paragraphs.append('')
    return '\n'.join(paragraphs)


def extract_docx_text(docx_path):
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(f"File not found: {docx_path}")

    with zipfile.ZipFile(docx_path, 'r') as z:
        # Read main document
        document_xml = z.read('word/document.xml')
        text_main = extract_paragraph_text(document_xml)

        # Optionally include headers/footers if present
        extra_parts = []
        for name in z.namelist():
            if name.startswith('word/header') and name.endswith('.xml'):
                try:
                    extra_parts.append(extract_paragraph_text(z.read(name)))
                except Exception:
                    pass
            if name.startswith('word/footer') and name.endswith('.xml'):
                try:
                    extra_parts.append(extract_paragraph_text(z.read(name)))
                except Exception:
                    pass
        extra_text = '\n'.join([p for p in extra_parts if p.strip()])

    # Combine with clear separators
    combined = text_main
    if extra_text:
        combined = combined + '\n\n[Headers/Footers]\n' + extra_text
    return combined


def main():
    if len(sys.argv) < 2:
        print('Usage: python extract_docx_text.py <path_to_docx>')
        sys.exit(1)

    docx_path = sys.argv[1]
    try:
        text = extract_docx_text(docx_path)
        out_path = os.path.splitext(docx_path)[0] + '.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Extracted text written to: {out_path}')
        print('--- BEGIN TEXT PREVIEW (first 4000 chars) ---')
        print(text[:4000])
        print('--- END TEXT PREVIEW ---')
    except Exception as e:
        print('ERROR extracting DOCX:', e)
        sys.exit(2)


if __name__ == '__main__':
    main()