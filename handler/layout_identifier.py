from pypdf import PdfReader
import ocrmypdf
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

class PDFToMarkdownConverter:
    def __init__(self, pdf_file):
        """
        Initialize the converter with a PDF file and MongoDB connection details.
        """
        self.pdf_file = pdf_file
        self.reader = PdfReader(pdf_file)
        self.metadata = self._extract_metadata()
        self.full_text = ""
        self.image_count = 0

    def _extract_metadata(self):
        """
        Extract metadata from the PDF file.
        """
        return {
            "Author": self.reader.metadata.author,
            "Creator": self.reader.metadata.creator,
            "Producer": self.reader.metadata.producer,
            "Subject": self.reader.metadata.subject,
            "Title": self.reader.metadata.title,
        }

    def _extract_text_from_pdf(self):
        """
        Extract text from the PDF file.
        """
        full_text = ""
        for idx, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if len(text) > 0:
                full_text += f"### Page {idx + 1}\n\n{text}\n\n"
        return full_text.strip()

    def _count_images(self):
        """
        Count the number of images in the PDF file.
        """
        image_count = 0
        for page in self.reader.pages:
            image_count += len(page.images)
        return image_count

    def _perform_ocr(self):
        """
        Perform OCR on the PDF file if necessary and reload the reader.
        """
        out_pdf_file = self.pdf_file.replace(".pdf", "_ocr.pdf")
        ocrmypdf.ocr(self.pdf_file, out_pdf_file, force_ocr=True)
        self.reader = PdfReader(out_pdf_file)
        self.full_text = self._extract_text_from_pdf()

    def convert(self):
        """
        Convert the PDF file to Markdown, extracting text and metadata, and store it in MongoDB.
        """
        self.full_text = self._extract_text_from_pdf()
        self.image_count = self._count_images()
        if self.image_count > 0 and len(self.full_text) < 1000:
            self._perform_ocr()
        markdown_content = "# Extracted Text\n\n" + self.full_text + "\n\n"
        markdown_content += "# Metadata\n\n"
        for key, value in self.metadata.items():
            markdown_content += f"- **{key}**: {value if value else 'N/A'}\n"
        document_id = os.path.splitext(os.path.basename(self.pdf_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{document_id}_{timestamp}"
        return unique_id,markdown_content
