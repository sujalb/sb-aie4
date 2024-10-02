import PyPDF2

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_pdf(self):
        with open(self.file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def load_and_split(self, chunk_size=1000, overlap=200):
        full_text = self.load_pdf()
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks