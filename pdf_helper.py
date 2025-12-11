import os
import uuid
from typing import List
from dataclasses import dataclass
import pypdf


@dataclass
class ExtractedParagraph:
    """Represents an extracted paragraph from a PDF."""
    id: str
    location: str
    title: str
    content: str


@dataclass
class ExtractionResult:
    """Represents the result of PDF extraction."""
    file_name: str
    company: str
    year: str
    extracted_paragraphs: List[ExtractedParagraph]


class PDFHelper:
    """Helper class to extract paragraphs from PDF files."""

    def __init__(self):
        pass

    def extract_paragraphs(self, file_path: str) -> ExtractionResult:
        """
        Extract paragraphs from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing extracted paragraphs and metadata
        """
        # Get the Company Name and Year from the file name
        filename = os.path.basename(file_path)
        parts = filename.replace('.pdf', '').split('-')
        
        company_name = ""
        year = ""
        
        if len(parts) >= 2:
            company_name = parts[0]
            year = parts[1]

        # Extract the paragraphs of text from the document
        extracted_paragraphs = []
        pending_content = ""

        with open(file_path, 'rb') as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_index in range(num_pages):
                page = pdf_reader.pages[page_index]
                page_text = page.extract_text()

                # Split by lines and combine into blocks
                lines = page_text.split('\n')
                blocks = [line.strip() for line in lines if line.strip()]

                paragraph_index = 0
                for block in blocks:
                    paragraph_content = block

                    # Check if pending content should be appended
                    if pending_content:
                        paragraph_content = pending_content + " " + paragraph_content
                        pending_content = ""

                    # Count the number of words in the paragraph
                    word_count = len(paragraph_content.split())

                    # If the paragraph has less than 250 words, store it in pending_content
                    if word_count < 250:
                        pending_content = paragraph_content
                    else:
                        # Check if the paragraph_content exceeds 7000 characters
                        while len(paragraph_content) > 7000:
                            # Split the paragraph_content into smaller paragraphs
                            sub_paragraph = paragraph_content[:7000]
                            last_space_index = sub_paragraph.rfind(' ')
                            sub_paragraph = sub_paragraph[:last_space_index]

                            # Add the smaller paragraph to the extracted_paragraphs list
                            extracted_paragraphs.append(ExtractedParagraph(
                                id=str(uuid.uuid4()),
                                location=f"{page_index + 1}-{paragraph_index + 1}",
                                title=f"Page {page_index + 1} - Paragraph {paragraph_index + 1}",
                                content=sub_paragraph
                            ))
                            paragraph_index += 1

                            # Update paragraph_content
                            paragraph_content = paragraph_content[last_space_index + 1:]

                        # Add the remaining paragraph_content to the extracted_paragraphs list
                        extracted_paragraphs.append(ExtractedParagraph(
                            id=str(uuid.uuid4()),
                            location=f"{page_index + 1}-{paragraph_index + 1}",
                            title=f"Page {page_index + 1} - Paragraph {paragraph_index + 1}",
                            content=paragraph_content
                        ))
                        paragraph_index += 1

            # Add any remaining pending content as the last paragraph
            if pending_content:
                extracted_paragraphs.append(ExtractedParagraph(
                    id=str(uuid.uuid4()),
                    location=f"{num_pages}-{len(extracted_paragraphs) + 1}",
                    title=f"Page {num_pages} - Paragraph {len(extracted_paragraphs) + 1}",
                    content=pending_content
                ))

        extraction_result = ExtractionResult(
            file_name=filename,
            company=company_name,
            year=year,
            extracted_paragraphs=extracted_paragraphs
        )

        print(extraction_result)
        return extraction_result
