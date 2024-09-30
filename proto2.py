from langchain_ollama import OllamaLLM  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
import pdfplumber  # Import pdfplumber as a fallback option
import pandas as pd
from pdf2image import convert_from_path
import numpy as np
import os
import easyocr  # Import EasyOCR for OCR

# Define a template for extracting tabular data
table_template = """
You are an AI assistant tasked with extracting and imaginary organizing information from the provided document. Use the following format to structure your response based on the content:

1. **Account Name**: (Name of the insured person or entity)
2. **Agency Name**: 
    - List of agencies involved (if multiple, specify with parentheses)
3. **Branch**: (Similar to Agency)
4. **Address**:
    - Street:
    - City:
    - State:
    - Zip Code:
5. **Primary Contact**:
    - First Name:
    - Middle Name:
    - Phone Number:
    - E-Mail:
6. **Servicing**:
    - Client Service Rep:
    - Underwriter:
    - Claims Adjustor:
7. **Billing**:
    - Broker:

Instructions:
- If there are multiple instances of a category (such as multiple agencies or contacts), specify in parentheses which entity or person it belongs to.
- If some information is missing, note it as "Not available."
- Provide the output in the specified format.
- The data presented is imaginary

Document Text:
{context}

Answer:
"""


model = OllamaLLM(model="llama3.2")
table_prompt = ChatPromptTemplate.from_template(table_template)
table_chain = table_prompt | model

# Define your Poppler path
poppler_path = r"C:\Users\mizan.chishty\Desktop\LLM\poppler-24.07.0\Library\bin"  # Adjust this path if necessary

# Function to retrieve relevant information (for demonstration purposes)
def retrieve_information(query):
    # Simulated retrieval logic (you can replace this with an actual knowledge base query)
    retrieved_info = f"Retrieved information based on query: {query}"
    return retrieved_info

# Primary function to read PDF pages by converting to images and applying OCR
def read_pdf(file_path):
    text = ""

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the language(s) you need

    # Function to extract text using EasyOCR from an image
    def extract_text_from_image(image):
        image_np = np.array(image)  # Convert PIL image to numpy array for EasyOCR
        results = reader.readtext(image_np, detail=0)  # detail=0 returns only the text
        return " ".join(results)

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        return text

    # Convert PDF pages to images and extract text using OCR with threading
    try:
        # Pass the poppler_path to convert_from_path
        images = convert_from_path(file_path, poppler_path=poppler_path)

        if not images:
            print(f"No images extracted from {file_path}.")
            return text

        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor() as executor:
            ocr_texts = list(executor.map(extract_text_from_image, images))

        text = "\n".join(ocr_texts)

        if text.strip():
            print(f"Successfully extracted content using EasyOCR from: {file_path}")
            return text
        else:
            print(f"EasyOCR could not extract content from: {file_path}; falling back to pdfplumber.")
    except Exception as e:
        print(f"Error converting {file_path} to images or extracting text with EasyOCR: {e}")
    
    # Fallback to pdfplumber or PyPDF2 if OCR fails (as in your original logic)
    try:
        with open(file_path, 'rb') as file:
            with pdfplumber.open(file) as pdf:
                with ThreadPoolExecutor() as executor:
                    page_texts = executor.map(lambda page: page.extract_text() or "", pdf.pages)
                text = "\n".join([pt for pt in page_texts if pt])
                
                if text.strip():
                    print(f"Successfully read content from: {file_path} with pdfplumber")
                    return text
                else:
                    print(f"pdfplumber could not extract content; falling back to PyPDF2.")
    except Exception as e:
        print(f"Error reading {file_path} with pdfplumber: {e}")

    try:
        with open(file_path, 'rb') as file:
            reader_pypdf = PyPDF2.PdfReader(file)
            with ThreadPoolExecutor() as executor:
                for page in reader_pypdf.pages:
                    text += page.extract_text() or ""
                    
        if text.strip():
            print(f"Successfully read content with PyPDF2 from: {file_path}")
        else:
            print(f"PyPDF2 could not extract content from: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path} with PyPDF2: {e}")

    return text

# Function to extract tables and combine them into a structured format
def extract_tables_from_pdf_folder(pdf_folder):
    all_tables = []

    # Loop through all files in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):  # Only process PDF files
            pdf_path = os.path.join(pdf_folder, file_name)
            pdf_text = read_pdf(pdf_path)
            with open("context.text", "w") as f:
                f.write(pdf_text)

            
            if pdf_text.strip():
                # Retrieve additional information based on the PDF text context
                retrieved_info = retrieve_information(pdf_text)

                # Combine the extracted PDF text with the retrieved information for RAG
                result = table_chain.invoke({"context": pdf_text, "retrieved_context": retrieved_info})
                
                # Assuming result is a structured response from the model
                structured_data = result.strip()

                # Append to the list of all tables
                all_tables.append(structured_data)
            else:
                print(f"No content was read from: {pdf_path}")

    # Combine all structured data into a single string, with each result separated by a newline
    combined_csv_data = "\n\n".join(all_tables)

    # Save combined data to a CSV file
    with open("combined_data.csv", "w") as f:
        f.write(combined_csv_data)
    print("Temporary data made")

if __name__ == "__main__":
    # Specify the folder containing the PDFs
    pdf_folder = r"c:\Users\mizan.chishty\Downloads\Docs\Docs\999"
    extract_tables_from_pdf_folder(pdf_folder)
