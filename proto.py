from langchain_ollama import OllamaLLM  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
import pdfplumber  # Import pdfplumber as a fallback option

template = """
You are an AI assistant designed to help users work with large documents. Your job is to prioritize and extract information that is most relevant to the user’s question.

Context:
{context}

Question:
{question}

Instructions:
- Focus on the sections of the context that are most relevant to the question.
- If the question is broad, summarize key points first before addressing specifics.
- Highlight any assumptions made if the exact answer isn’t in the context.

Answer:
"""



model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to read PDF pages concurrently with fallback to pdfplumber
def read_pdf(file_path):
    text = ""

    def extract_page_text(page):
        try:
            page_text = page.extract_text()
            return page_text.strip() if page_text else ""
        except Exception as e:
            print(f"Error extracting page: {e}")
            return ""

    try:
        # Attempt to read with pdfplumber
        with open(file_path, 'rb') as file:
            with pdfplumber.open(file) as pdf:
                with ThreadPoolExecutor() as executor:
                    page_texts = executor.map(extract_page_text, pdf.pages)
            text = "\n".join([pt for pt in page_texts if pt])
            if text.strip():  # If PyPDF2 extraction is successful
                print(f"Successfully read content from: {file_path} with pdfplumber")
                return text
            else:
                print(f"pdfplumber could not extract content; trying pypdf2.")
    except Exception as e:
        print(f"Error reading {file_path} with pdfplumber: {e}")

    # Fallback to pdfplumber if PyPDF2 fails
    try:
        reader=PyPDF2.PdfReader(file)
        with ThreadPoolExecutor() as executor:
            for page in reader:
                text += page.extract_text() or ""
        print(f"Successfully read content with pypdf2 from: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path} with pypdf2: {e}")

    return text

# Function to update context based on user interactions and PDFs
def update_context(current_context, new_text, max_length=5000):
    updated_context = current_context + "\n" + new_text
    return updated_context[-max_length:]  # Keeps the last 'max_length' characters

# Function to handle conversation with fixed context management
def handle_conversation(pdf_files):
    context = ""
    for pdf_path in pdf_files:
        pdf_text = read_pdf(pdf_path)
        context = update_context(context, pdf_text)  # Update context with newly read PDF text

    if not context.strip():
        print("Warning: No content was read from the provided PDF files.")
    else:
        print("PDF content successfully loaded.")

    print("Welcome to AI Chatbot. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Use only the initial context loaded from the PDF for each question
        result = chain.invoke({"context": context, "question": user_input})

        print("Bot:", result)


if __name__ == "__main__":
    pdf_files = [r"C:\Users\mizan.chishty\Downloads\lekl101.pdf"]  # Add your PDF files here
    handle_conversation(pdf_files)
