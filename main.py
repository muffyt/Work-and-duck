from langchain_ollama import OllamaLLM  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
import PyPDF2
 
 
template = """
Answer the question below.
 
Here is the conversation history: {context}
 
Question: {question}
 
Aswer:
"""
 
 
model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
 
# Function to read the PDF and extract text
def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        print(f"Successfully read content from: {file_path}")  # Confirmation message        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text
 
def handle_conversation(user_input, context):
    result = chain.invoke({"context": context, "question": user_input})
    return result

if __name__ == "__main__":
     pdf_files = [r"C:\Users\mizan.chishty\Downloads\16-17-Business Insurance-Policy-Invoice $335.00.pdf"]  # Add your PDF files here
     handle_conversation(pdf_files)