import pandas as pd
import PyPDF2
import re

# Function to convert PDF to CSV
def pdf_to_csv(pdf_file, csv_file):
    # Open the PDF file
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        
        # Extract text from each page
        for page in reader.pages:
            text += page.extract_text() + '\n'

    # Process the text to create a list of rows (assuming a simple line-based format)
    # Here, we split by line breaks and can further split by commas or spaces if needed
    rows = text.strip().split('\n')

    # If your data is separated by spaces or tabs, you might need to modify the splitting logic
    data = [re.split(r'\s{2,}', row) for row in rows]  # Splits by two or more spaces

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to CSV
    df.to_csv(csv_file, index=False, header=False)  # Change header to True if you have header information
    print(f"Successfully converted '{pdf_file}' to '{csv_file}'.")

# Example usage
pdf_file = r"C:\Users\mizan.chishty\Downloads\005_ShubhamSuthar-2392024.pdf"  # Path to your PDF file
csv_file = r"C:\Users\mizan.chishty\Desktop\LLM\new.csv"  # Desired output CSV file name
pdf_to_csv(pdf_file, csv_file)
