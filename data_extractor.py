import pandas as pd
from langchain_ollama import OllamaLLM  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore

# Define a template for extracting business-related information
table_template = """
You are an AI assistant tasked with extracting and organizing **business-related information** from the provided document. The information relates to companies, agencies, and services. Avoid focusing on personal, sensitive, or confidential data unrelated to business entities.

Use the following format to structure your response based on the content:

1. **Account Name**: (Name of the insured business or entity)
2. **Agency Name**: 
    - List of business agencies involved (if multiple, specify with parentheses)
3. **Branch**: (Similar to Agency)
4. **Address**:
    - Street:
    - City:
    - State:
    - Zip Code:
5. **Primary Contact** (Business contact):
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
- If there are multiple instances of a category (such as multiple agencies or contacts), specify in parentheses which business entity or person it belongs to.
- If some information is missing, note it as "Not available."
- Focus strictly on business-related information and do not include any personal, confidential, or sensitive data.
- Provide the output in the specified format, ensuring clarity and structure.

Document Text:
{context}

Answer:
"""

# Initialize the model and prompt template
model = OllamaLLM(model="llama3.2")
table_prompt = ChatPromptTemplate.from_template(table_template)
table_chain = table_prompt | model

def process_csv(input_csv, output_csv):
    """
    Process the input CSV file to extract and organize business-related information,
    then save the structured results to an output CSV file.
    
    :param input_csv: Path to the input CSV file
    :param output_csv: Path to save the output CSV file
    """
    
    try:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(input_csv, delimiter=';')  # Adjust the delimiter as needed
        print(df.head())  # Print the first few rows of the DataFrame

        # Prepare a list to store the results
        results = []

        # Process each row in the DataFrame
        for index, row in df.iterrows():
            # Assuming the relevant text is in a column named 'Document Text'
            document_text = row.get('Document Text', '')  # Adjust column name as necessary
            
            # Generate structured output using the model
            if document_text.strip():  # Check if the document text is not empty
                result = table_chain.invoke({"context": document_text})
                results.append(result.strip())
            else:
                print(f"No document text found for row {index}")

        # Create a new DataFrame to store the structured results
        results_df = pd.DataFrame(results, columns=['Extracted Data'])

        # Ensure the results DataFrame is not empty before proceeding
        if not results_df.empty:
            # Extract Account Names
            results_df['Account Name'] = results_df['Extracted Data'].str.extract(r'1\. \*\*Account Name\*\*: (.*)')[0]
            # Group the results by Account Name
            grouped_results = results_df.groupby('Account Name')['Extracted Data'].apply(list).reset_index()

            # Write the organized data to a new CSV file
            grouped_results.to_csv(output_csv, index=False)
            print(f"Processed data saved to '{output_csv}'.")
        else:
            print("No valid extracted data to process.")

    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")

if __name__ == "__main__":
    input_csv = r"C:\Users\mizan.chishty\Desktop\LLM\combined_data.csv"  # Specify the input CSV file
    output_csv = r"C:\Users\mizan.chishty\Desktop\LLM\new.csv"  # Specify the output CSV file
    process_csv(input_csv, output_csv)
    

