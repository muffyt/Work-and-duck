from flask import Flask, render_template, request, jsonify
from main import handle_conversation, read_pdf  # Import functions from your script

app = Flask(__name__)

# Global variable to store the context
context = ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global context
    user_input = request.form['question']
    if user_input.lower() == "exit":
        return jsonify({'response': 'Goodbye!'})
    
    # Process user input and generate a response
    result = handle_conversation(user_input, context)
    context += f"\nUser: {user_input}\nAI: {result}"  # Update context
    return jsonify({'response': result})

if __name__ == '__main__':
    pdf_files = [r"C:\Users\mizan.chishty\Downloads\16-17-Business Insurance-Policy-Invoice $335.00.pdf"]
    context = read_pdf(pdf_files[0])  # Read the PDF once at startup
    app.run(debug=True)
