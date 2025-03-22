from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask_cors import CORS
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)
# Directory containing PDFs
PDF_DIRECTORY = "pdfs"

# Function to extract text from PDFs
def get_pdf_text(pdf_directory):
    text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_reader = PdfReader(os.path.join(pdf_directory, filename))
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a specialized banking and financial services assistant. Your primary role is to provide accurate, clear, and helpful information about banking principles, financial services, and forex trading.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    1. Respond with precise, factual information directly addressing the question.
    2. Use simple, accessible language while maintaining technical accuracy.
    3. If the exact answer isn't in the context, respond with "The information requested is not available in the provided context."
    4. For numerical data, present information in a structured format when appropriate.
    5. If the question is ambiguous, briefly note the ambiguity before answering the most likely interpretation.
    6. Do not speculate or provide information beyond what's in the context.
    7. If the question requests financial advice that would typically require personalized assessment, note the limitations of general information.

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Preload and process PDFs
print("Processing PDFs at startup...")
raw_text = get_pdf_text(PDF_DIRECTORY)
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)
print("PDF processing complete.")

# Endpoint to handle user questions
@app.route("/ask_question", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        user_question = data.get("question", "")
        
        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return jsonify({"response": response["output_text"]}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
