import os
import glob
import pandas as pd
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Configuration
MODEL_NAME = "llama3.2:3b"
DATA_PATH = "./data"
DB_FAISS_PATH = "vectorstore/db_faiss"

print(f"Starting CloudNexus ChatBot (Fixed Version)...")

# 1. Load Data (Hybrid Approach)
documents = []

if not os.path.exists(DATA_PATH):
    print(f"Error: The folder '{DATA_PATH}' does not exist.")
    exit()

# A. Load PDFs (Company Policy/Info)
pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
for pdf_file in pdf_files:
    print(f"Loading PDF: {os.path.basename(pdf_file)}")
    try:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = 'policy_doc'
            doc.metadata['priority'] = 'low'
            doc.metadata['filename'] = os.path.basename(pdf_file)
        documents.extend(docs)
    except Exception as e:
        print(f"Error loading PDF {pdf_file}: {e}")

# B. Load CSV (Employee Data) - Robust Loading
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
for csv_file in csv_files:
    print(f"Loading CSV: {os.path.basename(csv_file)}")
    try:
        # 'skipinitialspace' fixes spacing issues in CSVs
        df = pd.read_csv(csv_file, skipinitialspace=True)
        
        # CRITICAL FIX: Clean column names (remove hidden spaces)
        df.columns = df.columns.str.strip()
        
        # Debug: Print columns to verify correct loading
        print(f"   -> Columns found: {list(df.columns)}")

        for _, row in df.iterrows():
            # Robust .get() calls with stripped keys
            name = str(row.get('employee_name', 'Unknown')).strip()
            role = str(row.get('designation', 'Unknown')).strip()
            dept = str(row.get('department', 'Unknown')).strip()
            team = str(row.get('team_name', 'Unknown')).strip()
            lead = str(row.get('team_lead', 'Unknown')).strip()
            
            # Skip empty rows if any
            if name.lower() in ['nan', 'unknown', '']: continue

            content = (
                f"[SOURCE: EMPLOYEE_DB] Employee Record:\n"
                f"Name: {name}\n"
                f"Role: {role}\n"
                f"Department: {dept}\n"
                f"Team: {team} (Led by: {lead})"
            )
            
            doc = Document(
                page_content=content, 
                metadata={"source": "employee_db", "priority": "high", "filename": os.path.basename(csv_file)}
            )
            documents.append(doc)
            
    except Exception as e:
        print(f"Failed to load CSV {csv_file}: {e}")

if not documents:
    print("No documents found!")
    exit()

print(f"Total documents processed: {len(documents)}")

# 2. Split Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 3. Create Vector Store
print("Building Index...")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)
print("Index Saved!")

# INCREASED k to 15 to ensure we fetch enough candidates from both PDF and CSV
retriever = db.as_retriever(search_kwargs={"k": 15})

# 4. Create Chain
llm = OllamaLLM(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template("""
You are the internal AI assistant for CloudNexus.
Use the provided context to answer the user's question.

CRITICAL RULES:
1. **Source of Truth**: 
   - Information marked '[SOURCE: EMPLOYEE_DB]' is the absolute truth for employee details.
   - Information from PDFs (Company Profile) is the truth for Directors, CEO, and Company Policies.
2. **Conflict**: If PDF and CSV conflict regarding an *employee*, trust the CSV.
3. **Completeness**: Check ALL context chunks. If the answer is in the PDF (like Directors), use it.
4. **Directness**: Answer concisely.

<context>
{context}
</context>

Question: {input}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response(x):
    # A. Fetch docs
    all_docs = retriever.invoke(x["input"])
    
    # B. Keyword Re-Ranking (FIXED: Removed the massive +5 bias)
    query_terms = x["input"].lower().split()
    scored_docs = []
    
    for doc in all_docs:
        content = doc.page_content.lower()
        score = 0
        
        # 1. Exact Name Match Bonus (High precision)
        # If a query term matches a name exactly in the DB, boost it.
        # This helps "Anup" float to the top if mentioned.
        if "" in content:
            for term in query_terms:
                if term in content: 
                    score += 3  # Boost relevant employee records
        
        # 2. General Context Match
        score += sum(1 for term in query_terms if term in content)
        
        scored_docs.append((doc, score))
    
    # Sort and take Top 8 (Increased context window slightly)
    scored_docs.sort(key=lambda item: item[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:8]]
    
    context_str = format_docs(top_docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context_str, "input": x["input"]})
    
    return {"answer": answer, "context": top_docs}

rag_chain = RunnableLambda(get_response)

# 5. Run Loop
print("CloudNexus Bot Ready! (Type 'exit' to quit)")
while True:
    query = input("\nYou: ")
    if query.lower() == 'exit':
        break
    
    response = rag_chain.invoke({"input": query})
    print(f"Bot: {response['answer']}")
    
    print("\n[Sources Used]:")
    seen_sources = set()
    for doc in response['context']:
        # Show filename AND source type to be sure
        src = f"{doc.metadata.get('filename')} ({doc.metadata.get('source', 'Unknown')})"
        if src not in seen_sources:
            print(f" - {src}")
            seen_sources.add(src)