import os
import glob
import pandas as pd
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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

print(f"Starting CloudNexus ChatBot (Production Version)...")

# 1. Load Data (Hybrid Approach)
documents = []

if not os.path.exists(DATA_PATH):
    print(f"Error: The folder '{DATA_PATH}' does not exist.")
    exit()

# A. Load Markdown/PDF (Company Profile)
# We handle both .md and .pdf for flexibility
files = glob.glob(os.path.join(DATA_PATH, "*.*"))
for file_path in files:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.md', '.txt', '.pdf']:
        print(f"Loading Company Doc: {os.path.basename(file_path)}")
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = 'policy_doc' # Tag as Policy
                doc.metadata['priority'] = 'low'
                doc.metadata['filename'] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# B. Load CSV (Employee Data)
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
for csv_file in csv_files:
    print(f"Loading CSV: {os.path.basename(csv_file)}")
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
        # Clean columns and data
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        df = df.fillna("Unassigned")

        print(f"   -> Columns cleaned: {list(df.columns)}")

        for _, row in df.iterrows():
            name = str(row.get('employee_name', 'Unknown')).strip()
            if name.lower() in ['nan', 'unknown', '', 'unassigned']: continue

            role = str(row.get('designation', 'Unknown')).strip()
            dept = str(row.get('department', 'Unknown')).strip()
            team = str(row.get('team_name', 'Unassigned')).strip()
            lead = str(row.get('team_lead', 'Unassigned')).strip()

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
# CRITICAL FIX: Fetch top 60 documents to avoid 'crowding out' by employees
retriever = db.as_retriever(search_kwargs={"k": 60}) 
print("Index Ready!")

# 4. Create Chain
llm = OllamaLLM(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template("""
You are the internal AI assistant for CloudNexus.
Use the provided context to answer the user's questions.

CRITICAL RULES:
1. **Source of Truth**: 
   - '[SOURCE: EMPLOYEE_DB]' is the truth for employee details.
   - Markdown/PDF is the truth for Company Info (CEO, Directors, Links).
2. **Conflict**: If sources conflict, trust the Employee DB for people, and Markdown for policy.
3. **Completeness**: Use all provided context. If the answer is in the text, say it.
4. **Format**: Answer concisely.

<context>
{context}
</context>

Question: {input}
""")

def format_docs(docs):
    unique_docs = {}
    for doc in docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
    return "\n\n".join(doc.page_content for doc in unique_docs.values())

def get_response(x):
    query_text = x["input"]
    all_docs = []

    # 1. Retrieve
    if query_text.count('?') > 1:
        print("   [Log] Compound Query Detected. Splitting retrieval...")
        sub_queries = [q.strip() for q in query_text.split('?') if q.strip()]
        for sub_q in sub_queries:
            docs = retriever.invoke(sub_q)
            all_docs.extend(docs[:15]) # Fetch top 15 for each sub-query
    else:
        all_docs = retriever.invoke(query_text)
    
    # 2. Advanced Re-Ranking
    query_terms = query_text.lower().split()
    scored_docs = []
    
    for doc in all_docs:
        content = doc.page_content.lower()
        score = 0
        
        # A. Boost Company Info (Docs) if query asks for non-employee stuff
        is_employee_record = "" in content
        if not is_employee_record:
            # If doc is NOT an employee record, give it a base survival score
            score += 2
            # Huge boost if query asks for leadership/company info
            if any(k in query_text.lower() for k in ['ceo', 'cto', 'director', 'policy', 'link', 'url', 'service', 'focus']):
                score += 10

        # B. Boost Employee Records ONLY if specific name matches
        if is_employee_record:
            # Extract the actual name from the record "Name: Aryan"
            try:
                # Find the line starting with "Name:"
                name_line = [line for line in content.split('\n') if 'name:' in line][0]
                record_name = name_line.split(':')[1].strip().lower()
                
                # Check if this specific name is in the user's query
                if record_name in query_text.lower():
                     score += 15 # Massive boost for exact person match
            except:
                pass

        # C. General Keyword Match
        score += sum(1 for term in query_terms if term in content)
        scored_docs.append((doc, score))
    
    # Sort and take Top 15 (Context window is large enough for this)
    scored_docs.sort(key=lambda item: item[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:15]]
    
    context_str = format_docs(top_docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context_str, "input": query_text})
    
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