import os
import json
import pickle
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import List, Dict, Any





# Set Google Cloud Storage Bucket Name
BUCKET_NAME = "rag-model-bucket"

# Initialize Google Cloud Storage
storage_client = storage.Client()

# Split documents into smaller chunks


def split_documents(legal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []

    def process_law(law: Dict[str, Any], path: List[str]):
        # Process law metadata
        law_metadata = {
            "id": law.get("id"),
            "title": law.get("title"),
            "short_title": law.get("short_title"),
            "department": law.get("metadata", {}).get("department"),
            "issued_date": law.get("metadata", {}).get("issued_date"),
            "last_amended": law.get("metadata", {}).get("last_amended", {}).get("sfs")
        }
        
        # Process chapters
        for chapter in law.get("chapters", []):
            chapter_path = path + [f"Chapter {chapter['number']}"]
            
            # Process sections
            for section in chapter.get("sections", []):
                section_path = chapter_path + [f"Section {section['number']}"]
                
                # Create chunk for section text
                chunk = {
                    "content": section.get("text", ""),
                    "metadata": {
                        "path": " > ".join(section_path),
                        "law_info": law_metadata,
                        "chapter_title": chapter.get("title"),
                        "section_number": section.get("number"),
                        "amendments": [a.get("law") for a in section.get("amendments", [])],
                        "case_law_count": section.get("case_law", [{}])[0].get("count", 0),
                        "citations_count": section.get("citations", [{}])[0].get("count", 0)
                    }
                }
                chunks.append(chunk)
                
                # Process comments as separate chunks
                for comment in section.get("comments", []):
                    comment_chunk = {
                        "content": comment.get("text", ""),
                        "metadata": {
                            "path": " > ".join(section_path + ["Comment"]),
                            "law_info": law_metadata,
                            "comment_type": comment.get("type"),
                            "section_number": section.get("number")
                        }
                    }
                    chunks.append(comment_chunk)
                
                # Process paragraphs if present
                for i, paragraph in enumerate(section.get("paragraphs", []), 1):
                    para_chunk = {
                        "content": paragraph.get("text", ""),
                        "metadata": {
                            "path": " > ".join(section_path + [f"Paragraph {i}"]),
                            "law_info": law_metadata,
                            "section_number": section.get("number")
                        }
                    }
                    chunks.append(para_chunk)
                    
                    # Process points if present
                    for j, point in enumerate(paragraph.get("points", []), 1):
                        point_chunk = {
                            "content": point,
                            "metadata": {
                                "path": " > ".join(section_path + [f"Paragraph {i}", f"Point {j}"]),
                                "law_info": law_metadata,
                                "section_number": section.get("number")
                            }
                        }
                        chunks.append(point_chunk)

    # Start processing from the top level
    for category, laws in legal_data.get("lagen", {}).get("lagar", {}).items():
        for law_name, law_content in laws.items():
            process_law(law_content, [category, law_name])

    return chunks

# Read documents from GCS

def read_documents():
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs())
        all_documents = []
        for blob in blobs:
            if blob.name.endswith('.json'):
                content = blob.download_as_text()
                print(f"Loaded document: {blob.name}")
                legal_data = json.loads(content)
                documents = split_documents(legal_data)
                all_documents.extend(documents)
            else:
                print(f"Skipped non-JSON file: {blob.name}")
        return all_documents
    except Exception as e:
        print(f"Error reading documents from GCS: {e}")
        return []


# Set up FAISS vector store with Vertex AI embeddings
def setup_vector_store(documents):
    try:
        embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
        # Initialize FAISS Vector Store
        vector_store = FAISS(embedding_function=lambda x: embedding_model.get_embeddings(x)[0].values)

        # Add documents and embeddings to the vector store
        for idx, doc in enumerate(documents):
            vector_store.add_texts(
                texts=[doc["content"]],
                metadatas=[{**doc["metadata"], "chunk_id": f"chunk_{idx}"}]
            )
        return vector_store
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None

# Query LLM with context retrieved from FAISS
def query_llm(query, vector_store):
    try:
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)

        # Prepare context from retrieved documents
        context = "\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])

        # Prepare prompt
        prompt = f"Based on the following information:\n\n{context}\n\nAnswer the question: {query}"

        # Generate response
        llm = TextGenerationModel.from_pretrained("text-bison@latest")
        response = llm.predict(prompt, max_output_tokens=1024, temperature=0.2)

        # Prepare source information
        sources = [f"Source {i+1}: {doc.metadata.get('path', 'Unknown')}" for i, doc in enumerate(relevant_docs)]

        return {"answer": response.text, "sources": sources}
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {"answer": "Sorry, I couldn't process your query.", "sources": []}

# Main RAG pipeline
def main():
    print("Reading documents from GCS...")
    documents = read_documents()
    if not documents:
        print("No documents found. Exiting...")
        return

    print("Setting up vector store...")
    vector_store = load_vector_store() or setup_vector_store(documents)
    if not vector_store:
        print("Failed to initialize vector store. Exiting...")
        return
    
    save_vector_store(vector_store)

    print("System is ready! Enter your query (or type 'exit' to quit):")
    session_history = []
    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() == 'exit':
            print("Exiting the system. Goodbye!")
            break
        if len(user_query) < 3:
            print("Query too short. Please provide a more detailed question.")
            continue

        try:
            response = query_llm(user_query, vector_store)
            print(f"Answer: {response['answer']}")
            print("Sources:")
            for source in response['sources']:
                print(source)
            print()  # Add an empty line for better readability between queries
            
            session_history.append((user_query, response))
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a different query.")

    if session_history:
        print("\nSession History:")
        for i, (query, response) in enumerate(session_history, 1):
            print(f"{i}. Q: {query}")
            print(f"   A: {response['answer'][:100]}...")  # Truncate long answers
            print()

def load_vector_store():
    try:
        with open('vector_store.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def save_vector_store(vector_store):
    with open('vector_store.pkl', 'wb') as f:
        pickle.dump(vector_store, f)

if __name__ == "__main__":
    main()
