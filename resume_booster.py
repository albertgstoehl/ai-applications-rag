import os
import glob
import pickle
import re
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
DATA_DIR = "data/" # Adjust path to your resume data
CHUNK_SIZE = 1000 # Approximate chunk size in characters
CHUNK_OVERLAP = 100 # Approximate overlap in characters
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
VECTORSTORE_INDEX_PATH = "faiss_index_resume.index"
VECTORSTORE_METADATA_PATH = "faiss_index_resume_metadata.pkl"
GEMINI_API_KEY_ENV = "GOOGLE_API_KEY" # Ensure this env var is set
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"
TOP_K_RESULTS = 5 # Number of relevant chunks to retrieve

# --- Load Environment Variables ---
load_dotenv()

# --- Helper Functions ---

def simple_text_splitter(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Basic text splitter based on character count with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # Ensure we don't overshoot due to overlap calculation
            break
        # Ensure overlap doesn't cause infinite loop on very short texts relative to overlap
        if len(chunks) > 1 and chunks[-1] == chunks[-2]:
             print(f"Warning: Potential splitting loop detected for text length {len(text)}")
             break # Prevent infinite loop for edge cases
    return chunks


def load_and_chunk_documents(data_dir):
    """Loads documents from PDF/TXT and chunks them, storing metadata."""
    chunks_with_metadata = []
    # Load PDFs
    for pdf_path in glob.glob(os.path.join(data_dir, "*.pdf")):
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                     text += page_text + "\n" # Add newline between pages

            if text.strip(): # Proceed if text was extracted
                 doc_chunks = simple_text_splitter(text)
                 for i, chunk in enumerate(doc_chunks):
                     chunks_with_metadata.append({
                         "text": chunk,
                         "source": filename,
                         "chunk_index": i
                     })
            else:
                print(f"Warning: No text extracted from {filename}")

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    # Load TXTs
    for txt_path in glob.glob(os.path.join(data_dir, "*.txt")):
        filename = os.path.basename(txt_path)
        print(f"Processing {filename}...")
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if text.strip():
                doc_chunks = simple_text_splitter(text)
                for i, chunk in enumerate(doc_chunks):
                    chunks_with_metadata.append({
                        "text": chunk,
                        "source": filename,
                        "chunk_index": i
                    })
            else:
                 print(f"Warning: No text read from {filename}")

        except Exception as e:
            print(f"Error processing {txt_path}: {e}")

    print(f"Loaded and chunked documents into {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata


def create_or_load_vectorstore(chunks_with_metadata, embedding_model):
    """Creates a FAISS index or loads it if it exists."""
    if os.path.exists(VECTORSTORE_INDEX_PATH) and os.path.exists(VECTORSTORE_METADATA_PATH):
        print(f"Loading existing vector store index from {VECTORSTORE_INDEX_PATH}")
        index = faiss.read_index(VECTORSTORE_INDEX_PATH)
        print(f"Loading existing metadata from {VECTORSTORE_METADATA_PATH}")
        with open(VECTORSTORE_METADATA_PATH, 'rb') as f:
            stored_metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} vectors.")
        return index, stored_metadata
    else:
        if not chunks_with_metadata:
             raise ValueError("No chunks provided to create a new vector store.")

        print("Creating new vector store...")
        texts = [item['text'] for item in chunks_with_metadata]

        print(f"Generating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL_NAME}...")
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32') # FAISS requires float32

        if not embeddings.any():
            raise ValueError("Embedding generation failed, received empty embeddings.")

        dimension = embeddings.shape[1]
        print(f"Creating FAISS index with dimension {dimension}...")
        index = faiss.IndexFlatL2(dimension) # Using L2 distance
        index.add(embeddings)

        print(f"Saving FAISS index to {VECTORSTORE_INDEX_PATH}...")
        faiss.write_index(index, VECTORSTORE_INDEX_PATH)

        print(f"Saving metadata to {VECTORSTORE_METADATA_PATH}...")
        with open(VECTORSTORE_METADATA_PATH, 'wb') as f:
            pickle.dump(chunks_with_metadata, f)

        print(f"Created and saved index with {index.ntotal} vectors.")
        return index, chunks_with_metadata


def search_index(query, embedding_model, index, stored_metadata, k=TOP_K_RESULTS):
    """Searches the FAISS index for relevant chunks."""
    print(f"Embedding query and searching index for top {k} results...")
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, k)

    results = []
    if indices.size > 0:
        for i, idx in enumerate(indices[0]):
             if 0 <= idx < len(stored_metadata): # Check index validity
                 results.append({
                    "metadata": stored_metadata[idx],
                    "distance": distances[0][i]
                 })
             else:
                 print(f"Warning: Retrieved invalid index {idx} from FAISS search.")
    return results


def rewrite_chunks_with_gemini(retrieved_chunks, target_role, api_key):
    """Uses Gemini API to rewrite resume chunks for the target role."""
    if not api_key:
        raise ValueError(f"API key not found. Set the {GEMINI_API_KEY_ENV} environment variable.")

    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)

    rewritten_bullets = []
    print("Rewriting relevant chunks using Gemini...")
    for item in retrieved_chunks:
        chunk_text = item['metadata']['text']
        source_id = item['metadata']['source']
        prompt = f"""
        Given the following resume excerpt from resume ID '{source_id}':
        ---
        {chunk_text}
        ---
        Rewrite the key responsibilities or achievements relevant to the target role '{target_role}' into concise, impactful bullet points.
        Use the second-person imperative voice (e.g., "Manage project budgets," "Develop software applications," "Lead cross-functional teams").
        Focus *only* on extracting and rephrasing relevant points from the provided text. If no part of the excerpt is relevant to the target role, output only the text "N/A".

        Rewritten Bullet Points:
        """
        try:
            response = model.generate_content(prompt)

            if response.parts:
                rewritten_text = response.text.strip()
            elif response.prompt_feedback.block_reason:
                 rewritten_text = f"Blocked: {response.prompt_feedback.block_reason}"
                 print(f"Warning: Generation blocked for chunk from {source_id}. Reason: {response.prompt_feedback.block_reason}")
            else:
                 rewritten_text = "Error: Empty response from model."
                 print(f"Warning: Empty response received from Gemini for chunk from {source_id}.")

            if rewritten_text.upper() != "N/A" and rewritten_text and "Blocked:" not in rewritten_text and "Error:" not in rewritten_text:
                 rewritten_bullets.append({
                     "original_content": chunk_text,
                     "rewritten": rewritten_text,
                     "source_id": source_id
                 })
            elif rewritten_text.upper() == "N/A":
                print(f"Chunk from {source_id} deemed not relevant by the model.")
            # Else: Blocked or Error cases handled above

        except Exception as e:
            print(f"Error calling Gemini API for chunk from {source_id}: {e}")
            # Optional: Log error or append an error message to output
            # rewritten_bullets.append({
            #     "original_content": chunk_text,
            #     "rewritten": f"Error processing: {e}",
            #     "source_id": source_id
            # })

    return rewritten_bullets

def main():
    """Main execution function."""
    # --- 1. Setup: Initialize Model, Load/Create Index ---
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}")
        print("Please ensure 'sentence-transformers' is installed and the model is available.")
        return

    index = None
    stored_metadata = []

    # Load or create the vector store
    try:
        if not os.path.exists(VECTORSTORE_INDEX_PATH) or not os.path.exists(VECTORSTORE_METADATA_PATH):
            print("--- Data Ingestion and Indexing ---")
            if not os.path.isdir(DATA_DIR):
                 print(f"Error: Data directory '{DATA_DIR}' not found or not a directory.")
                 print("Please create the directory and place your resume PDF/TXT files inside.")
                 return

            chunks_with_metadata = load_and_chunk_documents(DATA_DIR)
            if not chunks_with_metadata:
                 print(f"No processable documents found in {DATA_DIR}. Exiting.")
                 return

            index, stored_metadata = create_or_load_vectorstore(chunks_with_metadata, embedding_model)
        else:
             print("--- Loading Existing Index and Metadata ---")
             index, stored_metadata = create_or_load_vectorstore([], embedding_model) # Pass empty list when loading

        if index is None or not stored_metadata:
             print("Error: Failed to load or create the vector store.")
             return

    except Exception as e:
        print(f"Error during vector store setup: {e}")
        return


    # --- 2. User Interaction: Query and Generation ---
    print("\n--- Resume Booster ---")
    target_role_query = input("Enter the target role or a short job description: ")

    if not target_role_query.strip():
        print("No query entered. Exiting.")
        return

    # Retrieve relevant chunks
    try:
        relevant_chunks_data = search_index(target_role_query, embedding_model, index, stored_metadata)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return

    if not relevant_chunks_data:
        print("No relevant resume sections found in the indexed documents for your query.")
        return

    print(f"\nFound {len(relevant_chunks_data)} potentially relevant sections. Preparing suggestions...\n")

    # Load Gemini API Key
    gemini_api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not gemini_api_key:
         print(f"Error: Gemini API Key not found in environment variable {GEMINI_API_KEY_ENV}")
         print("Please set the environment variable and try again.")
         return

    # Rewrite using Gemini
    try:
        rewritten_suggestions = rewrite_chunks_with_gemini(relevant_chunks_data, target_role_query, gemini_api_key)
    except ValueError as e: # Catch specific API key error from the function
        print(f"Configuration Error: {e}")
        return
    except Exception as e: # Catch other potential errors during Gemini call
        print(f"Error during suggestion generation with Gemini: {e}")
        return

    # --- 3. Output ---
    print("\n--- Suggested Resume Bullets ---")
    if rewritten_suggestions:
        bullet_count = 0
        for suggestion in rewritten_suggestions:
             print(f"\nSource Résumé ID: {suggestion['source_id']}")
             print("Suggestion(s):")
             points = suggestion['rewritten'].split('\n')
             for point in points:
                 point = point.strip()
                 if point.startswith(("*", "-")):
                     point = point[1:].strip()
                 if point:
                     print(f"  - {point}")
                     bullet_count += 1
             print("-" * 20)
        if bullet_count == 0:
            print("No specific bullet points were generated based on the retrieved sections.")
        else:
            print(f"\nTotal bullet points generated: {bullet_count}")
    else:
        print("Could not generate suggestions. The relevant sections might not contain information suitable for rewriting or there was an issue.")


if __name__ == "__main__":
    # Initial check for data directory only if index files don't exist
    if not os.path.exists(VECTORSTORE_INDEX_PATH) or not os.path.exists(VECTORSTORE_METADATA_PATH):
        if not os.path.isdir(DATA_DIR):
            print(f"Error: Data directory '{DATA_DIR}' not found.")
            print("Please create this directory and add resume files (PDF/TXT) before the first run.")
            # Optionally create the directory:
            # try:
            #     os.makedirs(DATA_DIR, exist_ok=True)
            #     print(f"Created directory '{DATA_DIR}'. Please add resume files and run again.")
            # except OSError as e:
            #     print(f"Could not create directory '{DATA_DIR}': {e}")
        else:
             main() # Data dir exists, proceed with creation
    else:
         main() # Index exists, proceed to load
