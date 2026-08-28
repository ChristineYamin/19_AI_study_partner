import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def extract_pdf_pages(uploaded_files):

    extracted_pages = []

    for uploaded_file in uploaded_files:

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                extracted_pages.append(
                    {
                        "source": uploaded_file.name,
                        "page": page_number,
                        "text": text.strip()
                    }
                )

    return extracted_pages

def create_text_chunks(
    extracted_pages,
    chunk_size=200,
    chunk_overlap=40
):

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    step_size = chunk_size - chunk_overlap

    for page_data in extracted_pages:

        words = page_data["text"].split()

        for start_index in range(
            0,
            len(words),
            step_size
        ):

            chunk_words = words[
                start_index:start_index + chunk_size
            ]

            if not chunk_words:

                continue

            chunks.append(
                {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "text": " ".join(chunk_words)
                }
            )

    return chunks

def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


def create_vector_index(
    text_chunks,
    embedding_model
):

    chunk_texts = [
        chunk["text"]
        for chunk in text_chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    vector_index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_index.add(embeddings)

    return vector_index

def retrieve_relevant_chunks(
    question,
    text_chunks,
    embedding_model,
    vector_index,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    similarity_scores, chunk_indices = (
        vector_index.search(
            question_embedding,
            min(top_k, len(text_chunks))
        )
    )

    retrieved_chunks = []

    for score, chunk_index in zip(
        similarity_scores[0],
        chunk_indices[0]
    ):

        if chunk_index == -1:

            continue

        retrieved_chunk = text_chunks[
            chunk_index
        ].copy()

        retrieved_chunk["similarity_score"] = float(
            score
        )

        retrieved_chunks.append(
            retrieved_chunk
        )

    return retrieved_chunks