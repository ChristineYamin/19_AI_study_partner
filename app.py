import streamlit as st

from rag_pipeline import (
    extract_pdf_pages,
    create_text_chunks,
    load_embedding_model,
    create_vector_index,
    retrieve_relevant_chunks
)

@st.cache_data(
    show_spinner=False,
    max_entries=3
)
def process_documents(uploaded_files):

    extracted_pages = extract_pdf_pages(
        uploaded_files
    )

    text_chunks = create_text_chunks(
        extracted_pages
    )

    return extracted_pages, text_chunks



@st.cache_resource(show_spinner=False)
def get_embedding_model():

    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_vector_index(chunk_texts):

    embedding_model = get_embedding_model()

    chunks_for_index = [
        {"text": text}
        for text in chunk_texts
    ]

    return create_vector_index(
        chunks_for_index,
        embedding_model
    )

st.set_page_config(
    page_title="AI Study Partner",
    page_icon="📚",
    layout="wide"
)


st.title("📚 AI Study Partner")

st.write(
    "Upload your study materials and learn through "
    "grounded answers, summaries, quizzes, and flashcards."
)


with st.sidebar:

    st.header("Study Materials")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )


if not uploaded_files:

    st.info(
        "Upload at least one PDF to begin studying."
    )

else:

    with st.spinner("Reading your documents..."):

        extracted_pages, text_chunks = process_documents(uploaded_files)
        



    if extracted_pages:

        with st.spinner(
            "Creating the semantic search index..."
        ):
            embedding_model = get_embedding_model()
            chunk_texts = tuple(
                chunk["text"] for chunk in text_chunks
            )
            vector_index = get_vector_index(chunk_texts)


        st.success(
            f"Semantic index ready with"
            f"{vector_index.ntotal} vectors."
        )   

        st.success(
            f"Successfully extracted "
            f"{len(extracted_pages)} page(s) from "
            f"{len(uploaded_files)} document(s)."
        )

        page_column, chunk_column = st.columns(2)
        page_column.metric(
             "Readable Pages",
              len(extracted_pages)
        )

        chunk_column.metric(
            "Text Chunks",
            len(text_chunks)
        )

        with st.expander("Preview extracted text"):

            first_page = extracted_pages[0]

            st.write(
                f"**Source:** {first_page['source']}"
            )

            st.write(
                f"**Page:** {first_page['page']}"
            )

            st.write(first_page["text"][:2000])

        st.divider()

        st.subheader("Search Your Study Materials")

        question = st.text_input(
            "Ask a question about your documents"
        )

        search_button = st.button(
            "Find Relevant Information",
            type="primary",
            disabled=not question.strip()
        )

        if search_button:

            retrieved_chunks = retrieve_relevant_chunks(
                question=question,
                text_chunks=text_chunks,
                embedding_model=embedding_model,
                vector_index=vector_index,
                top_k=5
            )

            st.write(
                f"Found {len(retrieved_chunks)} "
                f"relevant sections:"
            )

            for result_number, result in enumerate(
                retrieved_chunks,
                start=1
            ):

                with st.expander(
                    f"Result {result_number} — "
                    f"{result['source']}, "
                    f"page {result['page']}"
                ):

                    st.caption(
                        f"Similarity score: "
                        f"{result['similarity_score']:.3f}"
                    )

                    st.write(result["text"])

    else:

        st.error(
            "No readable text was found in the uploaded PDFs."
        )