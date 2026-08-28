import streamlit as st

from rag_pipeline import (
    extract_pdf_pages,
    create_text_chunks,
    load_embedding_model,
    create_vector_index
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

        extracted_pages = extract_pdf_pages(
            uploaded_files
        )

        text_chunks = create_text_chunks(
            extracted_pages
        )



    if extracted_pages:

        with st.spinner(
            "Creating the semantic search index..."
        ):
            
            embedding_model = load_embedding_model()

            vector_index = create_vector_index(
                text_chunks,
                embedding_model
            )

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

    else:

        st.error(
            "No readable text was found in the uploaded PDFs."
        )