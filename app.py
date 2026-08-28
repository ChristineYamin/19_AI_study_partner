import streamlit as st


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

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) uploaded"
        )


st.subheader("Ask Your Documents")

st.info(
    "Upload at least one PDF to begin studying."
)