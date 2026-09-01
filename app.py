import streamlit as st

from rag_pipeline import (
    extract_pdf_pages,
    create_text_chunks,
    load_embedding_model,
    create_vector_index,
    retrieve_relevant_chunks,
    generate_grounded_answer,
    generate_topic_summary,
    generate_topic_quiz,
    generate_topic_flashcards
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

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #FFE4C7 0%,
                #FFF7ED 35%,
                #FDFCFB 75%
            );
        color: #292524;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #292524;
    }

    .stApp p,
    .stApp label {
        color: #57534E;
    }

    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.85);
        border: 2px dashed #D97706;
        border-radius: 16px;
        padding: 1rem;
    }

    [data-testid="stFileUploader"] button {
        background-color: #FFF7ED;
        color: #9A3412;
        border: 1px solid #FDBA74;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] small {
        color: #78716C;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #292524;
    }

    .stButton > button {
    min-height: 46px;
    border-radius: 12px;
    font-size: 0.95rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button[kind="secondary"] p {
    color: #7C2D12 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

.stButton > button[kind="primary"] p {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

div[class*="st-key-nav_"] button {
    min-height: 62px !important;
}

div[class*="st-key-nav_"] button p {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}

.stButton > button[kind="secondary"] {
    background-color: rgba(255, 255, 255, 0.9);
    color: #57534E;
    border: 1px solid #FED7AA;
    box-shadow: 0 4px 12px rgba(124, 45, 18, 0.07);
}

.stButton > button[kind="secondary"]:hover {
    color: #C2410C;
    border-color: #F97316;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(194, 65, 12, 0.12);
}

.stButton > button[kind="primary"] {
    color: white;
    background: linear-gradient(
        135deg,
        #EA580C,
        #F59E0B
    );
    border: none;
    box-shadow: 0 6px 16px rgba(234, 88, 12, 0.22);
}

.document-ready {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin: 1.25rem 0;
    padding: 1rem 1.2rem;
    background-color: rgba(255, 255, 255, 0.88);
    border: 1px solid #BBF7D0;
    border-left: 5px solid #22C55E;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.07);
}

.ready-icon {
    font-size: 1.4rem;
}

.ready-title {
    color: #166534;
    font-weight: 700;
}

.ready-details {
    color: #57534E;
    font-size: 0.9rem;
}

[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.88);
    border: 1px solid #FED7AA;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 12px rgba(124, 45, 18, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #78716C !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #9A3412 !important;
    font-weight: 700;
}

[data-testid="stTextInput"] input {
    background-color: rgba(255, 255, 255, 0.92);
    color: #292524;
    border: 1px solid #FED7AA;
    border-radius: 12px;
}

[data-testid="stTextInput"] input:focus {
    border-color: #F97316;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15);
}

[data-testid="stTextInput"] input::placeholder {
    color: #A8A29E;
}

[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.82);
    border: 1px solid #FED7AA;
    border-radius: 12px;
}

    </style>
    """,
    unsafe_allow_html=True
)



st.title("📚 AI Study Partner")

st.write(
    "Upload your study materials and learn through "
    "grounded answers, summaries, quizzes, and flashcards."
)


st.subheader("Upload Study Materials")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
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


        st.markdown(
            f"""
            <div class="document-ready">
                <div class="ready-icon">✅</div>
                <div>
                    <div class="ready-title">
                        Study materials ready
                    </div>
                    <div class="ready-details">
                        {len(uploaded_files)} document(s) indexed
                        with {vector_index.ntotal} searchable sections.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
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
        if "active_tool" not in st.session_state:

            st.session_state.active_tool = "Ask"

        navigation_columns = st.columns(4)

        navigation_items = [
            ("💬 Ask", "Ask"),
            ("📝 Summary", "Summary"),
            ("🧠 Quiz", "Quiz"),
            ("🗂️ Flashcards", "Flashcards")
        ]

        for navigation_column, (
            button_label,
            tool_name
        ) in zip(
            navigation_columns,
            navigation_items
        ):

            with navigation_column:

                if st.button(
                    button_label,
                    key=f"nav_{tool_name}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.active_tool
                        == tool_name
                        else "secondary"
                    )
                ):

                    st.session_state.active_tool = (
                        tool_name
                    )

                    st.rerun()

    

        if st.session_state.active_tool == "Ask":

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

                    show_sources = False
                    

                    try:
                        with st.spinner(
                            "Generating a grounded answer..."
                        ):
                            answer = generate_grounded_answer(
                                question=question,
                                retrieved_chunks=retrieved_chunks
                            )

                        st.subheader("AI Answer")
                        st.markdown(answer)
                        normalized_answer = answer.lower().replace(
                            "’",
                            "'"
                        )
                        show_sources = (
                            "could not find" not in normalized_answer
                            and "couldn't find" not in normalized_answer
                        )

                    except Exception as error:
                        st.error(
                            f"Answer generation failed: {error}"
                        )

                    if show_sources:
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
        if st.session_state.active_tool == "Summary":
            st.divider()
            st.subheader("Create a Topic Summary")

            summary_topic = st.text_input(
                "Enter a topic to summarize",
                placeholder="Example: Stacks",
                key="summary_topic"
            )

            summary_button = st.button(
                "Generate Summary",
                disabled=not summary_topic.strip()
            )

            if summary_button:
                summary_chunks = retrieve_relevant_chunks(
                    question=summary_topic,
                    text_chunks=text_chunks,
                    embedding_model=embedding_model,
                    vector_index=vector_index,
                    top_k=8
                )

                try:
                    with st.spinner(
                        "Creating your study summary..."
                    ):
                        summary = generate_topic_summary(
                            topic=summary_topic,
                            retrieved_chunks=summary_chunks
                        )

                    st.subheader(
                        f"Summary: {summary_topic}"
                    )
                    st.markdown(summary)

                    normalized_summary = (
                        summary.lower().replace("’", "'")
                    )

                    summary_has_sources = (
                        "could not find" not in normalized_summary
                        and "couldn't find" not in normalized_summary
                    )

                    if summary_has_sources:
                        st.markdown("#### Source Pages")
                        displayed_sources = set()

                        for chunk in summary_chunks:
                            source_key = (
                                chunk["source"],
                                chunk["page"]
                            )

                            if source_key not in displayed_sources:
                                st.markdown(
                                    f"- **{chunk['source']}** — "
                                    f"Page {chunk['page']}"
                                )

                                displayed_sources.add(
                                    source_key
                                )
                except Exception as error:
                    st.error(
                        f"Summary generation failed: {error}"
                                    )

        if st.session_state.active_tool == "Quiz":
            st.divider()
            st.subheader("Test Your knowledge")
            quiz_topic = st.text_input(
                "Enter a quiz topic",
                placeholder="Example: Stacks",
                key="quiz_topic"
            )

            generate_quiz_button = st.button(
                "Generate Quiz",
                disabled=not quiz_topic.strip()
            )

            if generate_quiz_button:
                quiz_chunks = retrieve_relevant_chunks(
                    question=quiz_topic,
                    text_chunks=text_chunks,
                    embedding_model=embedding_model,
                    vector_index=vector_index,
                    top_k=10
                )

                try:
                    with st.spinner(
                        "Creating your quiz.."
                    ):
                        quiz_questions = generate_topic_quiz(
                            topic=quiz_topic,
                            retrieved_chunks=quiz_chunks,
                            question_count=5
                        )

                    for session_key in list(
                        st.session_state.keys()
                    ):
                        if session_key.startswith(
                            "quiz_answer_"
                        ):
                            del st.session_state[
                                session_key
                            ]
                    st.session_state[
                        "quiz_questions"
                    ] = quiz_questions

                except Exception as error:
                    st.error(
                        f"Quiz generation failed: {error}"
                    )
            if "quiz_questions" in st.session_state:
                quiz_questions = st.session_state[
                    "quiz_questions"
                ]

                selected_answers = []

                with st.form("quiz_form"):
                    for question_number, quiz_item in enumerate(
                        quiz_questions,
                        start=1
                    ):
                        selected_answer = st.radio(
                            (f"{question_number}. "
                            f"{quiz_item['question']}"
                        ),
                        quiz_item["options"],
                        index=None,
                        key=(
                            f"quiz_answer_"
                            f"{question_number}"
                        )
                        )
                        selected_answers.append(
                            selected_answer
                        )
                    submit_quiz = st.form_submit_button(
                        "Submit Quiz"
                    )
                if submit_quiz:
                    if None in selected_answers:
                        st.warning(
                            "Please answer every question."
                        )
                    else:
                        score = 0
                        for selected_answer, quiz_item in zip(
                            selected_answers,
                            quiz_questions
                        ):
                            correct_option = quiz_item[
                                "options"
                            ][
                                quiz_item[
                                    "correct_answer"
                                ]
                            ]

                            if selected_answer == correct_option:
                                score += 1
                        st.success(
                            f"Your Score: {score}/"
                            f"{len(quiz_questions)}"
                        )

                        for question_number, (
                            selected_answer,
                            quiz_item
                        ) in enumerate(
                            zip(
                                selected_answers,
                                quiz_questions
                            ),
                            start=1
                        ):
                            correct_option = quiz_item[
                                "options"
                            ][
                                quiz_item[
                                    "correct_answer"
                                ]
                            ]

                            if selected_answer == correct_option:
                                st.success(
                                    f"Question {question_number}: "
                                    f"Correct"
                                )
                            else:
                                st.error(
                                    f"Question {question_number}: "
                                    f"Correct answer - "
                                    f"{correct_option}"
                                )

                            st.write(
                                quiz_item["explanation"]
                            )
        if st.session_state.active_tool == "Flashcards":

            st.divider()
            st.subheader("Create Flashcards")

            flashcard_topic = st.text_input(
                "Enter a flashcard topic",
                placeholder="Example: Stacks",
                key="flashcard_topic"
            )

            generate_flashcards_button = st.button(
                "Generate Flashcards",
                disabled=not flashcard_topic.strip()
            )

            if generate_flashcards_button:
                flashcard_chunks = retrieve_relevant_chunks(
                    question=flashcard_topic,
                    text_chunks=text_chunks,
                    embedding_model=embedding_model,
                    vector_index=vector_index,
                    top_k=10
                )

                try:
                    with st.spinner(
                        "Creating your flashcards..."
                    ):
                        flashcards = generate_topic_flashcards(
                            topic=flashcard_topic,
                            retrieved_chunks=flashcard_chunks,
                            card_count=8
                        )
                    st.session_state[
                        "flashcards"
                    ] = flashcards

                except Exception as error:
                    st.error(
                        f"Flashcard generation failed: {error}"
                    )
            if "flashcards" in st.session_state:
                flashcards = st.session_state[
                    "flashcards"
                ]

                st.caption(
                    "Open a card to reveal its answer."
                )

                for card_number, flashcard in enumerate(
                    flashcards,
                    start=1
                ):
                    with st.expander(
                        f"Card {card_number}: "
                        f"{flashcard['front']}"
                    ):
                        st.markdown(
                            f"**Answer:** "
                            f"{flashcard['back']}"
                        )
        
    else:

        st.error(
            "No readable text was found in the uploaded PDFs."
        )