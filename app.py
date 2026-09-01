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
            f"Semantic index ready with "
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