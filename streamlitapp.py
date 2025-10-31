"""
Streamlit Web Application for RAG System
-----------------------------------------
This module provides a user-friendly web interface for the RAG (Retrieval-Augmented
Generation) system. Users can ask questions and evaluate system performance through
an interactive interface.
"""

import streamlit as st
from Qwen_model import evaluate_model, query_rag


def display_question_answering_page():
    """
    Display the main question answering interface.
    
    This page allows users to:
    - Enter custom questions
    - Get answers based on the document corpus
    - View source documents used for answers
    """
    # Configure page settings
    st.set_page_config(page_title="LLM Question Answering", layout="centered")
    st.title("📚 Question and Answering tool")

    st.write(
        """
        Enter a question below -> press enter -> and then press Get Answer button \n
        You will get an answer based on uploaded documents.
        """
    )

    # Input field for user's question
    user_question = st.text_input("Enter your question:", key="user_question_input")
   
    # Process question when button is clicked
    # Button is disabled if user_question is empty or contains only whitespace
    is_disabled = not user_question or not user_question.strip()
    if st.button("Get Answer", key="get_answer_button", disabled=is_disabled) and user_question:
        with st.spinner("Searching documents and generating response..."):
            try:
                # Query the RAG system
                generated_answer, retrieval_results = query_rag(user_question)

                # Display the generated answer
                st.subheader("Answer")
                st.write(generated_answer)

                st.subheader("Sources")
                # Handle empty results (no relevant sources)
                if not retrieval_results:
                    st.write("No sources available")
                else:
                    source_ids = [doc.metadata.get("id") for doc, _ in retrieval_results]
                    for source_id in source_ids:
                        st.write(f"- {source_id}")

                # Display source documents used for the answer
                # st.subheader("Sources")
                # source_ids = [doc.metadata.get("id") for doc, _ in retrieval_results]
                # for source_id in source_ids:
                #     st.write(f"- {source_id}")

            except Exception as error:
                st.error(f"An error occurred: {error}")
    
    # Navigation button to evaluation page
    if st.button("Want to Evaluate? →", on_click=navigate_to_evaluation_page, key="nav_to_eval"):
        st.session_state.current_page = "evaluation_page"


def display_evaluation_page():
    """
    Display the evaluation interface for testing system performance.
    
    This page allows users to:
    - Select predefined test questions
    - View generated answers
    - See evaluation metrics (precision and semantic similarity)
    """
    st.title("📊 RAG System Evaluation")
    
    # Predefined test questions for evaluation
    test_questions = [
        "How AI helps in Recruitment and HR Processes?",
        "How does the Event-Driven Architecture (EDA) enhance the responsiveness of real-time business intelligence systems like RTBISF?",
        "How are companies like Claryo and Synkrato pioneering the integration of digital twin technology?",
        "What is the future of AI in the manufacturing industry?",
        "What are the regulation issues and perspectives on AI in healthcare?"
    ]
    
    # Dropdown for selecting a test question
    selected_question = st.selectbox(
        "Select a test question:",
        test_questions,
        key="test_question_selector"
    )

    print(f"Selected question: {selected_question}")
    
    # Evaluate the selected question
    if st.button("Evaluate", key="evaluate_button") and selected_question:
        with st.spinner("Evaluating system performance..."):
            try:
                # Run evaluation
                generated_answer, precision_score, semantic_similarity = evaluate_model(selected_question)

                # Display generated answer
                st.subheader("Generated Answer")
                st.write(generated_answer)

                # Display precision metric
                st.subheader("Precision Score")
                st.write(f"**Precision:** {precision_score:.4f}")
                st.caption("Precision measures the accuracy of retrieved document chunks")

                # Display semantic similarity metric
                st.subheader("Semantic Similarity Score")
                st.write(f"**Similarity Score:** {semantic_similarity:.4f}")
                st.caption("Semantic similarity measures how close the generated answer is to the expected answer")

            except Exception as error:
                st.error(f"An error occurred during evaluation: {error}")

    # Navigation button back to main page
    if st.button("← Back to Home", on_click=navigate_to_home_page, key="nav_to_home"):
        st.session_state.current_page = "home_page"


def navigate_to_home_page():
    """Navigate to the main question answering page."""
    st.session_state.current_page = "home_page"


def navigate_to_evaluation_page():
    """Navigate to the evaluation page."""
    st.session_state.current_page = "evaluation_page"


# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "home_page"

# Route to appropriate page based on session state
if st.session_state.current_page == "home_page":
    display_question_answering_page()
elif st.session_state.current_page == "evaluation_page":
    display_evaluation_page()