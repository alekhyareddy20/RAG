import streamlit as st
from query_data import evaluate_model, query_rag  # your function from the code you already have


# app.py
import streamlit as st



# ---------- PAGE 1 ----------
def page1():
    st.set_page_config(page_title="RAG Question Answering", layout="centered")
    st.title("📚 RAG Question Answering with GPT & Chroma")

    st.write(
        """
    Enter a question below and get the answer based on your document corpus.
    """
    )

    # Input field for the user question
    user_question = st.text_input("Enter your question:")
   
    # Button to trigger query
    if st.button("Get Answer", key = "answer") and user_question:
        with st.spinner("Generating response..."):
            try:
                # Call your RAG function
                response_text, results = query_rag(user_question)

                # Display the answer
                st.subheader("Answer")
                st.write(response_text)

                # Display the sources / document chunks used
                st.subheader("Sources")
                sources = [doc.metadata.get("id") for doc, _ in results]
                for src in sources:
                    st.write(f"- {src}")

            except Exception as e:
                st.error(f"Error: {e}")
    
    # Use a button as an arrow
    if st.button(" Want to Evaluate ??", on_click= go_to_page2,  key="btn2"):
        st.session_state.page = "page2"


# ---------- PAGE 2 ----------
def page2():
    st.title("Ask Your Question")
    
    option = st.selectbox(
        "Select the context/option:",
        [
            "what is ipact of AI in finance sector",
            "Finance",
            "Education",
            "Technology",
            "Environment"
        ]
    )

    
    if st.button("Get Answer", key = 'page2answer') and option:
        with st.spinner("Generating response..."):
            try:
                # Call your RAG function
                results, precision, recall, similarity_score = evaluate_model()

                # Display the answer
                st.subheader("Answer")
                st.write(results)

                st.subheader("Precision")
                st.write("Precision : ",precision)

                # Display the sources / document chunks used
                st.subheader("Recall")
                st.write("Precision : ",recall)

                st.subheader("Similarity Score")
                st.write(f"Similarity score : {similarity_score}")
                

            except Exception as e:
                st.error(f"Error: {e}")

     # Use a button as an arrow
    if st.button(" Back to HOME ??", on_click= go_to_page1,  key="btn1"):
        st.session_state.page = "page1"

    # if st.button("Generate Answer"):
    #     # Mock response; replace with your LLaMA/LLM API call
    #     response = f"Generated answer for '{question}' with context '{option}'"
        
    #     # Display in scrollable text area
    #     st.text_area("Generated Answer", value=response, height=300)

# ---------- STREAMLIT APP ----------
if "page" not in st.session_state:
    st.session_state.page = "page1"

def go_to_page1():
    st.session_state.page = "page1"

def go_to_page2():
    st.session_state.page = "page2"

if st.session_state.page == "page1":
    page1()
elif st.session_state.page == "page2":
    page2()

