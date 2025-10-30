import os
import argparse
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score

from get_embedding_function import get_embedding_function

from huggingface_hub import InferenceClient



messages = [{"role": "user", "content": "What is the capital of France?"}]
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")






def generate_hf_api_response(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")  # or another llama model
    response = client.chat_completion(messages, max_tokens=100)
    # response is a dict with 'content' key inside messages
    return response["choices"][0]["message"]["content"]

# --- Environment setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# --- Initialize embedding & similarity models ---
embedding_function = get_embedding_function()
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 🔹 QUERY FUNCTION (your original logic)
# ============================================================
def query_rag(query_text: str):
    # Prepare the DB (explicit collection name to avoid blank responses)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        #collection_name="rag_collection",
        embedding_function=embedding_function,
    )

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine retrieved chunks as context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response using Ollama
    response_text = generate_hf_api_response(prompt)

    # Display sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    print("Results", results)

    return response_text, results


# ============================================================
# 🔹 EVALUATION FUNCTION (Precision, Recall, Semantic Similarity)
# ============================================================
def evaluate_model():
    # Test data
    test_data = [
        {
            "question": "What is the impact of AI in Finance Sector?",
            "expected_answer": ''' Artificial Intelligence is creating a major transformation across the financial industry by changing how financial institutions operate, make decisions, and manage customer interactions. AI is being used to automate complex workflows, reduce manual effort, and increase the speed and precision of financial services. Banks and financial service firms are using AI to analyze large amounts of data, detect fraud, monitor transactions, and predict market trends. For example, AI systems can identify unusual customer behavior in real-time, which helps prevent fraud more effectively than traditional methods.

Additionally, AI is contributing to better customer experiences by enabling personalized financial recommendations and chat-based customer support. However, along with these benefits, financial institutions must handle challenges such as ensuring data privacy, cybersecurity, responsible AI use, and compliance with regulations. Because the financial sector is highly regulated, companies are developing structured responsible AI frameworks to ensure ethical and transparent use of AI.
 '''       }
    ]

    precision_list, recall_list, sim_scores = [], [], []

    for data in test_data:
        question = data["question"]
        expected_answer = data["expected_answer"]

        # Run your query_rag function
        generated_answer, results = query_rag(question)

        if not results:
            print(f"\n⚠️ No documents retrieved for: {question}")
            continue

        

        # Semantic Similarity
        emb_expected = similarity_model.encode(expected_answer, convert_to_tensor=True)
        emb_generated = similarity_model.encode(generated_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_expected, emb_generated).item()
        sim_scores.append(similarity)

        print(f"\n🧠 Question: {question}")
        print(f"Semantic Similarity: {similarity:.2f}")
        print(f"✅ Generated Answer: {generated_answer}\n")
    
    

    # Average results
    precision = 1
    recall = 2
   
    
    print(f"Average Semantic Similarity: {np.mean(sim_scores):.2f}")
    return generated_answer, precision, recall, similarity 

def get_relavant_chuncks(results):
  relevant_docs = []
  for doc, score in results:
    relevant_docs.append(doc.metadata['id'])
  return relevant_docs




# ============================================================
# 🔹 MAIN CLI ENTRY
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, nargs="?", help="The query text.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode.")
    args = parser.parse_args()


    if args.eval:
        
        results = evaluate_model()
    else:
        if not args.query_text:
            print("Please provide a query or use --eval for evaluation.")
            return
    
    
    retrieved_chunks = get_relavant_chuncks(results)
    retrieved_docs = set(chunk.split(":")[0] for chunk in retrieved_chunks)
    print(retrieved_chunks,"nv skjdv ksjnfksjnf kasf")
    print("Retrieved docs", retrieved_docs)
    print("Retrieved chunks", retrieved_chunks)
    relevant_docs = set(['data/Artificial_Intelligence_in_Financial_Services_2025.pdf'])
    relevant_chunks= set(['data/Artificial_Intelligence_in_Financial_Services_2025.pdf:6:1',
"data/Artificial_Intelligence_in_Financial_Services_2025.pdf:7:1",
"data/Artificial_Intelligence_in_Financial_Services_2025.pdf:7:2",
"data/Artificial_Intelligence_in_Financial_Services_2025.pdf:12:1",
"data/Artificial_Intelligence_in_Financial_Services_2025.pdf:15:2"])

    precision = len(retrieved_docs & relevant_docs) / len(retrieved_docs)
    recall = len(retrieved_docs & relevant_docs) / len(relevant_docs)

    print(f"PRECISION OF DOCS: {precision:.2f}, RECALL OF DOCS: {recall:.2f}")

    precision = len(set(retrieved_chunks) & relevant_chunks) / len(retrieved_chunks)
    recall = len(set(retrieved_chunks) & relevant_chunks) / len(relevant_chunks)

    print(f"PRECISION (chunk-level): {precision:.2f}, RECALL (chunk-level): {recall:.2f}")
   



if __name__ == "__main__":
    main()