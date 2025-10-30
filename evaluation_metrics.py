from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from ollama import chat
import numpy as np

CHROMA_PATH = "chroma"

# Load embedding model for similarity metric
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define test dataset
test_data = [
    {
        "question": "What is the impact of AI in healthcare?",
        "expected_answer": "AI is transforming healthcare by improving diagnosis accuracy, enabling personalized treatments, predicting and preventing illnesses, accelerating drug discovery, and enhancing telemedicine and remote patient monitoring—ultimately improving patient outcomes and resource efficiency in hospitals."
    },
#    {
#        "question": "What are the benefits of AI in healthcare?",
#        "expected_answer": "AI assists in diagnostics, treatment planning, and patient monitoring."
#    },
#    {
#        "question": "How does AI help IT industry?",
#        "expected_answer": "AI automates routine IT tasks, enhances cybersecurity, and supports predictive maintenance."
#    },
]

# Initialize Chroma DB and embedding function
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


def get_answer_from_ollama(question, context):
    prompt = f"Answer based only on the context:\n\n{context}\n\nQuestion: {question}"
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']


def evaluate_model():
    precision_list, recall_list, sim_scores = [], [], []

    for data in test_data:
        question = data["question"]
        expected_answer = data["expected_answer"]

        # 1️⃣ Retrieve top-5 documents
        results = db.similarity_search_with_score(question, k=5)

        # Mock relevance: assume top-2 retrieved are relevant (for demo)
        y_true = [1, 1, 0, 0, 0]  # 1=relevant, 0=irrelevant
        y_pred = [1 if i < 2 else 0 for i in range(len(results))]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision_list.append(precision)
        recall_list.append(recall)

        # 2️⃣ Combine retrieved text as context
        context_text = "\n\n".join([doc.page_content for doc, _ in results])

        # 3️⃣ Generate answer using Ollama
        generated_answer = get_answer_from_ollama(question, context_text)

        # 4️⃣ Semantic similarity
        emb_expected = similarity_model.encode(expected_answer, convert_to_tensor=True)
        emb_generated = similarity_model.encode(generated_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_expected, emb_generated).item()
        sim_scores.append(similarity)

        print(f"\nQuestion: {question}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, Semantic Similarity: {similarity:.2f}")
        print(f"Generated Answer: {generated_answer}\n")

    print("\n===== Overall Evaluation =====")
    print(f"Average Precision: {np.mean(precision_list):.2f}")
    print(f"Average Recall: {np.mean(recall_list):.2f}")
    print(f"Average Semantic Similarity: {np.mean(sim_scores):.2f}")


if __name__ == "__main__":
    evaluate_model()


------
import numpy as np
from typing import List, Set, Union, Tuple

def precision_at_k(
    retrieved_items: List[Union[str, int]], 
    relevant_items: Set[Union[str, int]], 
    k: int
) -> float:
    """
    Calculates Precision@k: the fraction of the top k retrieved documents that are relevant.

    Args:
        retrieved_items: A list of items returned by the retriever (ordered by rank).
        relevant_items: A set of all known relevant items in the knowledge base.
        k: The number of top items to consider.

    Returns:
        The Precision@k score (float between 0.0 and 1.0).
    """
    # 1. Take only the top k items
    top_k_retrieved = retrieved_items[:k]
    
    # 2. Count how many of the top k items are actually relevant
    relevant_in_top_k = sum(1 for item in top_k_retrieved if item in relevant_items)
    
    # 3. Precision = (Relevant in top k) / (Total in top k)
    if k == 0:
        return 0.0
    return relevant_in_top_k / k

def recall_at_k(
    retrieved_items: List[Union[str, int]], 
    relevant_items: Set[Union[str, int]], 
    k: int
) -> float:
    """
    Calculates Recall@k: the fraction of all relevant items that are retrieved in the top k.

    Args:
        retrieved_items: A list of items returned by the retriever (ordered by rank).
        relevant_items: A set of all known relevant items in the knowledge base.
        k: The number of top items to consider.

    Returns:
        The Recall@k score (float between 0.0 and 1.0).
    """
    # 1. Take only the top k items
    top_k_retrieved = retrieved_items[:k]
    
    # 2. Count how many of the top k items are actually relevant
    relevant_in_top_k = sum(1 for item in top_k_retrieved if item in relevant_items)
    
    # 3. Recall = (Relevant in top k) / (Total number of relevant items)
    total_relevant = len(relevant_items)
    
    if total_relevant == 0:
        # If there are no relevant documents, recall is typically 1.0 (vacuously true)
        return 1.0
    return relevant_in_top_k / total_relevant
