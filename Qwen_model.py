"""
RAG Query and Evaluation Module
--------------------------------
This module implements Retrieval-Augmented Generation (RAG) using ChromaDB
for document retrieval and Hugging Face models for answer generation.
It includes evaluation metrics for assessing system performance.
"""

import os
import argparse
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from embed import HuggingFaceEmbeddingFunction

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration constants
VECTOR_DB_PATH = "chroma"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Prompt template for RAG system
SYSTEM_PROMPT_TEMPLATE = """
You are an expert assistant. Answer the user's question **using ONLY the information provided in the context below**. Do not assume or add information from outside the context. 

You may provide a detailed, verbose, and explanatory answer. Feel free to expand on concepts, examples, and reasoning, but do not introduce facts that are not in the context.

Context:
{context}

---

Answer the question based on the above context: {question}
"""

# Test dataset for evaluation
EVALUATION_TEST_DATA = [
    {
        "question": "How AI helps in Recruitment and HR Processes?",
        "expected_answer": "AI streamlines recruitment and HR processes by automating candidate screening, resume analysis, and employee onboarding. AI-powered tools can analyze resumes, assess candidates' skills, and match them with job requirements more efficiently than traditional methods, saving significant time and effort.\n\nBeyond recruitment, AI enhances the entire employee lifecycle through workforce analytics and decision-making support. By processing employee data, AI provides insights into workforce trends, skill gaps, and training needs, enabling data-driven decisions about talent development.\n\nAI also personalizes communication and development programs for individual employees, improving engagement and retention. AI chatbots provide 24/7 support for employee queries about policies and benefits, reducing administrative burden while improving employee experience.\n\nHowever, organizations must address ethical considerations, particularly regarding potential biases in AI algorithms that could lead to discriminatory hiring or promotion decisions."
    },
    {
        "question": "How does the Event-Driven Architecture (EDA) enhance the responsiveness of real-time business intelligence systems like RTBISF?",
        "expected_answer": "Event-Driven Architecture (EDA) enhances real-time business intelligence through the EDA-based Right-Time Business Intelligence System Framework (RTBISF), which combines real-time BI with business processes using EDA and Agent technologies.\n\nEDA's primary advantage is addressing environment uncertainty and business dynamics by enabling systems to react immediately to events as they occur. Instead of waiting for scheduled reports, the system detects significant business events—like inventory changes or supply chain disruptions—and triggers appropriate responses instantly.\n\nThe architecture continuously monitors data streams and business processes, meeting the needs for dynamic adjustment of business solutions in rapidly changing competitive environments. Integration of Agent technology allows autonomous analysis of events and initiation of appropriate actions, creating a self-adjusting system that adapts without constant human oversight.\n\nThis results in faster problem identification, reduced latency between events and responses, improved operational efficiency, and enhanced competitive advantage through agile decision-making."
    },
    {
        "question": "How are companies like Claryo and Synkrato pioneering the integration of digital twin technology?",
        "expected_answer": "Claryo and Synkrato are pioneering digital twin technology in warehouse operations through distinct but complementary approaches.\n\nClaryo's AI-powered Virtual Facility platform generates photo-realistic, spatially accurate digital representations of facilities using accessible methods like mobile phone photography and drone scanning. The platform's strength lies in seamless integration with existing facility management systems, providing real-time operational data that reflects current warehouse conditions and feeding this information back into management systems for continuous optimization.\n\nSynkrato's Warehouse Operating System (WOS) uses AI to rapidly generate digital twins and simulate real-world processes with high fidelity. It enables comprehensive testing of alternative operating scenarios without disrupting physical operations. The system ingests diverse data streams—inventory levels, order patterns, shipping timelines, and demand fluctuations—to generate accurate predictive models and actionable recommendations for optimal configuration and workflows.\n\nBoth platforms process vast operational data to identify optimal arrangements that maximize space efficiency while boosting performance and reducing costs, enabling warehouses to maintain operational agility in dynamic logistics environments."
    },
    {
        "question": "What is the future of AI in the manufacturing industry?",
        "expected_answer": "The future of AI in manufacturing is promising, driven by Industry 4.0 and digitalization trends. Key developments include:\n\nIntegration with IoT: Deeper AI-IoT integration will enable real-time data collection and analysis from connected devices, optimizing production processes and product quality.\n\nEnhanced Automation: AI will automate increasingly complex tasks requiring cognitive abilities and decision-making, freeing human resources for strategic work. Advanced analytics will provide sophisticated insights and predictive capabilities.\n\nEdge Computing: Processing data locally will reduce latency and improve AI system performance in real-time applications.\n\nHuman-AI Collaboration: Systems will enable seamless collaboration combining human creativity with AI processing power, rather than full replacement.\n\nAdvanced Capabilities: Natural language understanding will improve, robotics and autonomous systems will advance, and explainable AI will increase transparency and trust.\n\nCritical Challenges: Manufacturers must address cybersecurity, ethical considerations, scalability, and workforce training needs.\n\nMarket Growth: The AI in manufacturing market is projected to grow from USD 1.82 billion (2019) to USD 9.89 billion by 2027, with a CAGR of 24.2%."
    },
    {
        "question": "What are the regulation issues and perspectives on AI in healthcare?",
        "expected_answer": "AI healthcare regulation faces complex challenges as governments work to create effective frameworks.\n\nEstablished Frameworks: India's NITI Aayog develops national AI strategies with a vision of 'AI for all,' proposing a two-tiered structure of Research Excellence Centers and International Centers of Transformational AI. Other frameworks include Europe's GDPR and the US HIPAA for data protection.\n\nKey Challenges:\n1. Pace of Innovation: Regulatory bodies struggle to keep pace with rapid AI evolution\n2. What to Regulate: Identifying which AI models and use cases require scrutiny\n3. Who Should Regulate: Determining appropriate oversight levels (industry, state, federal, regional)\n\nData Privacy and Security: Medical data used for AI training requires extreme protection. Healthcare faces particular vulnerability since cyberattacks can be fatal. GDPR and HIPAA provide frameworks, but implementation remains challenging.\n\nTechnical Issues: Large datasets are required for AI training, but patient confidentiality and organizational reluctance to share data create obstacles. Overfitting, data leakage, and lack of explainability in deep learning algorithms pose additional challenges.\n\nEthical Considerations: Patient autonomy, informed consent, data privacy, and confidentiality violations must be addressed. Strong data protection laws are essential.\n\nBalance Required: Regulation must balance protecting patients with fostering innovation, enabling data sharing while preserving privacy, and promoting national competitiveness while cooperating on international standards."
    }
]

# Ground truth relevant chunks and documents for evaluation
RELEVANT_CHUNKS_MAPPING = {
    "How AI helps in Recruitment and HR Processes?": ["data/5.AI_on_BusinessProcesss.pdf:20:3"],
    "How does the Event-Driven Architecture (EDA) enhance the responsiveness of real-time business intelligence systems like RTBISF?": ["data/5.AI_on_BusinessProcesss.pdf:19:2"],
    "How are companies like Claryo and Synkrato pioneering the integration of digital twin technology?": ["data/3.AI_in_RealEstate.pdf:4:4", "data/3.AI_in_RealEstate.pdf:4:5"],
    "What is the future of AI in the manufacturing industry?": ["data/2.AI_in_manufacturing_industry.pdf:6:1", "data/2.AI_in_manufacturing_industry.pdf:6:2", "data/5.AI_on_BusinessProcesss.pdf:20:4", "data/5.AI_on_BusinessProcesss.pdf:21:1", "data/2.AI_in_manufacturing_industry.pdf:7:2"],
    "What are the regulation issues and perspectives on AI in healthcare?": ["data/4.AI_in_healthsector.pdf:4:1", "data/4.AI_in_healthsector.pdf:4:2", "data/4.AI_in_healthsector.pdf:4:3", "data/4.AI_in_healthsector.pdf:3:5", "data/4.AI_in_healthsector.pdf:3:6"]
}

RELEVANT_DOCUMENTS_MAPPING = {
    "How AI helps in Recruitment and HR Processes?": ["data/5.AI_on_BusinessProcesss.pdf"],
    "How does the Event-Driven Architecture (EDA) enhance the responsiveness of real-time business intelligence systems like RTBISF?": ["data/5.AI_on_BusinessProcesss.pdf"],
    "How are companies like Claryo and Synkrato pioneering the integration of digital twin technology?": ["data/3.AI_in_RealEstate.pdf", "data/3.AI_in_RealEstate.pdf"],
    "What is the future of AI in the manufacturing industry?": ["data/2.AI_in_manufacturing_industry.pdf", "data/2.AI_in_manufacturing_industry.pdf", "data/5.AI_on_BusinessProcesss.pdf", "data/5.AI_on_BusinessProcesss.pdf", "data/2.AI_in_manufacturing_industry.pdf"],
    "What are the regulation issues and perspectives on AI in healthcare?": ["data/4.AI_in_healthsector.pdf", "data/4.AI_in_healthsector.pdf", "data/4.AI_in_healthsector.pdf", "data/4.AI_in_healthsector.pdf", "data/4.AI_in_healthsector.pdf"]
}


def generate_llm_response(prompt_text: str) -> str:
    """
    Generate a response using Hugging Face's LLM via Inference API.
    
    Args:
        prompt_text (str): The formatted prompt containing context and question
        
    Returns:
        str: Generated answer from the language model
    """
    # Format the prompt as a chat message
    chat_messages = [{"role": "user", "content": prompt_text}]
    
    # Initialize the inference client with the specified LLM
    llm_client = InferenceClient(model=LLM_MODEL_NAME)
    
    # Generate response using chat completion
    api_response = llm_client.chat_completion(
        model=LLM_MODEL_NAME,
        messages=chat_messages,
        max_tokens=300,
    )
    
    # Extract and return the generated text
    return api_response["choices"][0]["message"]["content"]


def query_rag(user_query: str):
    """
    Perform Retrieval-Augmented Generation to answer a user query.
    
    This function:
    1. Searches the vector database for relevant document chunks
    2. Combines retrieved chunks as context
    3. Generates an answer using the LLM with the context
    
    Args:
        user_query (str): The user's question
        
    Returns:
        tuple: (generated_answer, search_results)
            - generated_answer (str): The LLM's response
            - search_results (list): Retrieved documents with similarity scores
    """
    # Initialize embedding function for query encoding
    embedding_function = HuggingFaceEmbeddingFunction(
        model_id=EMBEDDING_MODEL_NAME
    )
    
    # Load the vector database
    vector_database = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_function
    )

    # Retrieve top 5 most similar document chunks
    search_results = vector_database.similarity_search_with_score(user_query, k=5)

    # Combine retrieved chunks into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in search_results])
    
    # Format the prompt with context and question
    prompt_formatter = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    formatted_prompt = prompt_formatter.format(context=context_text, question=user_query)

    # Generate answer using the LLM
    generated_answer = generate_llm_response(formatted_prompt)

    # Extract and display source document IDs
    source_ids = [doc.metadata.get("id", None) for doc, _ in search_results]
    formatted_output = f"Response: {generated_answer}\nSources: {source_ids}"
    print(formatted_output)
    print("Retrieved results:", search_results)

    return generated_answer, search_results


def evaluate_model(test_question: str):
    """
    Evaluate the RAG system's performance on a test question.
    
    Computes:
    - Precision: Accuracy of retrieved documents/chunks
    - Semantic Similarity: How close the generated answer is to the expected answer
    
    Args:
        test_question (str): Question from the test dataset
        
    Returns:
        tuple: (generated_answer, precision_score, similarity_score)
    """
    # Find the expected answer for the given question
    expected_answer = None
    for test_item in EVALUATION_TEST_DATA:
        if test_item["question"].strip().lower() == test_question.strip().lower():
            expected_answer = test_item["expected_answer"]
            break

    # Generate answer using RAG
    generated_answer, retrieved_results = query_rag(test_question)

    # Handle case where no documents were retrieved
    if not retrieved_results:
        print(f"\n⚠️ No documents retrieved for: {test_question}")
        return "NO RELEVANT DOCUMENTS FOUND", 0.0, 0.0

    # Calculate semantic similarity between generated and expected answers
    embedding_function = HuggingFaceEmbeddingFunction(EMBEDDING_MODEL_NAME)
    generated_embedding = embedding_function.embed_query(generated_answer)
    expected_embedding = embedding_function.embed_query(expected_answer)
    
    # Convert to tensors for cosine similarity computation
    generated_tensor = torch.tensor(generated_embedding)
    expected_tensor = torch.tensor(expected_embedding)
    
    # Apply mean pooling if embeddings are multi-dimensional
    if len(generated_tensor.shape) > 1:
        generated_tensor = torch.mean(generated_tensor, dim=0)
    if len(expected_tensor.shape) > 1:
        expected_tensor = torch.mean(expected_tensor, dim=0)
    
    # Compute cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        expected_tensor, generated_tensor, dim=0
    )
    similarity_score = cosine_similarity.item()
    print(f"Semantic similarity: {similarity_score:.4f}")

    # Calculate precision at chunk and document level
    print(f"\n🧠 Question: {test_question}")
    print(f"Semantic Similarity: {similarity_score:.2f}")
    print(f"✅ Generated Answer: {generated_answer}\n")
    
    # Extract retrieved chunk IDs and document names
    retrieved_chunk_ids = extract_chunk_ids(retrieved_results)
    retrieved_document_names = set(chunk_id.split(":")[0] for chunk_id in retrieved_chunk_ids)
    
    # Get ground truth relevant chunks and documents
    ground_truth_documents = set(RELEVANT_DOCUMENTS_MAPPING[test_question])
    ground_truth_chunks = set(RELEVANT_CHUNKS_MAPPING[test_question])

    # Calculate document-level precision
    document_precision = len(retrieved_document_names & ground_truth_documents) / len(retrieved_document_names)
    print(f"Document-level Precision: {document_precision:.2f}")

    # Calculate chunk-level precision
    chunk_precision = len(set(retrieved_chunk_ids) & ground_truth_chunks) / len(retrieved_chunk_ids)
    print(f"Chunk-level Precision: {chunk_precision:.2f}")
   
    return generated_answer, chunk_precision, similarity_score


def extract_chunk_ids(search_results):
    """
    Extract chunk IDs from search results.
    
    Args:
        search_results (list): List of (document, score) tuples from vector search
        
    Returns:
        list[str]: List of chunk IDs
    """
    chunk_ids = []
    for document, score in search_results:
        chunk_ids.append(document.metadata['id'])
    return chunk_ids


def main():
    """
    Main entry point for command-line interface.
    Handles query execution and evaluation modes.
    """
    # Parse command-line arguments
    argument_parser = argparse.ArgumentParser(
        description="RAG system for question answering with optional evaluation"
    )
    argument_parser.add_argument(
        "query_text", 
        type=str, 
        nargs="?", 
        help="The question to answer"
    )
    argument_parser.add_argument(
        "--eval", 
        action="store_true", 
        help="Run in evaluation mode with metrics"
    )
    parsed_args = argument_parser.parse_args()

    # Execute in appropriate mode
    if parsed_args.eval:
        if not parsed_args.query_text:
            print("Error: Please provide a question for evaluation.")
            return
        evaluation_results = evaluate_model(parsed_args.query_text)
    else:
        if not parsed_args.query_text:
            print("Error: Please provide a query or use --eval for evaluation mode.")
            return
        # Standard query mode (could be implemented here)


if __name__ == "__main__":
    main()