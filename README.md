# RAG

## Setup and Installation Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/alekhyareddy20/RAG.git
cd RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Hugging Face API Key
> **Note:** I used Hugging Face CLI for authentication instead of hardcoding tokens.

1. Create a token at [Hugging Face]
2. Log in using the CLI:
```bash
huggingface-cli login
```
3. Enter your token when prompted

### 5. Create or Update the Chroma DB
```bash
# Standard creation:
python DB_model.py

# Clear existing database before creating new one:
python DB_model.py --reset
```

### 6. Launch Web Interface
```bash
streamlit run streamlitapp.py
```
This will automatically open the application in your browser.


## 2) A brief description of the approach and design decisions

### System Flow
```
PDF Documents → Text Extraction → Chunking → Embedding → Vector DB (ChromaDB) → User Query → Query Embedding → Similarity Search → Context Retrieval → LLM Generation → Answer + Citations & Evaluation
```

### Current Model Configuration
- **Embedding Model:** `Qwen/Qwen3-Embedding-8B`
- **Language Model:** `meta-llama/Llama-3.1-8B-Instruct`

---

### Design Decisions & Model Selection Journey

#### 1. Document Processing Strategy
I chose **RecursiveCharacterTextSplitter** with **800-character chunks** and **80-character overlap**.

**Why these numbers?**
- 800 characters give enough context for the model to understand what's going on
- 80-character overlap ensures I don't lose important information at chunk boundaries
- I tested smaller chunks (400) and larger ones (1200), but 800 hit the sweet spot

---

#### 2. Finding the Right Embedding Model

I tried several approaches before landing on the final setup. Here's what happened:

**❌ Attempt 1: Ollama (Llama 3 - Local)**
- **Why I tried it:** Full local control, no API costs, privacy benefits
- **The problem:** Query times of 7-10 minutes per question! My GPU was constantly maxed out (8GB+ VRAM), and my system became unresponsive
- **Verdict:** Great concept, but completely impractical for real-time use

**⚠️ Attempt 2: all-MiniLM-L6-v2**
- **Why I tried it:** Super lightweight (22M parameters), blazing fast (< 1 second)
- **The problem:** Semantic similarity scores were inconsistent (0.50-0.65), and I felt the system struggled with technical AI/ML terminology
- **Verdict:** Good for general projects, but not quite good enough for this use case

**⚠️ Attempt 3: e5-mistral-7b-instruct**
- **Why I tried it:** High MTEB ranking, instruction-following capabilities
- **The problem:** Slower inference (4-5 seconds), designed more for chat than pure embeddings, and API rate limits were stricter
- **Verdict:** Capable, but not the best fit for what I needed

**✅ Final Choice: Two-Model Architecture**

After all that testing, I went with a **specialized approach**:

**For Embeddings:** `Qwen/Qwen3-Embedding-8B`
- Ranked #8 on MTEB leaderboard for retrieval tasks
- Purpose-built for embedding generation (not a repurposed LLM)
- Fast API inference (2-3 seconds)
- Consistently strong performance on technical content
- Average precision: 0.84

**For Text Generation:** `meta-llama/Llama-3.1-8B-Instruct`
- Excellent at following instructions and reasoning
- Semantic similarity scores: 0.75-0.85 (way better than earlier attempts)
- Fast response times (2-4 seconds per query)
- Free tier API access works great for development

**Why separate models?**
Using specialized models for each task turned out way better than trying to make one model do everything. The embedding model is laser-focused on retrieval, while the LLM handles generation. Plus, I can optimize or swap them independently.
Since I tried the same model and saw that there was low semantic similarity, compared to the approach I followed now.

---

#### 3. Models I Considered But Didn't Use

**OpenAI Models (text-embedding-3-large, GPT-4)**
- **Why I was interested:** Industry-leading quality, great docs, proven track record
- **Why I didn't use them:** Cost scaling with usage, wanted to explore open-source options first
- **Future note:** Definitely worth trying for production with budget—quality would likely be even better

**Cohere Embed v3**
- **Why I was interested:** Competitive quality, strong multilingual support
- **Why I didn't use them:** Similar cost concerns, less community support

---

### Model Selection Criteria

Here's what I prioritized when choosing models:

1. **MTEB Leaderboard Ranking** - Focused specifically on retrieval task performance
2. **Inference Speed** - Target under 10 seconds per query for good user experience
3. **API Availability** - Needed accessible deployment without complex infrastructure
4. **Cost** - Free tier or reasonable pricing for development/testing phase
5. **Context Length** - Ability to handle my 800-character chunks effectively

---

### Key Learnings

- **Bigger isn't always better** - Qwen outperformed the larger e5-mistral on my specific task
- **Specialization matters** - Models built for specific tasks (like embeddings) often beat general-purpose models
- **Architecture flexibility** - A two-model setup gives you more control and optimization options
- **Real-world testing is crucial** - Leaderboard scores don't tell the whole story; you have to test with your actual data
- **Think holistically** - The best system balances speed, quality, AND cost—not just accuracy alone

## 4. Vector Database: ChromaDB

-  **Lightweight and easy to set up** — no external dependencies required  
-  **Excellent persistence and fast in-memory performance**  
-  **Built-in embedding function interface** enables seamless integration  
-  **No need for a separate server or container setup**  
- **Ideal for moderate-scale collections** (under ~10K chunks)

---

## 5. Retrieval Strategy

**Decision:** Retrieve **top-5 most similar chunks (`k=5`)**

**Rationale:**
-  Provides sufficient context for accurate answers  
- `k=3` → produced incomplete answers (too little context)  
- `k=10` → added noise and reduced precision  
- `k=5` → best tradeoff between **precision** and **recall**

---

## 6. Prompt Engineering

The system uses a **structured, context-restricted prompt template** designed to:

-  **Restrict** the LLM to only use the provided context  
-  **Encourage** detailed and well-structured explanations  
-  **Prevent hallucination** by reinforcing context boundaries  
-  **Separate** context and question sections clearly for better control



## 3) Technologies  previously familiar vs. new 

---

## Previously Familiar Technologies

- **Python:** Core programming language used for scripting, data processing, and model integration  
- **NumPy:** Efficient array operations and numerical computing  
- **AI/ML Concepts:** Familiar with embeddings, vector similarity, and model evaluation  
- **LangChain:** Experience with document loaders, text splitters, and prompt templates  
- **ChromaDB:** Used as a vector database for storing and retrieving document embeddings  
- **Hugging Face Inference API:** Worked with serverless model deployment and API-based integration  
- **RAG Architecture:** Implemented retrieval-augmented generation (RAG) workflows for document-based Q&A  
- **Semantic Similarity Metrics:** Used cosine similarity for measuring relevance between embeddings  

---

## New Technologies & Learnings

- **Streamlit:** Built interactive web apps for AI-powered document Q&A systems  
- **PyTorch (for Embeddings):** Learned tensor operations and embedding generation for semantic search  
- **Vector Search Optimization:** Explored tuning `k` values and similarity thresholds for better retrieval precision  
- **Hugging Face Models:** Experimented with multiple new models for embeddings and text generation, and continue exploring more advanced ones to improve performance and accuracy


## 4) Evaluation results and interpretation

--- See the full questions and expected answers in the provided txt file in the repository.
### Evaluation:

## 1) Ground Truth Creation

Since this is an open-ended QA task without pre-existing gold standards, I created ground truth answers using a human-expert evaluation approach.  

**Process:**

1. Read all 5 PDF documents thoroughly to understand the content.  
2. Selected 5 diverse questions covering different topics and varying levels of complexity.  
3. Used Claude AI to generate comprehensive expected answers by:
   - Feeding the relevant PDF sections to Claude  
   - Requesting detailed and accurate responses  
   - Validating the AI-generated answers against the source documents  
4. Manually identified all relevant document chunks (with IDs) that should be retrieved for each question.  
5. Created ground truth mappings for evaluation purposes.  

**Why this approach:**  

- Simulates a domain expert review process.  
- Ensures high-quality, thorough expected answers.  
- More scalable than writing all answers manually.  
- Provides consistent answer quality across all questions.  

**Limitations:**  

- Potential bias from using a single AI model for answer generation.  
- Not suitable for large-scale evaluation (currently limited to 5 questions; adding more requires updating `qwen_model.py`).  
- Requires careful validation against the source documents to maintain accuracy.

## 2) Semantic Similarity Score

**Definition:**  
The similarity score is calculated as:  
Similarity = cosine_similarity(embed(generated_answer), embed(expected_answer))

**What it measures:**  
This score indicates how semantically close the generated answer is to the expected "gold standard" answer in vector space. In other words, it captures whether the meaning of the generated response aligns with the reference, even if the wording differs.  

**Why it matters:**  

- Provides a measure of answer quality and completeness.  
- Captures meaning similarity beyond exact word matches.  
- More robust than traditional text-based metrics like BLEU or ROUGE, which rely heavily on surface-level text overlap. 

## Analysis

- **Consistently high scores:** All questions scored above the 0.60 threshold, indicating good answer quality.  
- **Best performer:** The "Claryo" question achieved a score of 0.83, showing that the system performs particularly well on specific, focused topics.  

## 5) What works well:

## Key Findings

### High-Quality Answer Generation
- Achieved an average semantic similarity of 60%, indicating that generated answers closely align with expert responses.  
- Llama-3.1-8B effectively leverages the retrieved context for accurate answers.  
- Thoughtful prompt engineering helps prevent hallucinations and ensures reliable outputs.  

### Reliable Source Attribution
- All answers correctly cite the source documents and corresponding chunk IDs.  
- Citations allow for easy verification and help build trust in the system’s outputs.  

### Consistent Cross-Domain Performance
- The system handles a wide range of topics, including HR, technical architecture, real estate, manufacturing, and healthcare.  
- No significant performance drop is observed across different document types.  
- Demonstrates the robustness of the RAG (Retrieval-Augmented Generation) approach.  

### Fast Response Times
- Queries are processed in 5-8 seconds, compared to 7–10 minutes with a local Ollama setup.  
- API-based architecture supports scalable deployment.  
- Provides a smooth and responsive user experience within the Streamlit interface.  


## 6 & 7) Known limitations/issues   AND     Add/remove/change with more time

## Limitations & Observations

### Manual Evaluation Process
- **Issue:** Creating ground truth answers currently requires manual effort using ChatGPT or another AI, and each question must be manually entered into the code for evaluation.  
- **Improvement:** Implement an automated evaluation pipeline to streamline this process (see Future Improvements).  

### Context Window Optimization
- **Issue:** Using a fixed `k=5` for retrieval may not be optimal for all types of questions.  
- **Observation:** Simple questions may require fewer chunks (e.g., `k=3`), while complex questions may benefit from more context (e.g., `k=8`).  
- **Improvement:** Implement adaptive retrieval strategies that adjust `k` based on query complexity.  







