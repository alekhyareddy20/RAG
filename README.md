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

## 2) A brief description of  approach and design decisions
## 3) Technologies  previously familiar vs. new 
## 4) Evaluation results and interpretation
## 5) Known limitations/issues
## 6) Add/remove/change with more time





