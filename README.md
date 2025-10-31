# RAG
## 1) Setup and installation instructions

**Clone the Repository:**
git clone https://github.com/alekhyareddy20/RAG.git
cd RAG
**Create Virtual Environment:**
python -m venv venv
MAC: source venv/bin/activate  #   Windows: venv\Scripts\activate
**Install Dependencies:**
pip install -r requirements.txt
**Set Up Hugging Face API Key:**
[I didn't include a token in the code; rather, I used Hugging Face CLI]
Create a token in Hugging Face 
Use the command below to log in:
huggingface-cli login
- Then enter your token
**Create or update the Chroma DB:**
python DB_model.py
OR
python DB_model.py --reset [This is to use to clear the existing database before creating a new one]
**Web Interface:**
streamlit run streamlitapp.py
- This will navigate to your browser

## 2) A brief description of  approach and design decisions
## 3) Technologies  previously familiar vs. new 
## 4) Evaluation results and interpretation
## 5) Known limitations/issues
## 6) Add/remove/change with more time





