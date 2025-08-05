import os
from langchain_core.prompts import PromptTemplate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data/")
EMBED_DELAY = 0.02  # 20 milliseconds
EMBEDDING_MODEL_NAME = "text-embedding-004"
LLM_NAME   = "gemini-1.5-flash"
# LLM_NAME = 'models/gemini-1.5-pro'

prompt = PromptTemplate(
template = ''' You are a helpful AI assitant for an EduTech company involved in OMS application.
Answer only from the provided context. If greeted with hi, hey , hello etc. please respond like a polite chatbot.
If the context is insufficient just respond with - Sorry I cannot assist you with your query since I lack the required knowledge at the moment. Please let me know if I can assist you with something else.
Context : {context}
Question : {question}

'''
, input_variables=['context', 'question'])


