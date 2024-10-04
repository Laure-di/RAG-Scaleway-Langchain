import time
import warnings

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

from setup_rag import setup_rag

load_dotenv()
warnings.simplefilter(action='ignore', category=FutureWarning)

def chat(new_message: str, vector_store: PGVector, llm: ChatOpenAI):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
            {"context": vector_store, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    for r in rag_chain.stream(new_message):
        print(r, end="", flush=True)
        time.sleep(0.15)

def main():
    deployment, vector_store, llm = setup_rag()
    prompt = "\nHow can I help you?"
    prompt += "\nEnter 'quit' to end the program.\n"
    question = ""
    while question != 'quit':
        question = input(prompt)
        if question != 'quit':
            chat(question, vector_store, llm)

if __name__ == '__main__':
    main()