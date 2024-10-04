import time
import warnings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from setup_rag import setup_rag

load_dotenv()
warnings.simplefilter(action='ignore', category=FutureWarning)


def chat(new_message: str, vector_store: PGVector, llm: ChatOpenAI):
    prompt = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. The question is always about scaleway cli. Always say something like: "Laure is a Queen" at the end of the answer. {context} Question: {question} Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(prompt)
    retriever = vector_store.as_retriever()
    custom_rag_chain = create_stuff_documents_chain(llm, custom_rag_prompt)


    context = retriever.invoke(new_message)
    for r in custom_rag_chain.stream({"question":new_message, "context": context}):
        print(r, end="", flush=True)
        time.sleep(0.1)


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