import os
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT_TEMPLATE = """You are an AI assistant that answers questions based only on the provided document context.

Instructions:
- Use only the information contained in the provided context to answer the user's question.
- If the answer cannot be found in the context, clearly state that the information is not available in the provided documents.
- Do not make up facts or use outside knowledge.
- Provide accurate, concise, and well-structured answers.
- If the context contains information from multiple documents, combine the relevant information where appropriate.
- Preserve important names, numbers, dates, and technical terms exactly as they appear in the documents.
- If the user's question is ambiguous, explain what additional information is needed.
- When possible, cite the relevant document name or section if it is available in the context.
- Respond in the same language as the user's question unless instructed otherwise. If the answer is not in the context,
say "I couldn't find this information in the uploaded documents." 
(or in Nepali: "यो जानकारी अपलोड गरिएका कागजातहरूमा फेला परेन।")

Language instruction: __LANG_INSTRUCTION__

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs):
    """
    combine retrived chunks into one text block for text block for the prompt.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def build_qa_chain(vector_store, k=10, source_filter=None,language="auto"):
    """
    Build an LCEL chain: retrieve --> format -- > LLM --> parser
    Retruns the chain ANF the retriever (we need the retriever separately to get the source documnts for citations)
    """
    search_kwargs: dict = {"k": k}
    if source_filter and source_filter != "All documents":
        search_kwargs["filter"] = {"source": source_filter}

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    #build the language instruction text 
    if language == "English":
        language_instruction = "Answer in English, regardless of the question language."
    elif language == "Nepali":
        language_instruction = "Answer in Nepali, regardless of the question language."
    else:
        language_instruction = "Detect the question language and answer in the same language."        
    
    prompt_text = PROMPT_TEMPLATE.replace("__LANG_INSTRUCTION__", language_instruction)
    prompt = ChatPromptTemplate.from_template(prompt_text)

    groq_key = os.getenv("GROQ_API_KEY")
    groq_secret = SecretStr(groq_key) if groq_key is not None else None

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=groq_secret)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask_question(chain, retriever, question):
    """
    Runs the chain AND retrieves source documents separately for citations.
    Returns (answer, source_documents)
    """
    answer = chain.invoke(question)

    # Get source seperately - same retriever, same query
    source_docs = retriever.invoke(question)

    return answer, source_docs