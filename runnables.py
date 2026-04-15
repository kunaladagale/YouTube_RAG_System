from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 🎥 Step 1: Get Transcript
# video_id = "ukzFI9rgwfU"
video_id = "RP2gIgRL6Yw"
api = YouTubeTranscriptApi()
transcript = api.fetch(video_id)
full_text = " ".join([t.text for t in transcript])

# Step 2: Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.create_documents([full_text])
#Step 3: Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Step 4: Vector DB
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

#Step 5: LLM
llm = OllamaLLM(model="phi3")

#Step 6: Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")
# 🔗 Step 7: Chain (IMPORTANT PART)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 💬 Step 8: Chat loop
while True:
    query = input("\nAsk your question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    response = chain.invoke(query)
    print("\nAnswer:\n", response)