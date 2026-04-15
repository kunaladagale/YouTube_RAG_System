from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# 🎥 Step 1: Get YouTube Transcript
video_id = "ukzFI9rgwfU"
api = YouTubeTranscriptApi()
transcript = api.fetch(video_id)
# Convert transcript to full text
full_text = " ".join([t.text for t in transcript])
# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       #
    chunk_overlap=100
)
docs = splitter.create_documents([full_text])
print("Total chunks:", len(docs))
#  Step 3: Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#  Step 4: Store in FAISS
db = FAISS.from_documents(docs, embeddings)
# Step 5: Create Retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# Step 6: Load LLM (Ollama)
llm = OllamaLLM(model="phi3")
#
# Step 7: Chat Loop
while True:
    query = input("\nAsk your question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    #Step 8: Retrieve relevant docs
    relevant_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

#    Step 9: Prompt
    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".
Context:
{context}
Question:
{query}
Answer:
"""
    # 🤖 Step 10: Get response
    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)