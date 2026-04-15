from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "This is a long text that you want to split into chunks for LLM processing..."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
for chunk in chunks:
    print(chunk)
    print("-----")