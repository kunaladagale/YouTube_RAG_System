from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="phi3")
print(llm.invoke("What is AI?"))