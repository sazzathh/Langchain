from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document=[
	"I love AI",
	"I love ml",
	"I love dl"
]
vector=embedding.embed_documents(document)
print(len(vector))