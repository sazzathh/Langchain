from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(
	model="gemini-embedding-2"
)

vector=embedding.embed_query("what is the capital of Bangladesh")

#print(vector)
print(len(vector),vector[:5])