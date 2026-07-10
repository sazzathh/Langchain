from sentence_transformers import SentenceTransformer,util

model=SentenceTransformer("all-MiniLM-L6-v2")

s1="I love programming in python"
s2="python coding is my passion"

s3="I love eating pizza"

emb1=model.encode(s1)
emb2=model.encode(s2)
emb3=model.encode(s3)

#print(emb1)

#cosine similarity
print(util.cos_sim(emb1,emb2))

print(util.cos_sim(emb1,emb3))