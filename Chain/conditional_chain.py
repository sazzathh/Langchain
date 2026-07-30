from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

positive_prompt=ChatPromptTemplate.from_template(
	"Reply to this positive moveie review with a friendly way: \n {review}"
)
negative_prompt=ChatPromptTemplate.from_template(
	"Reply to this negative review by apologizing and offering help: \n {review}"
)

positive_chain=positive_prompt|model|parser
negative_chain=negative_prompt|model|parser

conditional_chain=RunnableBranch(
	(
		lambda x:"good" in x["review"].lower(),positive_chain
	),
	negative_chain
)

result1=conditional_chain.invoke({"review":"the movie was really good and i enjoyed every moment of it"})

result2=conditional_chain.invoke({"review":"the movie was not good and it was also too long"})

print(result2)