from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

#step-1:classify the review
classifier=PromptTemplate(
	template=""" 
		you are a moview review classifier.classify the following review
		and return only one word 
		-positive
		-negative
		review:{review}
	""",
	input_variables=["reivew"]
)
classifier_chain=classifier|model|parser

#step-2:

positive_prompt=PromptTemplate(
	template="""Reply to this positive moveie review with a friendly way: \n
		{review}
	""",
	input_variables=["review"]
)
negative_prompt=PromptTemplate(
	template="""Reply to this negative review by apologizing and offering help: \n
	  {review}
	""",
	input_variables=["review"]	
)

positive_chain=positive_prompt|model|parser
negative_chain=negative_prompt|model|parser

review="the movie was fantastice and i love every minute of it"

#step-3:
sentiment=classifier_chain.invoke({"review":review})

print("prediction sentiment :",sentiment)

conditional_chain=RunnableBranch(
	(
		lambda x:x["sentiment"].strip().lower()=="positive",positive_chain
	),
	negative_chain
)

result=conditional_chain.invoke({
	"review":review,
	"sentiment":sentiment
})


print(result)