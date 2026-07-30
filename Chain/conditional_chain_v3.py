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
		-neutral
		-not determined
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

neutral_prompt=PromptTemplate(
	template="""
		reply to this neutral moview review with politeness and ask for suggestions
		review:{review}
	""",
	input_variables=["review"]
)
negative_prompt=PromptTemplate(
	template="""Reply to this negative review by apologizing and offering help: \n
	  {review}
	""",
	input_variables=["review"]	
)
default_prompt=PromptTemplate(
	template="""
		The sentiment could not determined
		review:{review}
	""",
	input_variables=["review"]
)

positive_chain=positive_prompt|model|parser
neutral_chain=neutral_prompt|model|parser
negative_chain=negative_prompt|model|parser
default_chain=default_prompt|model|parser

#review="the movie was fantastice and i love every minute of it"
review="the movie was a shit full of garbage"

#step-3:
sentiment=classifier_chain.invoke({"review":review})

print("prediction sentiment :",sentiment)

conditional_chain=RunnableBranch(
	(lambda x:x["sentiment"].strip().lower()=="positive",positive_chain),
	(lambda x:x["sentiment"].strip().lower()=="neutral",neutral_chain),
	(lambda x:x["sentiment"].strip().lower()=="negative",negative_chain),
	default_chain
)

result=conditional_chain.invoke({
	"review":review,
	"sentiment":sentiment
})


print(result)