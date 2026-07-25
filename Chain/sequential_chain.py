from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt_1=PromptTemplate(
	template="""
		Evaluate the student's answer
		Question: {question}
		student's answer: {answer}
		provide a detailed evaluation including 
		-correctness
		-strengths
		-weaknesses
		-suggestions for improvement
	""",
	input_variables=["question","answer"]
)
prompt_2=PromptTemplate(
	template="""
		convert the following detailed evaluation into concise 5-point feedback
		{evalution}
	""",
	input_variables=["evaluation"]
)

parser=StrOutputParser()


chain=prompt_1|model|parser|prompt_2|model|parser

result=chain.invoke({
	"question":"what is machine learning",
	"answer":"""Machine learning is a subset of artificial intelligence where computers learn patterns from data and make predictions without being explicitly programmed for every step"""
})

print(result)

chain.get_graph().print_ascii()

