from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt=PromptTemplate(
	template="""
		you are an ai tutor.
		explain the topic in a begineer friendly way to the students.
		topic {topic}
	""",
	input_variables=["topic"]
)

parser=StrOutputParser()


chain=prompt|model|parser

result=chain.invoke({"topic":"machine_learning"})

print(result)

chain.get_graph().print_ascii()

