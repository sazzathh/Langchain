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

#print(prompt.invoke({"topic":"machine learning"}))
prompt_value=prompt.invoke({"topic":"machine learning"})


#step-1:send the prompt to the model

model_output=model.invoke(prompt_value)

#print(model_output)

#step-2:parse the response

parser=StrOutputParser()
final_output=parser.invoke(model_output)

print(final_output)