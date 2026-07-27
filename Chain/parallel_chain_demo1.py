from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

prompt1=PromptTemplate(
	template="""
		summarize the following text in 3 lines: \n {text}
	""",
	input_variables=["text"]
)
prompt2=PromptTemplate(
	template="""
		write 3 simple questions from the following text: \n {text}
	""",
	input_variables=["text"]
)

parallel_chain=RunnableParallel(
	{
		"summary":prompt1|model|parser,
		"questions":prompt2|model|parser
	}
)

text="""
	Deep learning is a subset of machine learning that utilizes multilayered artificial neural networks to automatically learn complex patterns and representations from large amounts of data. Inspired by the biological structure of the human brain, it eliminates the need for manual feature engineering by passing raw data through successive, interconnected "hidden" layers. Each successive layer extracts increasingly abstract and high-level features—for instance, passing from simple edges to full faces in image recognition tasks.
"""

result=parallel_chain.invoke({"text":text})

print("summary:\n",result["summary"])
print("\nquestions:\n",result["questions"])

#print(result)

parallel_chain.get_graph().print_ascii()