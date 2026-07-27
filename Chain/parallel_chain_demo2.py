from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=ChatGroq(model="llama-3.3-70b-versatile")
model2=ChatGroq(model="qwen/qwen3.6-27b")

parser=StrOutputParser()

prompt1=PromptTemplate(
	template="""
		Generate a simple notes from the following text: \n {text}
	""",
	input_variables=["text"]
)
prompt2=PromptTemplate(
	template="""
		Geneate 5 questions from the following text: \n {text}
	""",
	input_variables=["text"]
)
prompt3=PromptTemplate(
	template="""
		merged the notes and quiz in a single document: \n notes->{notes} and quiz->{quiz}
	""",
	input_variables=["notes","quiz"]
)

parallel_chain=RunnableParallel(
	{
		"notes":prompt1|model1|parser,
		"quiz":prompt2|model2|parser
	}
)

#chain=prompt3|model1|parser
#merge_chain=parallel_chain|chain

merge_chain=parallel_chain|prompt3|model1|parser


text="""
	Deep learning is a subset of machine learning that utilizes multilayered artificial neural networks to automatically learn complex patterns and representations from large amounts of data. Inspired by the biological structure of the human brain, it eliminates the need for manual feature engineering by passing raw data through successive, interconnected "hidden" layers. Each successive layer extracts increasingly abstract and high-level features—for instance, passing from simple edges to full faces in image recognition tasks.
"""

result=merge_chain.invoke({"text":text})


print(result)

merge_chain.get_graph().print_ascii()