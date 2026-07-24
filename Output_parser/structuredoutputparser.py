from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
	StructuredOutputParser,ResponseSchema
)

load_dotenv()

#define the model
model=ChatGroq(model="llama-3.3-70b-versatile")

#define the schema
response_schema=[
	ResponseSchema(
		name="fact_1",
		description="the first fact about the topic"
	),
	ResponseSchema(
			name="fact_2",
			description="the second fact about the topic"
		),
	ResponseSchema(
			name="fact_3",
			description="the third fact about the topic"
	),
	ResponseSchema(
			name="fact_5",
			description="the fourth fact about the topic"
	),
	ResponseSchema(
			name="fact_5",
			description="the fifth fact about the topic"
	)
]

#define the parser
parser=StructuredOutputParser.from_response_schemas(response_schema)

#define the template
template=PromptTemplate(
	template="""
	Give me 5 facts about {topic}.
	{format_instruction}
	""",
	input_variables=["topic"],
	partial_variables={
		"format_instruction":parser.get_format_instructions()
	}
)

#define the chain
chain=template|model|parser

#invoke the chain
result=chain.invoke({"topic":"Machine Learning"})

print(result)

#print(result["fact_1"])
