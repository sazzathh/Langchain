from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

#define the model
model=ChatGroq(model="llama-3.3-70b-versatile")

#define the schema
class ModelEvaluation(BaseModel):
	model_name:str=Field(description="name of the machine learning model")

	accuracy:float=Field(gt=0,lt=1,description="accuracy of the model,less than 1,greater than 0")

	dataset:str=Field(description="dataset used for model evaluation")

#define the parser
parser=PydanticOutputParser(pydantic_object=ModelEvaluation)

#promptTemplate
template=PromptTemplate(
	template="""
	Generate the name,accuracy,dataset used for fictional machine learning model for {task}
	{format_instruction}
	""",
	input_variables=["task"],
	partial_variables={
		"format_instruction":parser.get_format_instructions
	}
)

#define the chain
chain=template|model|parser

#invoke the chain
result=chain.invoke({"task":"image_classification"})

print(result)
