from langchain_groq import ChatGroq
from pydantic import BaseModel,EmailStr,Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class JobApplication(BaseModel):
	candidate_name:str="unknown"

	experience_level:Optional[int]=None

	email:EmailStr

	expected_salary:int=Field(gt=0,description="expected salary of the the candidate")


#create the model
model=ChatGroq(model="llama-3.3-70b-versatile")

#create the structured model
structured_model=model.with_structured_output(JobApplication)

#invoke the model
result=structured_model.invoke("""My name is Arif Hasan. I have 3 years of experience
    as a Machine Learning Engineer.

    My email is arif@gmail.com.
    I am expecting an annual salary of 50000 dollars.
	""")
print(result.expected_salary)