from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal,Optional

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

#create the schema
class ResumeAnalyzer(TypedDict):
    
    key_skills:Annotated[list[str],"extract the important technical and soft skill from the resume"]

    summary:Annotated[str,"write a brief summary of candidates profile"]

    experience_level:Annotated[Literal["Entry Level","Mid Level","High Level"],"Classify the candidate experience level"]

    strengths:Annotated[Optional[list[str]],"List the candidate major strengths "]

    weaknesses:Annotated[Optional[list[str]],"List the possible weaknesses or areas to improve"]

    candidate_name:Annotated[str,"Extract the name of the candidate from the resume"]

#create the structured output
structured_ouptut=model.with_structured_output(ResumeAnalyzer)

result=structured_ouptut.invoke("""My name is  Sazzath Hossen. I am a Computer Science graduate with two years
    of experience working as a Machine Learning Engineer.

    I have experience with Python, PyTorch, TensorFlow, Scikit-learn,
    FastAPI, Docker, Kubernetes, MLflow, and AWS. I have built machine
    learning pipelines, deployed deep learning models, and developed
    REST APIs for AI applications.

    I also have experience working with LangChain and Retrieval-Augmented
    Generation (RAG) systems.

    My main strength is my ability to build complete machine learning
    systems from data preprocessing to deployment. However, I have limited
    experience managing large engineering teams and need to improve my
    system design skills.

    Education:
    BSc in Computer Science and Engineering.
""")
print(result)
#print(result["candidate_name"])
#print(type(result))