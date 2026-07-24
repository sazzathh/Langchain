from pydantic import BaseModel

class User(BaseModel):
	name:str
	age:int

user_data={
	"name":"john",
	"age":"32"
}

user=User(**user_data)

print(user)
#print(type(user))