from typing import TypedDict

class ProductReview(TypedDict):
	product_name:str
	rating:int
	review:str

new_review:ProductReview={
	"product_name":"earbuds",
	"rating":"4.5",
	"review":"pretty good"
}

print(new_review)