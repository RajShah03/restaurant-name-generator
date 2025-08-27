from langchain.prompts import PromptTemplate
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Initialize LLM
llm = OllamaLLM(model="mistral")

# --------------------
# Define format schemas
# --------------------
name_schema = ResponseSchema(
    name="restaurant_name", description="A fancy restaurant name"
)
menu_schema = ResponseSchema(
    name="menu_items", description="List of 8-10 menu items as array of strings"
)

name_parser = StructuredOutputParser.from_response_schemas([name_schema])
menu_parser = StructuredOutputParser.from_response_schemas([menu_schema])

name_format = name_parser.get_format_instructions()
menu_format = menu_parser.get_format_instructions()

# --------------------
# Build generation function
# --------------------
def generate_restaurant_names_and_items(cuisine: str):
    print("⚡ Cuisine input:", cuisine)

    # ---- Step 1: Restaurant Name ----
    prompt_name = PromptTemplate(
        input_variables=["cuisine"],
        template="Suggest a fancy restaurant name for {cuisine} cuisine.\n{format_instructions}",
        partial_variables={"format_instructions": name_format}, 
    )
    
    name_chain = prompt_name | llm | name_parser | RunnableLambda(lambda x: {"restaurant_name": x["restaurant_name"]})
    name_result = name_chain.invoke({"cuisine": cuisine})
    print("✅ Name Result:", name_result)

    # ---- Step 2: Menu Items ----
    prompt_menu = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest 8-10 creative menu items for a restaurant named {restaurant_name}.\n{format_instructions}",
        partial_variables={"format_instructions": menu_format}, 
    )
    
    menu_chain = prompt_menu | llm | menu_parser | RunnableLambda(lambda x: {"menu_items": x["menu_items"]})
    menu_result = menu_chain.invoke({"restaurant_name": name_result["restaurant_name"]})
    print("✅ Menu Result:", menu_result)

    # ---- Final output ----
    return {
        "restaurant_name": name_result["restaurant_name"],
        "menu_items": menu_result["menu_items"]
    }
