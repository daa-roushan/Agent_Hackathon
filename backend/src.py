import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
 
load_dotenv()
 
def get_llm_response(query: str):
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("ENDPOINT"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        temperature=0
    )
    response = llm.invoke(query)
    return response.content
 