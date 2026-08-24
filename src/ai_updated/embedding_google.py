from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    dimensions=32
    #google_api_key=os.getenv("GOOGLE_API_KEY")
)

result = embedding.embed_query("I am a very good person")
print(str(result))
