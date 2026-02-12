#it is working fine just that there are no free creditsleft in account

import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.responses.create(
    model="gpt-4o-mini",
    input="What is FastAPI?"
)

print(response.output_text)
