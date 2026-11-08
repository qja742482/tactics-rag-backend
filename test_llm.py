from langchain_google_genai import ChatGoogleGenerativeAI
import config

print("키 있음:", bool(config.GEMINI_API_KEY), "/ 모델:", config.GEMINI_MODEL)

llm = ChatGoogleGenerativeAI(
    model=config.GEMINI_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    timeout=90,
    max_retries=3,
)
print(llm.invoke("4-3-3 포메이션을 한 문장으로 설명해줘").text)