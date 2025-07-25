from config import Config

llm = Config.OPENAI_MODEL
try:
    response = llm.invoke("Test prompt: What is 2+2?")
    print("LLM Response:", response)  # Should print "4" or similar
except Exception as e:
    print("LLM Error:", str(e))
