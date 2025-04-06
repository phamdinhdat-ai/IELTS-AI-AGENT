from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


from utils.hf_helper import load_model


human_message = HumanMessage(
    content="Hello, how are you?",
)
llm = load_model(model_hf_path=HF_REPO_ID, hf_api_token=HF_TOKEN)

resutls = llm.invoke([human_message])
print(resutls)