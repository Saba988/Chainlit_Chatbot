import chainlit as cl
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig
from dotenv import load_dotenv,find_dotenv
import os

load_dotenv(find_dotenv())
gemini_apikey=os.getenv("GEMINI_API_KEY")

#Step # 01: Provider

provider = AsyncOpenAI(
    api_key=gemini_apikey, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#Step # 02: Model

model=OpenAIChatCompletionsModel(
    model= "gemini-1.5-flash",
    openai_client= provider,
)


# Config : Define at run level

run_config=RunConfig(
    model= model,
    model_provider= provider,
    tracing_disabled= True
)

#Step 03 : Agents

agent1=Agent(
    name="Practice Agent",
    instructions= "You are helpful assistant that you can answer any question  or any tasks."
)

# #Step 04 : Run
# result=Runner.run_sync(
#     agent1,
#     input="What is the capital of france",
#     run_config= run_config,
    
# )

# print(result)

# @cl.on_message
# async def handle_message(message:cl.Message):
#     result= await Runner.run(
#         agent1,
#         input= message.content,
#         run_config= run_config,    
#     )
#     await cl.Message(content=result.final_output).send()

    

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="👋 Hello! I'm your Smart Virtual Assistant. How can I help you today?").send()

@cl.on_message
async def handle_message(message:cl.Message):
    history= cl.user_session.get("history") 

    history.append({"role":"user","content":message.content})
    result=await Runner.run(
        agent1,
        input= history,
        run_config= run_config,
    )

    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history",history)
    await cl.Message(content=result.final_output).send()



