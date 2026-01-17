from langchain_openai import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from tools.sql import run_query_tool, list_tables, describe_tables_tool

tables = list_tables()
chat = ChatOpenAI()
prompt = ChatPromptTemplate(
  messages=[
    SystemMessage(
      content=f"You are an AI that has access to a SQLite database.\n{tables}"
    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
  ]
)
tools = [
  run_query_tool,
  describe_tables_tool,
]

agent = OpenAIFunctionsAgent(
  llm=chat,
  prompt=prompt,
  tools=tools,
)

agent_executor = AgentExecutor(
  agent=agent,
  verbose=True,
  tools=tools,
)

agent_executor("How many users have provided a shipping address?")
# agent_executor("How many users are in the database?")
