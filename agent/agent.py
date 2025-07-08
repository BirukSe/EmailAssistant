from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START, MessagesState
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.types import Command
from interrupt_handler import triage_interrupt_handler
from prompt import triage_system_prompt, default_triage_instructions, triage_user_prompt, AGENT_TOOLS_PROMPT, \
    default_cal_preferences, default_response_preferences, agent_system_prompt
from utils import parse_email, format_email_markdown

load_dotenv()
model_name="gemini-flash-1.5"
llm=ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

class State(MessagesState):
    """We can add specific key for our state for the email input"""
    email_input: dict
    classification_decision: Literal["respond", "ignore", "notify"]
class RouterSchema(BaseModel):
    reasoning: str=Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["respond", "ignore", "notify"]=Field(description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply")
llm_router=llm.with_structured_output(RouterSchema)
def triate_router(state: State)->Command[Literal["response_agent", "__end__"]]:
    """Analyze the email to decide weather we should respond, ignore or notify the user"""
    author,to,subject,email_thread=parse_email(state["email_input"])
    system_prompt=triage_system_prompt.format(background="I am Biruk a Software Engineer from Ethiopia",triage_instructions=default_triage_instructions)
    user_prompt=triage_user_prompt.format(author=author, to=to,email_thread=email_thread,subject=subject)
    result=llm_router.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    if result.classification=="respond":
        goto="response_agent"
        update={
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: \n\n{format_email_markdown(subject, author, to, email_thread)}",
                }
            ],
            "classification_decision": result.classification,
        }
    elif result.classification=="ignore":
        goto=END
        update={
            "classification_decision": result.classification
        }
    elif result.classification=="notify":
        goto="triage_notifier"
        update={
            "classification_decision": result.classification
        }
    else:
        raise ValueError("Error occured")
    return Command(goto=goto, update=update)
tools=[]
llm_with_tools=llm.bind_tools(tools=tools)
def llm_call(state: State):
    """Decides weather to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(    [
                    {"role": "system", "content": agent_system_prompt.format(
                        tools_prompt=AGENT_TOOLS_PROMPT,
                        background="I am Biruk a Software Engineer from Ethiopia",
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences,
                    )}
                ]
                # Add the current messages to the prompt
                + state["messages"])

        ]
    }
tools_by_name={tool.name: tool for tool in tools}
# def tool_handler(state: State):
#     """Performs the tool call"""
#     result=[]
#     for tool_call in state["messages"][-1].tool_calls:
#         tool=tools_by_name(tool_call["name"])
#         observation=tool.invoke(tool_call["args"])
#         result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
#     return {"messages": result}
def should_continue(state: State):
    """Conditional edge wheather end or make a tool call"""
    messages=state["messages"]
    last_message=messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"]=="Done":
                return END
            else:
                return "interrupt_handler"
agent_builder=StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"interrupt_handler", "__end__"})
response_agent = agent_builder.compile()
overall_workflow=(
    StateGraph(State)
    .add_node(triate_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")

)
email_assistant=overall_workflow.compile()









