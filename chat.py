from langchain_core.messages import HumanMessage, AIMessage
import datetime
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
import socket
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
import pickle
from huggingface_hub import hf_hub_download


wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
GROQ_API_KEY="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"

dt = datetime.datetime.now()
formatted = dt.strftime("%A, %B %d, %Y %I:%M:%S %p")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 4000)
llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)

try:
    with open("data_chat.pkl", 'rb') as fp:
        chat_history = pickle.load(fp)
        # print(chat_history)
except:
    chat_history = []

try: 
    with open("template.pkl", 'rb') as file:
        template_abox = pickle.load(file)
except:
    hf_hub_download(repo_id="linl03/dataAboxChat",local_dir="./", filename="template.pkl", repo_type="dataset")
    # snapshot_download(repo_id="linl03/Chat_", local_dir="./", repo_type="space", allow_patterns="*.pkl" , local_dir_use_symlinks=False)
    with open("./template.pkl", 'rb') as file:
        template_abox = pickle.load(file)
        
router_prompt = PromptTemplate(
    template=template_abox["router_template"],
    input_variables=["question"],
)
generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template_abox["system_prompt"],
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
query_prompt = PromptTemplate(
    template=template_abox["query_template"],
    input_variables=["question"],
)

remind_prompt = PromptTemplate(
    template=template_abox["schedule_template"],
    input_variables=["time"],
)
class State(TypedDict):

    question : str
    generation : str
    search_query : str
    context : str

# Node - Generate
question_router = router_prompt | llm | JsonOutputParser()
generate_chain = generate_prompt | llm | StrOutputParser()
query_chain = query_prompt | llm | JsonOutputParser()
remind_chain = remind_prompt | llm | StrOutputParser()

def Agent():
    workflow = StateGraph(State)
    workflow.add_node("websearch", web_search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    # Build the edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "websearch")
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", END)

    # Compile the workflow
    local_agent = workflow.compile()
    return local_agent

def transform_query(state):
    print("Step: Tối ưu câu hỏi của người dùng")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    print(gen_query)
    search_query = gen_query["query"]
    return {"search_query": search_query}

def web_search(state):
    search_query = state['search_query']
    print(f'Step: Đang tìm kiếm web cho: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)
    print("Search result:", search_result)
    return {"context": search_result}

def route_question(state):
    print("Step: Routing Query")
    question = state['question']
    output = question_router.invoke({"question": question})
    print('Lựa chọn của AI là: ', output)
    if output['choice'] == "web_search":
        # print("Step: Routing Query to Web Search")
        return "websearch"
    elif output['choice'] == 'generate':
        # print("Step: Routing Query to Generation")
        return "generate"
def generate(state):    
    print("Step: Đang tạo câu trả lời từ những gì tìm được")
    question = state["question"]
    context = state["context"]
    return {'question': question, 'context': context}

def plan_in_day():
    for chunk in remind_chain.stream({"time": formatted}):
        print(chunk, end="", flush=True)
        yield chunk     

def generate_response(prompt):
    session_state = ''
    local_agent = Agent()
    output = local_agent.invoke({"question": prompt})
    context = output['context']
    questions = output['question']

    for chunk in generate_chain.stream({"context": context, "question": questions, "chat_history": chat_history}):
        print(chunk, end="", flush=True)
        session_state += chunk
        sock.sendto(str.encode(session_state), serverAddressPort)

    chat_history.append(HumanMessage(content=questions))
    chat_history.append(AIMessage(content=session_state))
    with open('data_chat.pkl', 'wb') as fp:
        pickle.dump(chat_history, fp)

def udp_server():
    server_address = ('localhost', 3030)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)
    print("Python UDP server is listening on {}:{}".format(*server_address))
    while True:
        data, address = sock.recvfrom(4096)
        if len(data.decode())>0:
            generate_response(data.decode())
        print("Received {} bytes from {}: {}".format(len(data), address, data.decode()))

if __name__ == "__main__":
    udp_server()
