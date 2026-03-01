from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

#__________Tool Implementations__________

@tool
def get_price(item: str) -> float:
    """
    Returns the price of the item or product in the catalog
    input-args:
        - item : str = item to search the price for
    output : float = the value of the item from inventory 
    """
    print(f"tool: def get_price(item: {item}) -> float")

    inventory = {"laptop": 1299.55, "phone": 479.69, "earphones": 100.00}
    return inventory.get(item, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final discounted price.
    Available discount_tiers are gold, silver, bronze.
    input_args:
    - price:float = float number representing the price of an item
    - discount_tier: str = discount type can be one of [gold, silver, bronze]
    output: float = total discounted price after calculation
    """
    print(f"tool: def apply_discount(price:{price}, discount_tier:{discount_tier}) -> float")
    discount_percentage = {"gold": 40, "silver": 20, "bronze": 5}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

model = "qwen3:1.7b"
# model = "gpt-oss:20b"
MAX_ITERATIONS = 10

@traceable(name="LangChain Agent Loop")
def run_agent(query:str):
    # implement tool object retrival
    tools = [get_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # implement chat_model for ollama:qwen3:1.7b
    llm = init_chat_model(f"ollama:{model}", temperature=0)
    # implement bind_tools()
    llm_with_tools = llm.bind_tools(tools)

    print(f"Users Query: {query}")
    print("="*60)

    # implement messages array
    messages = [
        SystemMessage(
            content=(
                "You are a shopping assistant with a multi-step workflow.\n"
                "To answer the user, you must follow this sequence exactly:\n"
                "1. THOUGHT: What do I need to do? (e.g., 'I need the base price first').\n"
                "2. ACTION: Call 'get_price'.\n"
                "3. OBSERVATION: Review the price returned.\n"
                "4. THOUGHT: Do I need a discount? (e.g., 'The user asked for a silver discount, so I must now use apply_discount').\n"
                "5. ACTION: Call 'apply_discount'.\n"
                "6. FINAL ANSWER: Provide the final calculated price.\n\n"
                "IMPORTANT: Never skip a tool if the user requested both a price and a discount. If the user has requested an item in plural (e.g.: 'laptops') consider singluar 'laptop' and process the request."
            )
        ),
        HumanMessage(content=query)
    ]

    # implement agent loop with for loop
    for i in range(1, MAX_ITERATIONS+1):
        print(f"_________Iteration: {i}________")

        ai_message = llm_with_tools.invoke(messages)

        tool_call = ai_message.tool_calls

        if not tool_call:
            print(f"Final output: {ai_message}")
            return ai_message
        
        tool_call = tool_call[0]  # as in normal cases LLMs call multiple tools at once.
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)   # got the tool object
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)   # This is where you will execute the tool requested by LLM

        print(f"    [Tool Result] : {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without final answer!")


if __name__ == "__main__":
    print("Hello from ReAct Agent!")
    run_agent("I'm interested in knowing the price of laptops that you have!")