import json
from groq import Groq
from src.components.tools import TOOLS, get_ip_address, search_web, get_college_info, query_uploaded_pdf
from src.components.pdf_handler import SessionPDFStore
from src.utils.config import GROQ_API_KEY, LLM_MODEL_RES, LLM_TEMP_RES, LLM_MAX_TOKENS_RES, LLM_MODEL_FIN, LLM_TEMP_FIN, LLM_MAX_TOKENS_FIN, SYSTEM_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger("agent")
groq_client = Groq(api_key=GROQ_API_KEY)



def process_with_agent(user_text: str, session_store: SessionPDFStore) -> str:
    """Process user text with LLM that can use tools"""

    try:
        logger.info("Agent processing...")
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}]
        
        response = groq_client.chat.completions.create(
                                messages=messages,
                                model=LLM_MODEL_RES,
                                tools=TOOLS,
                                tool_choice="auto",
                                temperature=LLM_TEMP_RES,
                                max_tokens=LLM_MAX_TOKENS_RES,
                                stream=False,
                                reasoning_format="hidden",)

        response_message = response.choices[0].message            # this object contains .content, .tool_calls 

        if response_message.tool_calls:         # if "tool_calls" present in the content of the response schema 

            logger.info(f"LLM wants to use {len(response_message.tool_calls)} tool(s)")

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Calling tool: {function_name}")

                # Execute tool
                if function_name == "get_ip_address":
                    function_response = get_ip_address()
                elif function_name == "search_web":
                    function_response = search_web(function_args.get("query", ""))
                elif function_name == "get_college_info":
                    function_response = get_college_info(function_args.get("question", ""))
                elif function_name == "query_uploaded_pdf":
                    function_response = query_uploaded_pdf(function_args.get("question", ""), session_store)
                else:
                    function_response = f"Unknown tool: {function_name}"

                # append tool results back to message 
                messages.append({"role": "tool",
                                 "tool_call_id": tool_call.id,
                                 "name": function_name,
                                 "content": function_response})

            logger.info("Getting final response from LLM...")

            final_response = groq_client.chat.completions.create(       # LLM sees original question, tool output and generates a response
                                messages=messages,
                                model=LLM_MODEL_FIN,
                                temperature=LLM_TEMP_FIN,
                                max_tokens=LLM_MAX_TOKENS_FIN,
                                tools=None,)

            agent_response = final_response.choices[0].message.content
        else:
            logger.info("No tools needed, responding directly")
            agent_response = response_message.content
        
        logger.info(f"✅ Response generated")
        return agent_response
    
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return "Sorry, I encountered an error processing your request."