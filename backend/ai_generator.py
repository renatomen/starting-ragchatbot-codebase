import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2  # Maximum sequential tool calling rounds per query

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search course materials for specific content or detailed educational materials
2. **get_course_outline**: Get the complete structure of a course including title, link, and all lessons with their numbers and titles

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure, outline, or syllabus
  - What lessons a course contains
  - Course overview or topics covered
  - When listing all lessons in a course
- Use **search_course_content** for questions about:
  - Specific content within lessons
  - Detailed explanations or concepts
  - Finding information across courses
- **Sequential tool use supported**: You may make up to 2 rounds of tool calls if needed
- Use multiple calls for complex queries requiring information from different courses/lessons
- After each tool result, determine if additional information is needed before answering
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline, then present the course title, course link, and complete lesson list (lesson number and title for each)
- **Course content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_tool_loop(response, api_params, tool_manager, tools)

        # Return direct response
        return self._extract_text_response(response)

    def _extract_text_response(self, response) -> str:
        """Extract text content from API response.

        Args:
            response: Anthropic API response object

        Returns:
            Text content or empty string if no text found
        """
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _execute_tool_loop(self, response, api_params: Dict[str, Any],
                           tool_manager, tools: List) -> str:
        """Execute tools in a loop, allowing up to MAX_TOOL_ROUNDS rounds.

        Args:
            response: Initial response containing tool_use blocks
            api_params: Base API parameters with messages and system prompt
            tool_manager: Manager to execute tools
            tools: Tool definitions for subsequent API calls

        Returns:
            Final text response after all tool rounds complete
        """
        messages = api_params["messages"].copy()
        current_response = response
        round_count = 0

        while round_count < self.MAX_TOOL_ROUNDS:
            round_count += 1

            # Add assistant's tool_use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls
            tool_results = []
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        # On error, return text from current response or error message
                        return self._extract_text_response(current_response) or f"Tool error: {e}"

            # Add tool results
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Build next API call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": api_params["system"]
            }

            # Include tools if rounds remaining
            if round_count < self.MAX_TOOL_ROUNDS:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**next_params)

            # Exit if Claude didn't request more tools
            if current_response.stop_reason != "tool_use":
                break

        return self._extract_text_response(current_response)