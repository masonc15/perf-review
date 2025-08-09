"""
OpenAI API utilities for the performance review system.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI


class OpenAIClient:
    """Wrapper for OpenAI responses API with error handling and utilities."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for responses
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def create_response(
        self,
        input_messages: List[Dict[str, Any]],
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a response with retry logic.
        
        Args:
            input_messages: Messages to send to the model
            instructions: System instructions
            tools: Available tools for function calling
            tool_choice: Tool choice strategy
            reasoning: Reasoning configuration (e.g., {"effort": "high"})
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
            
        Returns:
            Response from the model
        """
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "input": input_messages
                }
                
                if instructions:
                    kwargs["instructions"] = instructions
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
                if reasoning:
                    kwargs["reasoning"] = reasoning
                
                response = self.client.responses.create(**kwargs)
                return response
                
            except Exception as e:
                if attempt == max_retries:
                    raise e
                time.sleep(retry_delay * (2 ** attempt))
        
        raise Exception("Max retries exceeded")
    
    def extract_text_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from response.
        
        Args:
            response: OpenAI response object
            
        Returns:
            Text content from the response
        """
        return response.output_text or ""
    
    def extract_function_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract function calls from response.
        
        Args:
            response: OpenAI response object
            
        Returns:
            List of function calls
        """
        function_calls = []
        
        for item in response.output:
            if item.type == "function_call":
                function_calls.append({
                    "id": item.id,
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": json.loads(item.arguments)
                })
        
        return function_calls
    
    def create_function_call_response(
        self,
        input_messages: List[Dict[str, Any]],
        function_calls: List[Dict[str, Any]],
        function_executor: Callable[[str, Dict[str, Any]], Any],
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute function calls and get final response.
        
        Args:
            input_messages: Original input messages
            function_calls: Function calls to execute
            function_executor: Function to execute calls
            instructions: System instructions
            tools: Available tools
            
        Returns:
            Final response after function execution
        """
        # Execute function calls and add results to input
        extended_input = input_messages.copy()
        
        for call in function_calls:
            # Execute the function
            result = function_executor(call["name"], call["arguments"])
            
            # Add function result to input
            extended_input.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result) if not isinstance(result, str) else result
            })
        
        # Get final response
        return self.create_response(
            extended_input,
            instructions=instructions,
            tools=tools
        )
    
    def create_structured_prompt(
        self,
        system_message: str,
        user_message: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create structured input messages.
        
        Args:
            system_message: System instructions
            user_message: User query
            examples: Optional few-shot examples
            
        Returns:
            Formatted input messages
        """
        messages = []
        
        # Add system message
        if system_message:
            messages.append({
                "role": "system", 
                "content": system_message
            })
        
        # Add examples
        if examples:
            for example in examples:
                messages.extend([
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ])
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (4 chars per token approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        return len(text) // 4