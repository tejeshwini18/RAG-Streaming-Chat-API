"""
Streaming LLM integration using OpenAI or Ollama (local)
"""
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Dict, Optional
from openai import OpenAI
from config import settings


class StreamingLLM:
    """Handles streaming LLM responses (OpenAI or Ollama)."""

    def __init__(self):
        if settings.use_ollama:
            self.client = OpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",  # Ollama doesn't check; key required by client
            )
            self.model = settings.ollama_model
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
    
    def create_system_prompt(self, context: str = "") -> str:
        """Create system prompt with RAG context and current date so the model can answer all questions."""
        now = datetime.now(timezone.utc)
        date_line = f"Current date and time (UTC): {now.strftime('%A, %B %d, %Y')}. Time: {now.strftime('%H:%M')} UTC."
        base_prompt = """You are a helpful AI assistant. Your job is to answer the user's questions directly and helpfully.

{date_info}

When the user asks about the current date, time, or any general knowledge (facts, math, history, etc.), use your knowledge and the date above to answer immediately. Do NOT say you lack context or suggest they look elsewhere—just answer.

When document context is provided below, use it to improve your answer when it is relevant. If the context is not relevant to the question, ignore it and answer from your own knowledge.

Context from retrieved documents (use only if relevant to the question):
{context}

Instructions:
- Answer every question directly. For dates/times use the current date/time given above.
- When document context is relevant, use it and cite it.
- When document context is not relevant, answer from general knowledge without saying you lack context.
- Be helpful, accurate, and concise."""
        doc_context = context if context else "(No documents retrieved for this query. Answer from your knowledge.)"
        return base_prompt.format(date_info=date_line, context=doc_context)
    
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        context: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with RAG context
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Retrieved context from RAG pipeline
        
        Yields:
            Chunks of text as they are generated
        """
        # Prepare messages with system prompt containing context
        system_message = {
            "role": "system",
            "content": self.create_system_prompt(context)
        }
        
        formatted_messages = [system_message] + messages
        queue: asyncio.Queue = asyncio.Queue()

        model_to_use = model if model else self.model
        def stream_into_queue():
            try:
                stream = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=formatted_messages,
                    stream=True,
                    temperature=temperature,
                )
                for chunk in stream:
                    try:
                        text = None
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta:
                                text = getattr(delta, "content", None)
                        if text is None and getattr(chunk, "message", None):
                            text = getattr(chunk.message, "content", None)
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
                    except (IndexError, AttributeError, TypeError):
                        continue
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # poison: signal error
                raise
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(None, stream_into_queue)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await fut  # consume so any exception is raised
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate non-streaming response (for testing)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Retrieved context from RAG pipeline
        
        Returns:
            Complete response text
        """
        system_message = {
            "role": "system",
            "content": self.create_system_prompt(context)
        }
        
        formatted_messages = [system_message] + messages
        model_to_use = model if model else self.model
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=formatted_messages,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
