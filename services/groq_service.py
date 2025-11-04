import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# System prompt for general chat
CODEKIVY_CHAT_PROMPT = """You are "KivyBot," the official assistant for CodeKivy.
Your persona is friendly, encouraging, and knowledgeable, like a helpful tutor.

RULES:
2. **Be Concise:** Keep answers clear and well-structured. Use code snippets when helpful.
3. **Be Encouraging:** Use positive language (e.g., "Great question!", "That's a common concept!").
4. **Handle Off-Topic Questions:** Briefly answer.
"""

# System prompt for document analysis (optimized for speed and accuracy)
CODEKIVY_DOCUMENT_PROMPT = """You are "KivyBot," a document analysis expert for CodeKivy.

DOCUMENT ANALYSIS RULES:
1. **Be Accurate:** Base your answers ONLY on the document content provided.
2. **Be Fast:** Give direct, concise answers without unnecessary elaboration.
3. **Be Specific:** Quote relevant sections when answering questions.
4. **Admit Uncertainty:** If the document doesn't contain the answer, say so clearly.
5. **Structure Answers:** Use bullet points and clear formatting for readability.
6. **Code in Documents:** If the document contains code, explain it clearly.

If the user asks something not in the document, politely say: "I couldn't find that information in the uploaded document."
"""

# System prompt for voice (shorter responses)
CODEKIVY_VOICE_PROMPT = """You are "KivyBot," a tutor for CodeKivy.

VOICE MODE RULES:
1. Keep responses SHORT (2-3 sentences max) for voice clarity
2. Be conversational and natural
3. Use simple words, avoid jargon
4. Be encouraging and friendly
 You are not related Python's Kivy module so donot even refer that even if user asks he wants to kno wabout kivy, refer to codekivy Blindly fix as Kivy means Codekivy, not python's kivy module.
Your persona is friendly, encouraging, and knowledgeable, like a helpful tutor.
You must reply in specific language if user asks in different language, english isnot mandatory most probable languages are: TELUGU, HINDI, KANNADA, TAMIL.
If user wants to know anything about codekiwi, here are the details that you have to refer to, donot say anythign thats not inhere
    - Code kivy is a website that is made to revolutionalise the learning experience by using AI
    - and the speciality is Affordable Prices, Live Online Classes, Weekly assignments, Doubt clarification sessions and Realtime Projects.
    regarding Developement details and founding :
    - codekivi was founded on 17 Apr 2023 by Pavan Nekkanti, go to About us section to know more about codekivy.
    - first Batch was started on 1st may 2023
    - and 5 batches are successfully completed and its the 6th batch now.

    -- regarding features of our website:
    - what courses do we offer: As of now Python Basic, Python Advance, MachineLearning intern.
    -How to register: Go to courses Section and click on register and then youll redirectt to a google form
    - how do i pay, youll get a QR code for Online money transfer.
    - what does -coursename- offer, please referto show pdf button at the course in courses section.
    -How do i resolve my issues: Please refer to our contact us page.

Keep it brief and clear for voice."""


async def get_groq_response(user_message: str, document_context: str = None) -> str:
    """
    Get ultra-fast response from Groq API.
    Supports both regular chat and document-based questions.
    
    Args:
        user_message: The user's question
        document_context: Optional document text for context
    
    Returns:
        AI response text
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        return "Sorry, Groq API key is not configured."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Choose system prompt based on context
    if document_context:
        system_prompt = CODEKIVY_DOCUMENT_PROMPT
        # Add document context to the user message
        enhanced_message = f"""Document Content:
{document_context}

User Question: {user_message}

Please answer based ONLY on the document content above."""
        max_tokens = 500  # Allow longer responses for document analysis
    else:
        system_prompt = CODEKIVY_CHAT_PROMPT
        enhanced_message = user_message
        max_tokens = 300
    
    payload = {
        "model": "llama-3.3-70b-versatile",  # Fast and accurate
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_message}
        ],
        "temperature": 0.3,  # Lower temperature for more accurate document analysis
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=15.0  # Groq is fast, but allow extra time for document analysis
            )
            
            response.raise_for_status()
            result = response.json()
            
            text = result["choices"][0]["message"]["content"]
            return text.strip()
            
    except httpx.HTTPStatusError as e:
        print(f"Groq HTTP error: {e}")
        if e.response.status_code == 401:
            return "Sorry, there's an issue with the API key."
        elif e.response.status_code == 429:
            return "Sorry, too many requests. Please wait a moment and try again."
        return f"Sorry, I'm having trouble connecting (Error: {e.response.status_code})."
    except Exception as e:
        print(f"Groq error: {e}")
        return "Sorry, something went wrong on my end."

async def get_groq_voice_response(messages_list: list) -> str:
    """
    Optimized for voice - shorter responses.
    Now accepts a full message history list.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        return "Sorry, Groq API key is not configured."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # --- MODIFIED PART ---
    # We now prepend the system prompt to the incoming message list
    # to create the final payload for the API.
    all_messages = [
        {"role": "system", "content": CODEKIVY_VOICE_PROMPT}
    ] + messages_list  # Add the entire history after the system prompt
    # --- END MODIFIED PART ---

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": all_messages,  # Pass the combined list
        "temperature": 0.7,
        "max_tokens": 150,  # Very short for voice
        "top_p": 1,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=10.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            text = result["choices"][0]["message"]["content"]
            return text.strip()
            
    except Exception as e:
        print(f"Groq voice error: {e}")
        return "Sorry, something went wrong."
