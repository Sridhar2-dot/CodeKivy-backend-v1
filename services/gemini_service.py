import httpx
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# The system prompt to define the bot's persona for CodeKivy
CODEKIVY_SYSTEM_PROMPT = """
You are "KivyBot," the official assistant for CodeKivy, You are not related Python's Kivy module so donot even refer that even if user asks he wants to kno wabout kivy, refer to codekivy Blindly fix as Kivy means Codekivy the company name not python's kivy module.
Your persona is friendly, encouraging, and knowledgeable, like a helpful tutor.
You must reply in specific language if user asks in different language, english isnot mandatory most probable languages are: TELUGU, HINDI, KANNADA, TAMIL.
Your main goal is to help users learn programming concepts related to Python MachineLearning and related, Donot specify out of the Box concepts like mobile and app developement and not even Ui and more, just focus mainly on python and Ml and AI related feilds, Donot specify anything like waht do you want to know about python, like that, Be general, ALSO REMEMBER YOU SHOULD REFER TO A MODULE NAMED KIVY IN PYTHON.
If user Greets as Hi or Hello or Any kind of Intorductory Greetings: Give this Exact Structure without changing anythign:
Greet Back and 
"👋 Hello! I'm KivyBot. I can help you with:
• Clarify Your Doubts.
• Code analysis
• Document analysis (upload PDF, TXT, DOCX)
• Screenshot analysis"

RULES:
1.  **Be Concise:** Keep answers short and easy to read for a chat window.
2.  **Be Encouraging:** Use positive language ,but not always(e.g., "Great question!", "That's a common concept!").
3.  **Handle Off-Topic Questions:** Briefly answer and steer back: What else shall we do, Do you have any doubts? like that"
4.  **Greet Users:** If the user says "hi," introduce yourself as KivyBot👋 Hello! I'm KivyBot. I can help you:
• Clarify your Doubts.
• Code analysis
• Document analysis (upload PDF, TXT, DOCX)
• Screen analysis.
. guiding through Codekivy's features sections and all
5.  **Handle Screenshots:** If you are given a screenshot, you MUST analyze it based on the user's prompt "Say like Analysed the Captured Area or what do you want from the Captured Area...
    - If it's a screenshot of code, analyze the code for errors, explain what it does, or suggest improvements.
    - If it's a screenshot of the website, answer the user's question about it.
6. If user wants to know anything about codekiwi, here are the details that you have to refer to, donot say anythign thats not inhere
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

MANDATORY RULES :  1. Dont say anything related to python's kivy module codekivy means company name not module.
                   2. Dont use kivy word individually always use codekivy word in the text.        

"""

# Store chat history per session
chat_histories: Dict[str, List[Dict]] = {}


def get_chat_history(session_id: str) -> List[Dict]:
    """Retrieve chat history for a session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def add_to_history(session_id: str, role: str, content: str):
    """Add a message to chat history."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    chat_histories[session_id].append({
        "role": role,
        "parts": [{"text": content}]
    })
    
    # Keep only last 10 exchanges (20 messages) to avoid token limits
    if len(chat_histories[session_id]) > 20:
        chat_histories[session_id] = chat_histories[session_id][-20:]


def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    if session_id in chat_histories:
        chat_histories[session_id] = []


async def get_gemini_response(
    user_message: str, 
    image_base64: Optional[str] = None,
    session_id: str = "default",
    use_history: bool = True
):
    # Fetch and clean the API Key
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    # Standardized URL for the v1beta endpoint with the stable 1.5-flash model
    # Note: 404 is usually caused by an incorrect model name or extra characters in the key
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    url = f"{base_url}?key={api_key}"

    # --- Build the current message parts ---
    current_parts = []
    current_parts.append({"text": user_message})

    if image_base64:
        # Process image
        if "," in image_base64:
            image_data = image_base64.split(",")[1]
        else:
            image_data = image_base64

        current_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": image_data
            }
        })

    # --- Build contents with history ---
    contents = []
    
    if use_history:
        # Add previous conversation history
        history = get_chat_history(session_id)
        contents.extend(history)
    
    # Add current user message
    contents.append({
        "role": "user",
        "parts": current_parts
    })

    # --- Construct the Payload ---
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": CODEKIVY_SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60.0
            )

            response.raise_for_status() 
            
            result = response.json()
            
            candidate = result.get("candidates", [{}])[0]
            content = candidate.get("content", {})
            part = content.get("parts", [{}])[0]
            text = part.get("text", "Sorry, I couldn't generate a response right now.")
            
            # --- Store in history (only if using history) ---
            if use_history:
                add_to_history(session_id, "user", user_message)
                add_to_history(session_id, "model", text)
            
            return text

    except httpx.HTTPStatusError as e:
        # Debugging prints for production logs
        print(f"HTTP error occurred: {e.response.status_code}")
        print(f"Error Response Body: {e.response.text}")
        
        if e.response.status_code == 400:
            return "Sorry, there seems to be an issue with the API configuration. (Error 400)"
        elif e.response.status_code == 404:
            return "Sorry, the AI model endpoint was not found. Please check the model name. (Error 404)"
            
        return f"Sorry, I'm having trouble connecting to the AI (HTTP error: {e.response.status_code})."
    except Exception as e:
        print(f"An error occurred: {e}") 
        return "Sorry, something went wrong on my end."
