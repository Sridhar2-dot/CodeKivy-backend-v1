import httpx
import json
import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# Strict XML-tagged system prompt to define the bot's persona and prevent hallucinations
CODEKIVY_SYSTEM_PROMPT = """
<IDENTITY>
You are "KivyBot," the official AI assistant for the company "CodeKivy".
IMPORTANT: You are NOT related to the Python 'Kivy' library. "Kivy" always refers to the company "CodeKivy".
</IDENTITY>

<CRITICAL_RULES>
1. NEVER mention or refer to the Python "Kivy" library/module.
2. "Kivy" ALWAYS means the company "CodeKivy".
3. Use the word "CodeKivy" instead of "Kivy" in your text.
4. If a user asks about mobile app development using Kivy, steer them back to CodeKivy's Python/AI/ML courses.
5. Reply in the user's language (Telugu, Hindi, Kannada, Tamil, or English).
</CRITICAL_RULES>

<COMPANY_DETAILS>
- Founded: 17 Apr 2023 by Pavan Nekkanti.
- Features: Affordable Prices, Live Online Classes, Weekly assignments, Doubt clarification, Realtime Projects.
- Courses: Python Basic, Python Advance, Machine Learning Intern.
- Registration: Go to Courses Section and click register to redirect to a Google Form.
- Founder Details: Founded by Pavan Nekkanti; currently on the 6th batch.
</COMPANY_DETAILS>

<GREETING_TEMPLATE>
If the user says "Hi", "Hello", or greets you, you MUST reply with this EXACT structure:
"👋 Hello! I'm KivyBot. I can help you with:
• Clarify Your Doubts.
• Code analysis
• Document analysis (upload PDF, TXT, DOCX)
• Screenshot analysis"
</GREETING_TEMPLATE>
"""

# Store chat history per session
chat_histories: Dict[str, List[Dict]] = {}

def get_chat_history(session_id: str) -> List[Dict]:
    """Retrieve chat history for a session."""
    return chat_histories.get(session_id, [])

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
    """Clear chat history for a session. (Required for Vercel main.py import)"""
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return {"status": "success", "message": f"History cleared for session {session_id}."}

async def get_gemini_response(
    user_message: str, 
    image_base64: Optional[str] = None,
    session_id: str = "default",
    use_history: bool = True
):
    """Get response from Gemini with chat history and deployment safety."""
    
    # Securely fetch and clean the API key for deployment environments
    raw_key = os.getenv("GEMINI_API_KEY", "")
    api_key = raw_key.strip().replace('"', '').replace("'", "")

    if not api_key:
        return "System Error: GEMINI_API_KEY is missing from environment variables."

    # Using Gemini 2.5 Flash on v1beta
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Build the current message parts
    current_parts = [{"text": user_message}]

    if image_base64:
        try:
            if "," in image_base64:
                header, image_data = image_base64.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
            else:
                image_data = image_base64
                mime_type = "image/jpeg"
            current_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": image_data
                }
            })
        except Exception as e:
            print(f"Image parse error: {e}")

    # Build contents array with history
    contents = []
    if use_history:
        contents.extend(get_chat_history(session_id))
    
    contents.append({
        "role": "user",
        "parts": current_parts
    })

    # Construct the Payload
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": CODEKIVY_SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "temperature": 0.3, # Strict adherence to CodeKivy rules
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }

    try:
        # 60s timeout allows for slow Vercel cold-starts
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload
            )

            # Clean error handling for the frontend
            if response.status_code != 200:
                print(f"API Error Log: Status {response.status_code} - {response.text[:200]}")
                if response.status_code == 429:
                    return "KivyBot is a bit busy right now. Please wait a few seconds and try again! (Rate Limit)"
                elif response.status_code == 400:
                    return "Sorry, there is an issue with the AI payload configuration. (Error 400)"
                return f"Sorry, I'm having trouble connecting to the AI (HTTP {response.status_code})."

            result = response.json()
            
            # Guard against safety filter blocking
            if not result.get('candidates'):
                return "Sorry, I couldn't generate a response to that request."
                
            text = result['candidates'][0]['content']['parts'][0]['text']

            # Store in history
            if use_history:
                add_to_history(session_id, "user", user_message)
                add_to_history(session_id, "model", text)

            return text

    except httpx.ConnectTimeout:
        return "Connection Timeout: The server took too long to reach the AI."
    except httpx.ConnectError:
        return "Connection Error: Unable to reach the AI servers from the backend."
    except Exception as e:
        print(f"System Exception: {e}") 
        return "Sorry, something went wrong on my end."
