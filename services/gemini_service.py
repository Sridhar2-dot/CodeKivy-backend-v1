import httpx
import json
import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# Strict KivyBot rules to avoid Python library confusion
CODEKIVY_SYSTEM_PROMPT = """
<IDENTITY>
You are "KivyBot," the official assistant for CodeKivy.
IMPORTANT: You are NOT related to the Python 'Kivy' library. "Kivy" always refers to the company "CodeKivy".
</IDENTITY>

<CRITICAL_RULES>
1. Always use the word "CodeKivy" instead of "Kivy".
2. You help with Python, Machine Learning, and AI.
3. Greet users with the exact bullet-point list provided in your instructions.
4. Reply in the user's language (Telugu, Hindi, Kannada, Tamil, or English).
</CRITICAL_RULES>

<GREETING>
"👋 Hello! I'm KivyBot. I can help you with:
• Clarify Your Doubts.
• Code analysis
• Document analysis (upload PDF, TXT, DOCX)
• Screenshot analysis"
</GREETING>
"""

chat_histories: Dict[str, List[Dict]] = {}

def get_chat_history(session_id: str) -> List[Dict]:
    return chat_histories.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append({"role": role, "parts": [{"text": content}]})
    if len(chat_histories[session_id]) > 20:
        chat_histories[session_id] = chat_histories[session_id][-20:]

async def get_gemini_response(user_message: str, image_base64: Optional[str] = None, session_id: str = "default", use_history: bool = True):
    # 1. Verification of Env Vars
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if not raw_key:
        return "DEPLOYMENT_LOG: Error - GEMINI_API_KEY is not set in the server environment variables."

    api_key = raw_key.strip().replace('"', '').replace("'", "")
    
    # Using v1beta for Gemini 2.5 Flash as confirmed by your local diagnostics
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # 2. Payload Construction
    current_parts = [{"text": user_message}]
    if image_base64:
        try:
            image_data = image_base64.split(",")[1] if "," in image_base64 else image_base64
            current_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_data}})
        except Exception:
            return "DEPLOYMENT_LOG: Error processing image data."

    contents = []
    if use_history:
        contents.extend(get_chat_history(session_id))
    contents.append({"role": "user", "parts": current_parts})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": CODEKIVY_SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        }
    }

    # 3. Request with High Timeout and Explicit Logging
    try:
        # 60 second timeout to handle slow cloud-to-google handshakes
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
            # If deployment fails, THIS section will tell you why
            if response.status_code != 200:
                error_detail = response.text[:150] # Get first 150 chars of error
                return f"DEPLOYMENT_LOG: Status {response.status_code} - Error: {error_detail}"

            result = response.json()
            
            if not result.get('candidates'):
                return "DEPLOYMENT_LOG: Empty response from AI. Possible safety filter trigger."

            text = result['candidates'][0]['content']['parts'][0]['text']

            if use_history:
                add_to_history(session_id, "user", user_message)
                add_to_history(session_id, "model", text)
            return text

    except httpx.ConnectTimeout:
        return "DEPLOYMENT_LOG: Connection Timeout. Google API is unreachable from this server region."
    except httpx.ConnectError as e:
        return f"DEPLOYMENT_LOG: Connection Error. DNS or Network issue: {str(e)}"
    except Exception as e:
        return f"DEPLOYMENT_LOG: {type(e).__name__} - {str(e)}"

if __name__ == "__main__":
    # Local Test Execution
    print("Starting local test...")
    response = asyncio.run(get_gemini_response("Hi!"))
    print(f"Response: {response}")
