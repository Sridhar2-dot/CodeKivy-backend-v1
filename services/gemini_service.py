import httpx
import json
import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# The system prompt with strict identity rules for CodeKivy
CODEKIVY_SYSTEM_PROMPT = """
<IDENTITY>
You are "KivyBot," the official assistant for CodeKivy. 
IMPORTANT: You are NOT related to the Python 'Kivy' library. "Kivy" always refers to the company "CodeKivy".
</IDENTITY>

<RULES>
1. Always refer to the company as "CodeKivy".
2. Focus on Python, Machine Learning, and AI courses.
3. If asked about mobile apps/UI, steer the user back to CodeKivy's AI/ML focus.
4. Reply in the user's language (Telugu, Hindi, Kannada, Tamil, or English).
</RULES>

<GREETING_TEMPLATE>
If the user greets you (Hi/Hello), reply exactly with:
"👋 Hello! I'm KivyBot. I can help you with:
• Clarify Your Doubts.
• Code analysis
• Document analysis (upload PDF, TXT, DOCX)
• Screenshot analysis"
</GREETING_TEMPLATE>
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
    # --- 1. ENV VAR CHECK ---
    raw_key = os.getenv("GEMINI_API_KEY", "AIzaSyBER0_4DGP9GD_UTjTMUH9KigQ8rOzMCMc")
    if not raw_key:
        return "DEBUG_ERROR: GEMINI_API_KEY is missing from your deployment environment variables."

    api_key = raw_key.strip().replace('"', '').replace("'", "")
    
    # We use v1beta for Gemini 2.5 Flash
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # --- 2. PAYLOAD CONSTRUCTION ---
    current_parts = [{"text": user_message}]
    if image_base64:
        try:
            image_data = image_base64.split(",")[1] if "," in image_base64 else image_base64
            current_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_data}})
        except Exception:
            return "DEBUG_ERROR: Failed to process image base64 data."

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

    # --- 3. REQUEST & SPECIFIC EXCEPTION HANDLING ---
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(url, json=payload)
            
            # If not 200 OK, return specific HTTP code debugs
            if response.status_code != 200:
                if response.status_code == 401:
                    return f"DEBUG_ERROR: 401 Unauthorized. Check your API Key string. Raw Response: {response.text[:50]}"
                elif response.status_code == 429:
                    return "DEBUG_ERROR: 429 Quota Exceeded. You are hitting the free tier limit."
                elif response.status_code == 403:
                    return f"DEBUG_ERROR: 403 Forbidden. Is your key restricted by region? Response: {response.text[:50]}"
                elif response.status_code == 404:
                    return f"DEBUG_ERROR: 404 Not Found. Model '{model_name}' might not be available in this region."
                else:
                    return f"DEBUG_ERROR: HTTP {response.status_code} - {response.text[:100]}"

            result = response.json()
            
            # Handle empty candidates (Safety filters)
            if not result.get('candidates'):
                return "DEBUG_ERROR: No candidates returned. The AI might have blocked the response for safety."
            
            text = result['candidates'][0]['content']['parts'][0]['text']

            if use_history:
                add_to_history(session_id, "user", user_message)
                add_to_history(session_id, "model", text)
            return text

    except httpx.ConnectTimeout:
        return "DEBUG_ERROR: Connection Timeout. The server is taking too long to reach Google's API."
    except httpx.ConnectError:
        return "DEBUG_ERROR: Connection Error. DNS failed or the server has no internet access."
    except Exception as e:
        return f"DEBUG_ERROR: Unexpected {type(e).__name__}: {str(e)}"

# --- DIAGNOSTIC UTILITY ---
async def run_diagnostics():
    print("\n--- 🔍 STARTING DEPLOYMENT DIAGNOSTICS ---")
    raw_key = os.getenv("GEMINI_API_KEY", "")
    print(f"Key Found in Env: {'YES' if raw_key else 'NO'}")
    if not raw_key: return

    api_key = raw_key.strip().replace('"', '').replace("'", "")
    diag_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(diag_url)
            print(f"Status: {res.status_code}")
            if res.status_code == 200:
                print("✅ Key is valid and can list models.")
            else:
                print(f"❌ Key check failed: {res.text}")
        except Exception as e:
            print(f"❌ Diagnostic Network Error: {e}")

if __name__ == "__main__":
    # For local testing, ensure your .env is set
    asyncio.run(run_diagnostics())
    test_resp = asyncio.run(get_gemini_response("Hi!"))
    print(f"\nFinal Test Response:\n{test_resp}")
