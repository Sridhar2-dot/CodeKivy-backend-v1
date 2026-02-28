import httpx
import json
import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# Structured prompt for stricter rule adherence
CODEKIVY_SYSTEM_PROMPT = """
<IDENTITY>
You are "KivyBot," the official AI assistant for the company "CodeKivy".
</IDENTITY>

<CRITICAL_RULES>
1. NEVER mention or refer to the Python "Kivy" library/module. 
2. "Kivy" ALWAYS means the company "CodeKivy".
3. Use the word "CodeKivy" instead of "Kivy" in your text.
4. If a user asks about mobile app development using Kivy, steer them back to CodeKivy's Python/AI/ML courses.
5. You must reply in the user's language (TELUGU, HINDI, KANNADA, TAMIL, or English).
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

chat_histories: Dict[str, List[Dict]] = {}

def get_chat_history(session_id: str) -> List[Dict]:
    return chat_histories.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    # Standard Gemini role format
    chat_histories[session_id].append({"role": role, "parts": [{"text": content}]})
    if len(chat_histories[session_id]) > 20:
        chat_histories[session_id] = chat_histories[session_id][-20:]

async def get_gemini_response(user_message: str, image_base64: Optional[str] = None, session_id: str = "default", use_history: bool = True):
    # Sanitize API Key
    raw_key = os.getenv("GEMINI_API_KEY", "")
    api_key = raw_key.strip().replace('"', '').replace("'", "")

    if not api_key:
        return "Error: GEMINI_API_KEY is missing. Check your .env file."

    # Using gemini-2.5-flash as requested
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    current_parts = [{"text": user_message}]
    
    if image_base64:
        if "," in image_base64:
            header, image_data = image_base64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
        else:
            image_data = image_base64
            mime_type = "image/jpeg"
        current_parts.append({"inlineData": {"mimeType": mime_type, "data": image_data}})

    contents = []
    if use_history:
        contents.extend(get_chat_history(session_id))
    
    contents.append({"role": "user", "parts": current_parts})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": CODEKIVY_SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.3, # Lower temperature = stricter rule following
            "maxOutputTokens": 1024,
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                return f"API Error {response.status_code}: {response.text}"
            
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']

            if use_history:
                add_to_history(session_id, "user", user_message)
                add_to_history(session_id, "model", text)
            
            return text

    except Exception as e:
        return f"System Error: {str(e)}"

# --- Simple Test Execution ---
async def main():
    response = await get_gemini_response("Hi!")
    print(f"Bot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
