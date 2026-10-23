These 3 files are the chatbot backend (sample bakery website).
Upload ALL 3 files to a new GitHub repo (at the top level, not inside a folder):
  app.py
  knowledge.json
  requirements.txt
Render settings:
  Build command:  pip install -r requirements.txt
  Start command:  gunicorn app:app
  Environment variables: GROQ_API_KEY (your Groq key) and ALLOWED_ORIGIN (* while testing)
Do NOT put your API key in any file.
