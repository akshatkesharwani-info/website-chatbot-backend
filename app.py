import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

with open("knowledge.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"]
site_name = data["site_name"]

import re

stop_list = ["the", "a", "an", "is", "are", "do", "you", "i", "we", "of", "to", "for", "and", "in", "on", "at", "it", "my", "your", "what", "how", "can", "with", "this", "that", "me", "us", "there", "any", "have", "has", "be", "or"]

def stem_tokens(text):
    # cut every word to 5 letters so deliver / delivery / delivering match
    words = re.findall(r"[a-z0-9]+", text.lower())
    result = []
    for w in words:
        if w not in stop_list:
            result.append(w[:5])
    return result

vectorizer = TfidfVectorizer(tokenizer=stem_tokens, lowercase=False, token_pattern=None)
matrix = vectorizer.fit_transform([c["text"] for c in chunks])

client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
model_name = "openai/gpt-oss-120b"

# change * to your website address before going live, for example https://mybakery.com
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

@app.route("/")
def home():
    return "Chatbot backend is running"

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204
    body = request.get_json(force=True)
    question = str(body.get("message", ""))[:500]
    history = body.get("history", [])[-4:]
    if question.strip() == "":
        return jsonify({"reply": "Please type a question."})

    q_vec = vectorizer.transform([question])
    scores = linear_kernel(q_vec, matrix).flatten()
    top_ids = scores.argsort()[::-1][:3]
    context = ""
    for i in top_ids:
        if scores[i] > 0:
            context = context + "[" + str(chunks[i]["page"]) + "] " + chunks[i]["text"] + "\n\n"
    if context == "":
        for c in chunks:
            if "contact" in str(c["page"]).lower():
                context = context + "[" + str(c["page"]) + "] " + c["text"] + "\n\n"
    if context == "":
        context = "No matching website information found."

    history_text = ""
    for h in history:
        history_text = history_text + str(h.get("role", "")) + ": " + str(h.get("text", ""))[:300] + "\n"

    system_msg = (
        "You are the friendly website assistant for " + site_name + ". "
        "Answer the visitor using only the website information given. Keep answers short (1 to 3 sentences). "
        "If the answer is not in the information, say you are not sure and ask the visitor to contact the business directly. "
        "Never make up prices, timings or offers."
    )
    user_msg = "Website information:\n" + context + "\nChat so far:\n" + history_text + "\nVisitor question: " + question

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, the assistant is not available right now. Please contact us directly."
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
