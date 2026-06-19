import json
import pickle
import random

with open("intents.json", "r") as file:
    data = json.load(file)

with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

rule_keywords = {
    "hi": "greeting",
    "hello": "greeting",
    "bye": "goodbye",
    "thanks": "thanks"
}

def get_response(user_input):
    user_input_lower = user_input.lower()

    for keyword, tag in rule_keywords.items():
        if keyword in user_input_lower:
            return find_response(tag)

    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    return find_response(predicted_tag)

def find_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

print("Hybrid Chatbot started!")
print("Type 'quit' to exit.")

while True:
    message = input("You: ")

    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break

    response = get_response(message)
    print("Bot:", response)
