def classify_intent(text):
    return "general_question"


if __name__ == "__main__":
    result = classify_intent("hello")
    print(result)