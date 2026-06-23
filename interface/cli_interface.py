import sys
import os

# نضيف مسار مجلد nlp-module عشان نقدر نستورد الملفات
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "nlp-module"))

from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from summarizer import summarize_text


def analyze_text():
    text = input("اكتبي النص: ")

    intent = classify_intent(text)
    sentiment = analyze_sentiment(text)
    keywords = extract_keywords(text)
    summary = summarize_text(text)

    print("\n--- نتائج تحليل النص ---")
    print("Intent:", intent)
    print("Sentiment:", sentiment)
    print("Keywords:", keywords)
    print("Summary:", summary)


def analyze_image():
    print("تحليل الصورة mock - سيتم تطويره لاحقًا")


def show_last_interactions():
    print("عرض آخر التفاعلات mock - سيتم تطويره لاحقًا")


def show_statistics():
    print("عرض الإحصائيات mock - سيتم تطويره لاحقًا")


def main_menu():
    while True:
        print("\n--- Smart Assistant Menu ---")
        print("1. تحليل نص (نية + مشاعر + كلمات مفتاحية)")
        print("2. تحليل صورة")
        print("3. عرض آخر التفاعلات")
        print("4. عرض الإحصائيات")
        print("5. خروج")

        choice = input("اختاري رقم: ")

        if choice == "1":
            analyze_text()
        elif choice == "2":
            analyze_image()
        elif choice == "3":
            show_last_interactions()
        elif choice == "4":
            show_statistics()
        elif choice == "5":
            print("تم الخروج من البرنامج.")
            break
        else:
            print("اختيار غير صحيح، حاولي مرة ثانية.")


if __name__ == "__main__":
    main_menu()