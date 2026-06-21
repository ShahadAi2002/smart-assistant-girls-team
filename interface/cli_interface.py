# -*- coding: utf-8 -*-
def show_menu():
    print("===== Smart Assistant Menu =====")
    print("1. تحليل نص (نية + مشاعر + كلمات مفتاحية)")
    print("2. تحليل صورة")
    print("3. عرض آخر التفاعلات")
    print("4. عرض الإحصائيات")
    print("5. خروج")


def main():
    while True:
        show_menu()
        choice = input("اختاري رقم من القائمة: ")

        if choice == "1":
            print("تحليل النص - mock")
        elif choice == "2":
            print("تحليل الصورة - mock")
        elif choice == "3":
            print("عرض آخر التفاعلات - mock")
        elif choice == "4":
            print("عرض الإحصائيات - mock")
        elif choice == "5":
            print("خروج من البرنامج")
            break
        else:
            print("اختيار غير صحيح، حاولي مرة ثانية")


if __name__ == "__main__":
    main()