import os
import io
import torch
from transformers import pipeline
import speech_recognition as sr
import pyttsx3
import PyPDF2
import docx
import pandas as pd
from PIL import Image
import pytesseract

import torch
x = torch.rand(5, 3)
print(x)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif ext in [".doc", ".docx"]:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            text = df.to_string()
        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = "Format file tidak didukung."
    except Exception as e:
        text = f"Error ekstraksi: {str(e)}"
    return text

def voice_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            print("Silakan berbicara...")
            audio = recognizer.listen(source, phrase_time_limit=10)
        result = recognizer.recognize_google(audio, language="id-ID")
        print("Teks terdeteksi:", result)
        return result
    except sr.UnknownValueError:
        print("Maaf, suara tidak dapat dikenali.")
        return ""
    except sr.RequestError as e:
        print("Error pada layanan speech recognition:", str(e))
        return ""

def text_to_voice(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error text-to-speech:", str(e))

def answer_question(question, context=None, conversation_memory=None):
    # Jika ada konteks tambahan (misal dari file atau history), gabungkan
    combined_context = ""
    if conversation_memory:
        combined_context += "\n".join(conversation_memory) + "\n"
    if context:
        combined_context += context
    else:
        combined_context += "Jawaban bersifat general dari model."
    try:
        qa = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")
        result = qa(question=question, context=combined_context)
        return result['answer']
    except Exception as e:
        return f"Terjadi kesalahan pada proses jawaban: {str(e)}"

def process_input():
    print("Pilih mode input:")
    print("1. Ketik pertanyaan")
    print("2. Unggah file (PDF, Word, Excel, Gambar, atau teks)")
    print("3. Suara (voice-to-text)")
    mode = input("Masukkan pilihan (1/2/3): ").strip()
    question, context = "", ""
    if mode == "1":
        question = input("Tulis pertanyaan Anda: ")
    elif mode == "2":
        file_path = input("Masukkan path file: ").strip()
        if os.path.exists(file_path):
            context = extract_text(file_path)
            print("Isi file berhasil diekstraksi.")
            question = input("Tulis pertanyaan terkait file ini: ")
        else:
            print("File tidak ditemukan.")
    elif mode == "3":
        question = voice_to_text()
    else:
        print("Pilihan tidak valid.")
    return question, context

def main():
    conversation_memory = []
    print("Selamat datang di Zeronex AI")
    while True:
        question, context = process_input()
        if question:
            answer = answer_question(question, context, conversation_memory)
            print("Jawaban:", answer)
            text_to_voice(answer)
            conversation_memory.append("Q: " + question)
            conversation_memory.append("A: " + answer)
        else:
            print("Tidak ada pertanyaan yang diproses.")
        ulang = input("Ingin mencoba pertanyaan lain? (y/n): ").strip().lower()
        if ulang != "y":
            break

if __name__ == "__main__":
    main()
