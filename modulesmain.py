import sys
from config import CONFIG
from model import text_extractor, speech_processor, qa_system, conversation_manager
from utils import logger, error_handler

log = logger.get_logger()

def main():
    log.info("Memulai Enhanced AI System...")
    conversation = conversation_manager.ConversationManager(max_history=20)
    
    while True:
        try:
            print("Pilih mode input:")
            print("1. Ketik pertanyaan")
            print("2. Unggah file")
            print("3. Suara (voice-to-text)")
            mode = input("Masukkan pilihan (1/2/3): ").strip()
            
            if mode == "1":
                question = input("Tulis pertanyaan Anda: ")
                context = ""
            elif mode == "2":
                file_path = input("Masukkan path file: ").strip()
                context = text_extractor.extract_text(file_path)
                print("Isi file berhasil diekstraksi.")
                question = input("Tulis pertanyaan terkait file ini: ")
            elif mode == "3":
                question = speech_processor.voice_to_text()
                context = ""
            else:
                print("Pilihan tidak valid.")
                continue

            if question:
                answer = qa_system.answer_question(question, context, conversation.get_history())
                print("Jawaban:", answer)
                speech_processor.text_to_voice(answer)
                conversation.add_entry("Q: " + question, "A: " + answer)
            else:
                print("Tidak ada pertanyaan yang diproses.")
            
            ulang = input("Ingin mencoba pertanyaan lain? (y/n): ").strip().lower()
            if ulang != "y":
                break
        except Exception as e:
            error_handler.handle_error(e)
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Program dihentikan oleh pengguna.")
        sys.exit(0)
