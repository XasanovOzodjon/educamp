"""
OPTIMALLASHTIRILGAN Yuzni aniqlash tizimi
Tez ishlash uchun: embedding cache, threading, kichik rasm
"""

import cv2
import os
import pickle
import numpy as np
from threading import Thread
from queue import Queue
import time

# DeepFace import
from deepface import DeepFace
from deepface.commons import functions


class FastFaceRecognizer:
    def __init__(self, dataset_path="./dataset"):
        self.dataset_path = dataset_path
        self.embeddings_file = os.path.join(dataset_path, "embeddings.pkl")
        self.embeddings = {}  # {name: [embedding1, embedding2, ...]}
        
        # Parametrlar - TEZLIK UCHUN OPTIMALLASHTIRILGAN
        self.model_name = "Facenet"  # Facenet tez va aniq
        self.detector_backend = "opencv"  # Eng tez detector
        self.threshold = 0.7  # Facenet uchun chegara
        self.resize_scale = 0.5  # Rasmni kichiklashtirish (2x tez)
        
        # Dataset papkasi
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        # Oldindan hisoblangan embeddinglarni yuklash
        self.load_embeddings()
        
        print(f"Model: {self.model_name}")
        print(f"Yuklangan odamlar: {len(self.embeddings)}")
    
    def load_embeddings(self):
        """Saqlangan embeddinglarni yuklash"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Embeddinglar yuklandi: {list(self.embeddings.keys())}")
            except:
                self.embeddings = {}
    
    def save_embeddings(self):
        """Embeddinglarni saqlash"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def get_embedding(self, img):
        """Rasmdan embedding olish"""
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding']), result[0].get('facial_area')
        except:
            pass
        return None, None
    
    def cosine_distance(self, emb1, emb2):
        """Ikki embedding orasidagi masofa"""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return 1 - (dot / (norm1 * norm2))
    
    def find_person(self, embedding):
        """Embeddingni dataset bilan solishtirish - JUDA TEZ"""
        best_match = None
        best_distance = float('inf')
        
        for name, emb_list in self.embeddings.items():
            for stored_emb in emb_list:
                dist = self.cosine_distance(embedding, stored_emb)
                if dist < best_distance:
                    best_distance = dist
                    best_match = name
        
        if best_distance < self.threshold:
            confidence = (1 - best_distance) * 100
            return best_match, confidence, best_distance
        
        return None, 0, best_distance
    
    def add_person(self, name):
        """Yangi odam qo'shish"""
        person_path = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        collected_embeddings = []
        count = 0
        total = 5
        
        print(f"\n'{name}' uchun {total} ta rasm olinadi.")
        print("SPACE - rasm olish | Q - chiqish\n")
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        while count < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(display, f"Rasmlar: {count}/{total}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "SPACE - olish | Q - chiqish", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Qo'shish: {name}", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                embedding, _ = self.get_embedding(frame)
                if embedding is not None:
                    collected_embeddings.append(embedding)
                    img_path = os.path.join(person_path, f"{count+1}.jpg")
                    cv2.imwrite(img_path, frame)
                    count += 1
                    print(f"✓ Rasm {count} saqlandi")
                else:
                    print("✗ Yuz topilmadi, qayta urinib ko'ring")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_embeddings:
            self.embeddings[name] = collected_embeddings
            self.save_embeddings()
            print(f"\n'{name}' muvaffaqiyatli qo'shildi!")
        
        return count
    
    def recognize_realtime(self):
        """OPTIMALLASHTIRILGAN real vaqtda aniqlash"""
        
        if not self.embeddings:
            print("\n⚠ Dataset bo'sh! Avval odam qo'shing (menyu: 1)")
            return
        
        print(f"\nDataset: {list(self.embeddings.keys())}")
        print("Q - chiqish\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # OpenCV yuz detektori (juda tez)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Threading uchun
        result_queue = Queue(maxsize=1)
        frame_queue = Queue(maxsize=1)
        
        # Natija saqlash
        current_result = {'name': None, 'confidence': 0, 'box': None}
        frame_count = 0
        process_every = 5  # Har 5 kadrda bir embedding hisoblash
        
        def process_frame():
            """Alohida threadda yuzni aniqlash"""
            while True:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    if frame is None:
                        break
                    
                    # Kichikroq rasmda ishlash
                    small = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
                    embedding, face_area = self.get_embedding(small)
                    
                    if embedding is not None:
                        name, conf, dist = self.find_person(embedding)
                        
                        # Koordinatalarni katta rasmga moslashtirish
                        if face_area:
                            scale = 1 / self.resize_scale
                            box = (
                                int(face_area['x'] * scale),
                                int(face_area['y'] * scale),
                                int(face_area['w'] * scale),
                                int(face_area['h'] * scale)
                            )
                        else:
                            box = None
                        
                        if not result_queue.full():
                            result_queue.put({
                                'name': name,
                                'confidence': conf,
                                'box': box
                            })
        
        # Threadni boshlash
        process_thread = Thread(target=process_frame, daemon=True)
        process_thread.start()
        
        fps_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display = frame.copy()
            
            # FPS hisoblash
            if time.time() - fps_time >= 1:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Har N kadrda frameni yuborish
            if frame_count % process_every == 0:
                if frame_queue.empty():
                    frame_queue.put(frame.copy())
            
            # Natijani olish
            if not result_queue.empty():
                current_result = result_queue.get()
            
            # Tez yuz aniqlash (faqat ramka uchun)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            for (x, y, w, h) in faces:
                if current_result['name']:
                    color = (0, 255, 0)  # Yashil
                    name = current_result['name']
                    conf = current_result['confidence']
                    text = f"{name} ({conf:.0f}%)"
                else:
                    color = (0, 0, 255)  # Qizil
                    text = "Notanish"
                
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                
                # Matn foni
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display, (x, y-30), (x+tw+10, y), color, -1)
                cv2.putText(display, text, (x+5, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Info
            cv2.putText(display, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Q - chiqish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Yuz aniqlash (Optimallashtirilgan)", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Threadni to'xtatish
        frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
    
    def list_persons(self):
        """Ro'yxat"""
        if self.embeddings:
            print("\n=== Datasetdagi odamlar ===")
            for name, emb_list in self.embeddings.items():
                print(f"  {name}: {len(emb_list)} ta embedding")
        else:
            print("\nDataset bo'sh!")
        return list(self.embeddings.keys())
    
    def delete_person(self, name):
        """O'chirish"""
        import shutil
        if name in self.embeddings:
            del self.embeddings[name]
            self.save_embeddings()
            
            person_path = os.path.join(self.dataset_path, name)
            if os.path.exists(person_path):
                shutil.rmtree(person_path)
            
            print(f"'{name}' o'chirildi!")
            return True
        print(f"'{name}' topilmadi!")
        return False
    
    def rebuild_embeddings(self):
        """Barcha rasmlardan embeddinglarni qayta hisoblash"""
        print("\nEmbeddinglar qayta hisoblanmoqda...")
        self.embeddings = {}
        
        for name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, name)
            if os.path.isdir(person_path) and name != "__pycache__":
                emb_list = []
                for img_name in os.listdir(person_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_path, img_name)
                        emb, _ = self.get_embedding(img_path)
                        if emb is not None:
                            emb_list.append(emb)
                
                if emb_list:
                    self.embeddings[name] = emb_list
                    print(f"  {name}: {len(emb_list)} ta embedding")
        
        self.save_embeddings()
        print("Tayyor!")


def main():
    print("=" * 50)
    print("  TEZKOR YUZ ANIQLASH TIZIMI")
    print("=" * 50)
    
    recognizer = FastFaceRecognizer("./dataset")
    
    while True:
        print("\n" + "-" * 40)
        print("1. Yangi odam qo'shish")
        print("2. Real vaqtda aniqlash")
        print("3. Odamlar ro'yxati")
        print("4. Odamni o'chirish")
        print("5. Embeddinglarni qayta hisoblash")
        print("6. Chiqish")
        print("-" * 40)
        
        tanlov = input("Tanlang (1-6): ").strip()
        
        if tanlov == "1":
            name = input("Ism: ").strip()
            if name:
                recognizer.add_person(name)
        
        elif tanlov == "2":
            recognizer.recognize_realtime()
        
        elif tanlov == "3":
            recognizer.list_persons()
        
        elif tanlov == "4":
            recognizer.list_persons()
            name = input("O'chirish uchun ism: ").strip()
            if name:
                recognizer.delete_person(name)
        
        elif tanlov == "5":
            recognizer.rebuild_embeddings()
        
        elif tanlov == "6":
            print("Xayr!")
            break


if __name__ == "__main__":
    main()