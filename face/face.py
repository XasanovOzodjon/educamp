"""
SODDA VA ANIQ YUZ ANIQLASH TIZIMI
"""

import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace


class FaceRecognizer:
    def __init__(self, dataset_path="./dataset"):
        self.dataset_path = dataset_path
        self.embeddings_file = os.path.join(dataset_path, "embeddings.pkl")
        self.embeddings = {}
        
        # ENG ANIQ SOZLAMALAR
        self.model_name = "ArcFace"  # Eng aniq model
        self.threshold = 0.68  # ArcFace uchun optimal
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        self.load_embeddings()
        print(f"✓ Model: {self.model_name}")
        print(f"✓ Odamlar: {list(self.embeddings.keys()) if self.embeddings else 'yo`q'}")
    
    def load_embeddings(self):
        """Embeddinglarni yuklash"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
    
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
                enforce_detection=True,
                detector_backend="retinaface"  # Eng aniq detector
            )
            if result:
                return np.array(result[0]['embedding'])
        except:
            return None
        return None
    
    def find_person(self, embedding):
        """Eng yaqin odamni topish"""
        best_name = None
        best_score = 0
        
        for name, emb_list in self.embeddings.items():
            for stored_emb in emb_list:
                # Cosine similarity
                score = np.dot(embedding, stored_emb) / (np.linalg.norm(embedding) * np.linalg.norm(stored_emb))
                
                if score > best_score:
                    best_score = score
                    best_name = name
        
        if best_score > self.threshold:
            return best_name, best_score * 100
        return None, 0
    
    def add_person(self, name):
        """Yangi odam qo'shish"""
        person_path = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        
        cap = cv2.VideoCapture(0)
        embeddings_list = []
        count = 0
        total = 5
        
        print(f"\n{name} uchun {total} ta rasm kerak")
        print("SPACE - rasm olish | Q - chiqish\n")
        
        while count < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            cv2.putText(display, f"Rasmlar: {count}/{total}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "SPACE - olish", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Yangi odam", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print("Rasm olinmoqda...")
                emb = self.get_embedding(frame)
                if emb is not None:
                    embeddings_list.append(emb)
                    cv2.imwrite(os.path.join(person_path, f"{count+1}.jpg"), frame)
                    count += 1
                    print(f"✓ {count}/{total}")
                else:
                    print("✗ Yuz topilmadi!")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if embeddings_list:
            self.embeddings[name] = embeddings_list
            self.save_embeddings()
            print(f"\n✓ {name} qo'shildi!")
    
    def recognize(self):
        """Real vaqtda aniqlash"""
        if not self.embeddings:
            print("\n✗ Avval odam qo'shing!")
            return
        
        print(f"\nDataset: {list(self.embeddings.keys())}")
        print("Q - chiqish\n")
        
        cap = cv2.VideoCapture(0)
        
        current_name = None
        current_conf = 0
        skip = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Har 10 kadrda aniqlash
            skip += 1
            if skip >= 10:
                skip = 0
                emb = self.get_embedding(frame)
                if emb is not None:
                    name, conf = self.find_person(emb)
                    current_name = name
                    current_conf = conf
                else:
                    current_name = None
            
            # Natijani ko'rsatish
            if current_name:
                text = f"{current_name} ({current_conf:.0f}%)"
                color = (0, 255, 0)
            else:
                text = "Notanish"
                color = (0, 0, 255)
            
            # Katta matn yuqorida
            cv2.rectangle(display, (0, 0), (400, 80), color, -1)
            cv2.putText(display, text, (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            cv2.imshow("Yuz aniqlash", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def list_persons(self):
        """Odamlar ro'yxati"""
        if self.embeddings:
            print("\n--- Odamlar ---")
            for name in self.embeddings:
                print(f"  • {name}")
        else:
            print("\nBo'sh!")
    
    def delete_person(self, name):
        """O'chirish"""
        import shutil
        if name in self.embeddings:
            del self.embeddings[name]
            self.save_embeddings()
            path = os.path.join(self.dataset_path, name)
            if os.path.exists(path):
                shutil.rmtree(path)
            print(f"✓ {name} o'chirildi")
        else:
            print("✗ Topilmadi")


def main():
    print("\n" + "="*40)
    print("   YUZ ANIQLASH TIZIMI")
    print("="*40)
    
    fr = FaceRecognizer("./dataset")
    
    while True:
        print("\n1. Odam qo'shish")
        print("2. Aniqlash")
        print("3. Ro'yxat")
        print("4. O'chirish")
        print("5. Chiqish")
        
        t = input("\n> ").strip()
        
        if t == "1":
            name = input("Ism: ").strip()
            if name:
                fr.add_person(name)
        elif t == "2":
            fr.recognize()
        elif t == "3":
            fr.list_persons()
        elif t == "4":
            fr.list_persons()
            name = input("Ism: ").strip()
            if name:
                fr.delete_person(name)
        elif t == "5":
            break


if __name__ == "__main__":
    main()