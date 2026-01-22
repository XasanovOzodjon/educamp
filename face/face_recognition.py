"""
Real vaqtda yuzni aniqlash va dataset bilan solishtirish
DeepFace + OpenCV yordamida
"""

from deepface import DeepFace
import cv2
import os
import numpy as np
from datetime import datetime


class FaceRecognizer:
    def __init__(self, dataset_path="./dataset"):
        """
        dataset_path: Yuzlar saqlanadigan papka
        Struktura:
            dataset/
                Ali/
                    1.jpg
                    2.jpg
                Vali/
                    1.jpg
                    2.jpg
        """
        self.dataset_path = dataset_path
        self.model_name = "VGG-Face"  # Tez va aniq model
        self.detector_backend = "opencv"  # Tez detector
        self.distance_threshold = 0.6  # Qanchalik o'xshash bo'lishi kerak
        
        # Dataset papkasini yaratish
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            print(f"Dataset papkasi yaratildi: {dataset_path}")
        
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_path}")
    
    def add_person(self, name):
        """Yangi odam qo'shish - kameradan 5 ta rasm oladi"""
        person_path = os.path.join(self.dataset_path, name)
        
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        
        cap = cv2.VideoCapture(0)
        count = 0
        total_images = 5
        
        print(f"\n'{name}' uchun {total_images} ta rasm olinadi.")
        print("SPACE tugmasini bosing rasm olish uchun. Q - chiqish.\n")
        
        while count < total_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Yuzni aniqlash va ko'rsatish
            display_frame = frame.copy()
            
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                for face in faces:
                    if face['confidence'] > 0.9:
                        x = face['facial_area']['x']
                        y = face['facial_area']['y']
                        w = face['facial_area']['w']
                        h = face['facial_area']['h']
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            except:
                pass
            
            cv2.putText(display_frame, f"Rasmlar: {count}/{total_images}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE - rasm olish, Q - chiqish", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f"Yangi odam qo'shish: {name}", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE bosilganda
                img_path = os.path.join(person_path, f"{count+1}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1
                print(f"Rasm {count} saqlandi: {img_path}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n'{name}' uchun {count} ta rasm saqlandi.")
        return count
    
    def recognize_realtime(self):
        """Real vaqtda yuzni aniqlash va dataset bilan solishtirish"""
        
        # Datasetda odam bormi tekshirish
        persons = [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not persons:
            print("\nDataset bo'sh! Avval odam qo'shing.")
            print("Dasturni qayta ishga tushiring va '1' ni tanlang.\n")
            return
        
        print(f"\nDatasetdagi odamlar: {', '.join(persons)}")
        print("\nReal vaqtda aniqlash boshlandi...")
        print("Q - chiqish\n")
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        last_result = {}
        check_interval = 15  # Har 15 kadrda bir marta tekshirish
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            frame_count += 1
            
            # Har check_interval kadrda yuzni tekshirish (tezlik uchun)
            if frame_count % check_interval == 0:
                try:
                    # Datasetdan qidirish
                    results = DeepFace.find(
                        img_path=frame,
                        db_path=self.dataset_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(results) > 0 and len(results[0]) > 0:
                        df = results[0]
                        if len(df) > 0:
                            # Eng yaqin natija
                            best_match = df.iloc[0]
                            identity_path = best_match['identity']
                            distance = best_match['distance']
                            
                            # Papka nomidan ism olish
                            name = os.path.basename(os.path.dirname(identity_path))
                            
                            if distance < self.distance_threshold:
                                confidence = (1 - distance) * 100
                                last_result = {
                                    'name': name,
                                    'confidence': confidence,
                                    'found': True,
                                    'distance': distance
                                }
                            else:
                                last_result = {'found': False, 'name': 'Notanish'}
                        else:
                            last_result = {'found': False, 'name': 'Notanish'}
                    else:
                        last_result = {'found': False, 'name': 'Yuz topilmadi'}
                        
                except Exception as e:
                    last_result = {'found': False, 'name': 'Yuz topilmadi'}
            
            # Natijani ekranga chiqarish
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                for face in faces:
                    if face['confidence'] > 0.9:
                        x = face['facial_area']['x']
                        y = face['facial_area']['y']
                        w = face['facial_area']['w']
                        h = face['facial_area']['h']
                        
                        if last_result.get('found', False):
                            color = (0, 255, 0)  # Yashil - topildi
                            name = last_result['name']
                            conf = last_result.get('confidence', 0)
                            text = f"{name} ({conf:.1f}%)"
                        else:
                            color = (0, 0, 255)  # Qizil - notanish
                            text = last_result.get('name', 'Notanish')
                        
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Ism yozish uchun fon
                        cv2.rectangle(display_frame, (x, y-35), (x+w, y), color, -1)
                        cv2.putText(display_frame, text, (x+5, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                pass
            
            # Qo'shimcha ma'lumotlar
            cv2.putText(display_frame, "Q - chiqish", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Dataset: {len(persons)} odam", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Yuz aniqlash - DeepFace", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def list_persons(self):
        """Datasetdagi odamlar ro'yxati"""
        persons = []
        for name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, name)
            if os.path.isdir(person_path):
                images = len([f for f in os.listdir(person_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
                persons.append((name, images))
        
        if persons:
            print("\n=== Datasetdagi odamlar ===")
            for name, count in persons:
                print(f"  {name}: {count} ta rasm")
        else:
            print("\nDataset bo'sh!")
        
        return persons
    
    def delete_person(self, name):
        """Odamni datasetdan o'chirish"""
        import shutil
        person_path = os.path.join(self.dataset_path, name)
        
        if os.path.exists(person_path):
            shutil.rmtree(person_path)
            # representations faylini ham o'chirish (cache)
            for f in os.listdir(self.dataset_path):
                if f.startswith('representations') and f.endswith('.pkl'):
                    os.remove(os.path.join(self.dataset_path, f))
            print(f"'{name}' o'chirildi.")
            return True
        else:
            print(f"'{name}' topilmadi.")
            return False


def main():
    print("=" * 50)
    print("   YUZNI ANIQLASH TIZIMI - DeepFace")
    print("=" * 50)
    
    # Dataset papkasini ko'rsatish
    dataset_path = "./dataset"
    recognizer = FaceRecognizer(dataset_path)
    
    while True:
        print("\n" + "=" * 40)
        print("MENYU:")
        print("1. Yangi odam qo'shish (kameradan)")
        print("2. Real vaqtda yuzni aniqlash")
        print("3. Datasetdagi odamlar ro'yxati")
        print("4. Odamni o'chirish")
        print("5. Chiqish")
        print("=" * 40)
        
        tanlov = input("\nTanlovingiz (1-5): ").strip()
        
        if tanlov == "1":
            name = input("Yangi odamning ismi: ").strip()
            if name:
                recognizer.add_person(name)
            else:
                print("Ism kiritilmadi!")
        
        elif tanlov == "2":
            recognizer.recognize_realtime()
        
        elif tanlov == "3":
            recognizer.list_persons()
        
        elif tanlov == "4":
            recognizer.list_persons()
            name = input("\nO'chiriladigan ism: ").strip()
            if name:
                confirm = input(f"'{name}' ni o'chirishni tasdiqlaysizmi? (ha/yo'q): ")
                if confirm.lower() == 'ha':
                    recognizer.delete_person(name)
        
        elif tanlov == "5":
            print("\nDastur tugatildi.")
            break
        
        else:
            print("Noto'g'ri tanlov!")


if __name__ == "__main__":
    main()
