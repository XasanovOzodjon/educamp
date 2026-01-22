"""
YUZ TANISH MODULI - InsightFace
Fayl: face_recognition_module.py
Versiya: 3.0
"""

import cv2
import numpy as np
import json
import os


class InsightFaceRecognitionSystem:
    """InsightFace va ONNXRuntime bilan PROFESSIONAL yuz tanish tizimi"""
    
    def __init__(self, database_file='students_faces_insightface.json'):
        self.database_file = database_file
        self.known_faces = {}  # {student_name: [embedding1, embedding2, ...]}
        self.model_available = False
        self.face_app = None
        self.threshold = 0.4  # Similarity threshold (past = o'xshash)
        
        # InsightFace modelni yuklash
        self._load_insightface_model()
        self.load_database()
    
    def _load_insightface_model(self):
        """InsightFace modelni yuklash"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            print("\n🔄 InsightFace model yuklanmoqda...")
            
            # FaceAnalysis appni yaratish
            self.face_app = FaceAnalysis(
                name='buffalo_l',  # Eng yaxshi model
                providers=['CPUExecutionProvider']  # CPU uchun
            )
            
            # Modelni tayyorlash
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.model_available = True
            print("✅ InsightFace model muvaffaqiyatli yuklandi!")
            print("   Model: buffalo_l (SOTA accuracy)")
            print("   Backend: ONNXRuntime")
            
        except ImportError:
            print("⚠️ InsightFace o'rnatilmagan!")
            print("   O'rnatish: pip install insightface onnxruntime")
            self.model_available = False
        except Exception as e:
            print(f"⚠️ InsightFace model yuklashda xato: {e}")
            self.model_available = False
    
    def load_database(self):
        """Yuz ma'lumotlar bazasini yuklash"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, embeddings in data.items():
                        self.known_faces[name] = [np.array(emb) for emb in embeddings]
                print(f"✓ {len(self.known_faces)} ta o'quvchi yuz ma'lumotlari yuklandi (InsightFace)")
            except Exception as e:
                print(f"⚠ Yuz ma'lumotlarini yuklashda xato: {e}")
    
    def save_database(self):
        """Yuz ma'lumotlar bazasini saqlash"""
        try:
            data = {}
            for name, embeddings in self.known_faces.items():
                data[name] = [emb.tolist() for emb in embeddings]
            
            with open(self.database_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ Yuz ma'lumotlari saqlandi ({len(self.known_faces)} kishi)")
        except Exception as e:
            print(f"✗ Saqlashda xato: {e}")
    
    def register_face(self, frame, bbox, student_name):
        """Yangi yuzni ro'yxatdan o'tkazish - InsightFace"""
        if not self.model_available:
            print("✗ InsightFace model mavjud emas!")
            return False
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Bbox kengaytirish
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                print("✗ Yuz kesimi bo'sh")
                return False
            
            # InsightFace bilan yuzlarni aniqlash
            faces = self.face_app.get(face_crop)
            
            if len(faces) == 0:
                print("✗ Yuz aniqlanmadi")
                return False
            
            # Eng katta yuzni olish
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Face embedding (512-dimensional vector)
            embedding = face.embedding
            
            if embedding is not None:
                if student_name not in self.known_faces:
                    self.known_faces[student_name] = []
                
                self.known_faces[student_name].append(embedding)
                self.save_database()
                print(f"✅ {student_name} ro'yxatdan o'tkazildi! (InsightFace)")
                print(f"   Embedding size: {embedding.shape}")
                return True
            else:
                print("✗ Face embedding olinmadi")
                return False
                
        except Exception as e:
            print(f"✗ Xato: {e}")
            return False
    
    def recognize_face(self, frame, bbox):
        """Yuzni tanish - InsightFace"""
        if not self.model_available:
            return "Model yo'q", 0.0
        
        if not self.known_faces:
            return "Noma'lum", 0.0
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Bbox kengaytirish
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return "Yuz topilmadi", 0.0
            
            # InsightFace bilan yuzni aniqlash
            faces = self.face_app.get(face_crop)
            
            if len(faces) == 0:
                return "Yuz aniqlanmadi", 0.0
            
            # Eng katta yuzni olish
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            test_embedding = face.embedding
            
            if test_embedding is None:
                return "Embedding yo'q", 0.0
            
            # Eng o'xshash yuzni topish (Cosine similarity)
            best_match_name = "Noma'lum"
            best_similarity = -1.0
            
            for name, known_embeddings_list in self.known_faces.items():
                for known_embedding in known_embeddings_list:
                    # Cosine similarity
                    similarity = np.dot(test_embedding, known_embedding) / (
                        np.linalg.norm(test_embedding) * np.linalg.norm(known_embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_name = name
            
            # Threshold tekshirish
            if best_similarity < self.threshold:
                return "Noma'lum", float(best_similarity * 100)
            
            # Confidence (0-100)
            confidence = float(best_similarity * 100)
            
            return best_match_name, confidence
            
        except Exception as e:
            print(f"⚠️ Tanishda xato: {e}")
            return "Xato", 0.0


# Test funksiyasi
if __name__ == "__main__":
    print("Face Recognition Module - Test Mode")
    face_system = InsightFaceRecognitionSystem()
    
    if face_system.model_available:
        print("✅ Model tayyor!")
    else:
        print("❌ Model yuklanmadi!")
