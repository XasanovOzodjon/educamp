"""
TELEFON ANIQLASH MODULI
Fayl: phone_detection_module.py
Versiya: 3.0
"""

import numpy as np
from collections import deque, defaultdict


class PhoneDetector:
    """Telefon ishlatishni aniqlash"""
    
    def __init__(self, history_length=15, min_confidence=10):
        """
        Args:
            history_length: Tarix uzunligi (frame)
            min_confidence: Minimal tasdiqlash soni (phone ishlatayapti deyish uchun)
        """
        self.phone_history = defaultdict(lambda: deque(maxlen=history_length))
        self.history_length = history_length
        self.min_confidence = min_confidence
        
        # Parametrlar
        self.HANDS_DISTANCE_THRESHOLD = 200  # Qo'llar orasidagi maksimal masofa
        self.FACE_DISTANCE_THRESHOLD = 100   # Qo'llarning yuzdan maksimal masofasi
    
    def detect_phone_usage(self, keypoints, frame=None, bbox=None):
        """Telefon ishlatishni aniqlash"""
        try:
            # Zarur keypoints mavjudligini tekshirish
            if len(keypoints) < 11:
                return False, "Keypoints yetarli emas"
            
            # Keypoints olish
            nose = keypoints[0][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            
            # Confidence tekshirish
            left_wrist_conf = keypoints[9][2]
            right_wrist_conf = keypoints[10][2]
            
            if left_wrist_conf < 0.3 or right_wrist_conf < 0.3:
                return False, "Qo'llar ko'rinmayapti"
            
            # 1. QO'LLAR ORASIDAGI MASOFA
            hands_distance = np.sqrt(
                (left_wrist[0] - right_wrist[0])**2 + 
                (left_wrist[1] - right_wrist[1])**2
            )
            
            # 2. QO'LLARNING YUZGA NISBATAN POZITSIYASI
            left_wrist_to_nose = abs(left_wrist[1] - nose[1])
            right_wrist_to_nose = abs(right_wrist[1] - nose[1])
            
            hands_near_face = (
                left_wrist_to_nose < self.FACE_DISTANCE_THRESHOLD and 
                right_wrist_to_nose < self.FACE_DISTANCE_THRESHOLD
            )
            
            # 3. QO'LLAR YAQIN MI?
            hands_close = hands_distance < self.HANDS_DISTANCE_THRESHOLD
            
            # 4. QO'LLAR YUZ OLDIDA MI?
            hands_in_front = (
                left_wrist[0] > nose[0] - 150 and 
                left_wrist[0] < nose[0] + 150 and
                right_wrist[0] > nose[0] - 150 and 
                right_wrist[0] < nose[0] + 150
            )
            
            # 5. TELEFON DIAGNOZI
            phone_indicators = 0
            reasons = []
            
            if hands_near_face:
                phone_indicators += 2
                reasons.append("Qo'llar yuz darajasida")
            
            if hands_close:
                phone_indicators += 2
                reasons.append("Qo'llar yaqin")
            
            if hands_in_front:
                phone_indicators += 1
                reasons.append("Qo'llar yuz oldida")
            
            # Kamida 3 ta belgi bo'lsa - telefon ishlatayapti
            likely_phone = phone_indicators >= 3
            
            if likely_phone:
                reason = " + ".join(reasons)
                return True, f"TELEFON: {reason}"
            else:
                return False, "Telefon yo'q"
            
        except Exception as e:
            return False, f"Xato: {e}"
    
    def update_phone_status(self, person_id, using_phone):
        """Telefon holati yangilash"""
        self.phone_history[person_id].append(using_phone)
        
        # Oxirgi N frameda M tasi telefon ishlatayotgan bo'lsa
        recent_phone_count = sum(self.phone_history[person_id])
        is_stable_phone = recent_phone_count >= self.min_confidence
        
        return is_stable_phone
    
    def get_statistics(self):
        """Telefon statistikasini olish"""
        stats = {
            'total_persons': len(self.phone_history),
            'using_phone_now': 0,
            'not_using_now': 0
        }
        
        for person_id, history in self.phone_history.items():
            recent_count = sum(history)
            if recent_count >= self.min_confidence:
                stats['using_phone_now'] += 1
            else:
                stats['not_using_now'] += 1
        
        return stats
    
    def reset_person(self, person_id):
        """Shaxsning tarixini tozalash"""
        if person_id in self.phone_history:
            del self.phone_history[person_id]
    
    def reset_all(self):
        """Barcha tarixni tozalash"""
        self.phone_history.clear()
    
    def adjust_sensitivity(self, sensitivity='medium'):
        """
        Sezgirlikni sozlash
        
        Args:
            sensitivity: 'low', 'medium', 'high'
        """
        if sensitivity == 'low':
            self.min_confidence = 12
            self.HANDS_DISTANCE_THRESHOLD = 150
            self.FACE_DISTANCE_THRESHOLD = 80
        elif sensitivity == 'medium':
            self.min_confidence = 10
            self.HANDS_DISTANCE_THRESHOLD = 200
            self.FACE_DISTANCE_THRESHOLD = 100
        elif sensitivity == 'high':
            self.min_confidence = 8
            self.HANDS_DISTANCE_THRESHOLD = 250
            self.FACE_DISTANCE_THRESHOLD = 120
        else:
            print("⚠️ Noto'g'ri sensitivity. 'low', 'medium', yoki 'high' tanlang.")


# Test funksiyasi
if __name__ == "__main__":
    print("Phone Detection Module - Test Mode")
    detector = PhoneDetector()
    
    # Test keypoints (mock data - telefon ishlatayotgan pozitsiya)
    test_keypoints = np.array([
        [320, 240, 0.9],  # 0: nose
        [300, 220, 0.9],  # 1: left_eye
        [340, 220, 0.9],  # 2: right_eye
        [280, 200, 0.9],  # 3: left_ear
        [360, 200, 0.9],  # 4: right_ear
        [260, 280, 0.9],  # 5: left_shoulder
        [380, 280, 0.9],  # 6: right_shoulder
        [250, 320, 0.9],  # 7: left_elbow
        [390, 320, 0.9],  # 8: right_elbow
        [280, 260, 0.9],  # 9: left_wrist (yuz oldida)
        [360, 260, 0.9],  # 10: right_wrist (yuz oldida)
    ])
    
    using_phone, reason = detector.detect_phone_usage(test_keypoints)
    print(f"Telefon ishlatayaptimi: {using_phone}")
    print(f"Sabab: {reason}")
    
    # Status update
    stable = detector.update_phone_status(person_id=1, using_phone=using_phone)
    print(f"Barqaror holat: {stable}")
    
    # Statistika
    stats = detector.get_statistics()
    print(f"Statistika: {stats}")
    
    # Sezgirlikni sozlash
    print("\n--- Sezgirlikni sozlash ---")
    detector.adjust_sensitivity('high')
    print("Sezgirlik 'high' ga o'rnatildi")
