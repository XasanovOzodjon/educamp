"""
QO'L KO'TARISH ANIQLASH MODULI
Fayl: hand_raise_detection_module.py
Versiya: 3.0
"""

import numpy as np
from collections import deque


class HandRaiseDetector:
    """Qo'l ko'tarish aniqlagichi"""
    
    def __init__(self, pose_model=None, history_length=5, min_confidence=3):
        """
        Args:
            pose_model: YOLO pose model (ixtiyoriy)
            history_length: Tarix uzunligi (frame)
            min_confidence: Minimal tasdiqlash soni
        """
        self.pose_model = pose_model
        self.hand_raise_history = {}
        self.history_length = history_length
        self.min_confidence = min_confidence
        self.debug_mode = False
    
    def is_hand_raised(self, keypoints, box=None):
        """Qo'l ko'tarilganligini aniqlash"""
        try:
            if len(keypoints) < 11:
                return False, "Keypoints yetarli emas"
            
            # Keypoints olish
            nose = keypoints[0][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            
            # Confidence tekshirish
            left_wrist_conf = keypoints[9][2]
            right_wrist_conf = keypoints[10][2]
            left_elbow_conf = keypoints[7][2]
            right_elbow_conf = keypoints[8][2]
            
            # Elkalar o'rtasi
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            
            # Bosh darajasi (elkadan yuqorida)
            head_level = shoulder_y - shoulder_width * 0.5
            
            # CHAP QO'L TAHLILI
            left_raised = False
            left_reason = ""
            
            if left_wrist_conf > 0.2:
                # 1. Bilek elkadan yuqori
                if left_wrist[1] < shoulder_y:
                    left_raised = True
                    left_reason = "Bilek elkadan yuqori"
                
                # 2. Bilek bosh darajasiga yaqin
                elif left_wrist[1] < head_level + 50:
                    left_raised = True
                    left_reason = "Bilek boshga yaqin"
                
                # 3. Tirsak yuqori ko'tarilgan
                elif left_elbow_conf > 0.2 and left_elbow[1] < shoulder_y - 30:
                    left_raised = True
                    left_reason = "Tirsak yuqori"
            
            # O'NG QO'L TAHLILI
            right_raised = False
            right_reason = ""
            
            if right_wrist_conf > 0.2:
                # 1. Bilek elkadan yuqori
                if right_wrist[1] < shoulder_y:
                    right_raised = True
                    right_reason = "Bilek elkadan yuqori"
                
                # 2. Bilek bosh darajasiga yaqin
                elif right_wrist[1] < head_level + 50:
                    right_raised = True
                    right_reason = "Bilek boshga yaqin"
                
                # 3. Tirsak yuqori ko'tarilgan
                elif right_elbow_conf > 0.2 and right_elbow[1] < shoulder_y - 30:
                    right_raised = True
                    right_reason = "Tirsak yuqori"
            
            # Natija
            is_raised = left_raised or right_raised
            
            # Status teksti
            no_text = "Yo'q"
            left_status = left_reason if left_raised else no_text
            right_status = right_reason if right_raised else no_text
            reason = f"CHAP: {left_status} | O'NG: {right_status}"
            
            return is_raised, reason
            
        except Exception as e:
            return False, f"Xato: {str(e)}"
    
    def update_person_status(self, person_id, hand_raised):
        """Shaxsning qo'l ko'tarish holatini yangilash"""
        if person_id not in self.hand_raise_history:
            self.hand_raise_history[person_id] = deque(maxlen=self.history_length)
        
        self.hand_raise_history[person_id].append(hand_raised)
        
        # Oxirgi N frameda M tasi qo'l ko'targan bo'lsa - barqaror
        recent_count = sum(self.hand_raise_history[person_id])
        is_stable = recent_count >= self.min_confidence
        
        return is_stable
    
    def clean_old_persons(self, current_ids):
        """Eski shaxlarni tarixdan o'chirish"""
        to_remove = [pid for pid in self.hand_raise_history if pid not in current_ids]
        for pid in to_remove:
            del self.hand_raise_history[pid]
    
    def get_statistics(self):
        """Qo'l ko'tarish statistikasini olish"""
        stats = {
            'total_persons': len(self.hand_raise_history),
            'hand_raised_now': 0,
            'hand_down_now': 0
        }
        
        for person_id, history in self.hand_raise_history.items():
            recent_count = sum(history)
            if recent_count >= self.min_confidence:
                stats['hand_raised_now'] += 1
            else:
                stats['hand_down_now'] += 1
        
        return stats
    
    def reset_person(self, person_id):
        """Shaxsning tarixini tozalash"""
        if person_id in self.hand_raise_history:
            del self.hand_raise_history[person_id]
    
    def reset_all(self):
        """Barcha tarixni tozalash"""
        self.hand_raise_history.clear()
    
    def adjust_sensitivity(self, sensitivity='medium'):
        """
        Sezgirlikni sozlash
        
        Args:
            sensitivity: 'low', 'medium', 'high'
        """
        if sensitivity == 'low':
            self.min_confidence = 4  # Ko'proq frame kerak
            self.history_length = 7
        elif sensitivity == 'medium':
            self.min_confidence = 3
            self.history_length = 5
        elif sensitivity == 'high':
            self.min_confidence = 2  # Kam frame yetarli
            self.history_length = 4
        else:
            print("⚠️ Noto'g'ri sensitivity. 'low', 'medium', yoki 'high' tanlang.")
        
        # Mavjud tarixlarni yangi uzunlikka o'zgartirish
        for person_id in self.hand_raise_history:
            old_history = list(self.hand_raise_history[person_id])
            self.hand_raise_history[person_id] = deque(
                old_history[-self.history_length:], 
                maxlen=self.history_length
            )
    
    def enable_debug(self):
        """Debug rejimini yoqish"""
        self.debug_mode = True
        print("🐛 Debug rejimi yoqildi")
    
    def disable_debug(self):
        """Debug rejimini o'chirish"""
        self.debug_mode = False
        print("✓ Debug rejimi o'chirildi")


# Test funksiyasi
if __name__ == "__main__":
    print("Hand Raise Detection Module - Test Mode")
    detector = HandRaiseDetector()
    
    # Test keypoints (mock data - qo'l ko'tarilgan pozitsiya)
    test_keypoints = np.array([
        [320, 240, 0.9],  # 0: nose
        [300, 220, 0.9],  # 1: left_eye
        [340, 220, 0.9],  # 2: right_eye
        [280, 200, 0.9],  # 3: left_ear
        [360, 200, 0.9],  # 4: right_ear
        [260, 280, 0.9],  # 5: left_shoulder
        [380, 280, 0.9],  # 6: right_shoulder
        [250, 240, 0.9],  # 7: left_elbow (yuqorida)
        [390, 240, 0.9],  # 8: right_elbow (yuqorida)
        [240, 180, 0.9],  # 9: left_wrist (bosh darajasida)
        [400, 180, 0.9],  # 10: right_wrist (bosh darajasida)
    ])
    
    is_raised, reason = detector.is_hand_raised(test_keypoints)
    print(f"Qo'l ko'tarilganmi: {is_raised}")
    print(f"Sabab: {reason}")
    
    # Status update
    stable = detector.update_person_status(person_id=1, hand_raised=is_raised)
    print(f"Barqaror holat: {stable}")
    
    # Bir necha marta update qilish (test)
    for i in range(5):
        stable = detector.update_person_status(person_id=1, hand_raised=True)
        print(f"  Update {i+1}: {stable}")
    
    # Statistika
    stats = detector.get_statistics()
    print(f"Statistika: {stats}")
    
    # Sezgirlikni sozlash
    print("\n--- Sezgirlikni sozlash ---")
    detector.adjust_sensitivity('high')
    print("Sezgirlik 'high' ga o'rnatildi")
    
    # Debug rejimi
    detector.enable_debug()
