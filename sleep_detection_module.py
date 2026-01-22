"""
UYQU ANIQLASH MODULI
Fayl: sleep_detection_module.py
Versiya: 3.0 - YAXSHILANGAN
"""

import numpy as np
from collections import deque, defaultdict


class SleepDetector:
    """Uyqu holatini aniqlash - YAXSHILANGAN VERSIYA 2.0"""
    
    def __init__(self, debug_mode=False):
        self.sleep_history = defaultdict(lambda: deque(maxlen=30))  # 30 frame tarix
        self.eye_aspect_ratio_history = defaultdict(lambda: deque(maxlen=10))
        self.head_tilt_history = defaultdict(lambda: deque(maxlen=10))
        
        # BALANSLI PARAMETRLAR - FALSE POSITIVE KAMAYTIRILDI
        self.EAR_THRESHOLD = 0.25
        self.HEAD_TILT_THRESHOLD = 20  # Bosh egilish chegarasi
        self.SLEEP_CONFIRM_FRAMES = 15  # Tasdiqlash framelar
        self.MIN_SLEEP_INDICATORS = 4   # QAYTA 4 ga - false positive kamaytirish
        
        # Debug rejimi
        self.debug_mode = debug_mode
        if debug_mode:
            print("🐛 UYQU ANIQLASH DEBUG REJIMI YOQILDI")
            print(f"   EAR Threshold: {self.EAR_THRESHOLD}")
            print(f"   Head Tilt Threshold: {self.HEAD_TILT_THRESHOLD}°")
            print(f"   Confirm Frames: {self.SLEEP_CONFIRM_FRAMES}")
            print(f"   Min Sleep Indicators: {self.MIN_SLEEP_INDICATORS}")
    
    def calculate_head_tilt(self, keypoints):
        """Boshning pastga egilish burchagini hisoblash"""
        try:
            # Keypoints: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear
            nose = keypoints[0][:2]
            left_eye = keypoints[1][:2]
            right_eye = keypoints[2][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]
            
            # Ko'zlar o'rtasi
            eye_center = [(left_eye[0] + right_eye[0]) / 2, 
                         (left_eye[1] + right_eye[1]) / 2]
            
            # Quloqlar o'rtasi
            ear_center = [(left_ear[0] + right_ear[0]) / 2,
                         (left_ear[1] + right_ear[1]) / 2]
            
            # Yuz o'qi (quloqdan ko'zga)
            face_vector_y = eye_center[1] - ear_center[1]
            face_vector_x = eye_center[0] - ear_center[0]
            
            # Vertikal burchak (0 = tik, 90 = yotiq)
            angle = np.degrees(np.arctan2(face_vector_y, abs(face_vector_x) + 0.001))
            
            return angle
        except:
            return 0
    
    def detect_sleep(self, keypoints):
        """Uyquni aniqlash - BALANSLI VERSIYA + Bosh stolda uxlash"""
        try:
            # Zarur keypoints mavjudligini tekshirish
            if len(keypoints) < 5:
                return False, "Yetarli keypoints yo'q"
            
            nose = keypoints[0][:2]
            left_eye = keypoints[1][:2]
            right_eye = keypoints[2][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]
            
            # Confidence tekshirish
            nose_conf = keypoints[0][2]
            left_eye_conf = keypoints[1][2]
            right_eye_conf = keypoints[2][2]
            left_ear_conf = keypoints[3][2]
            right_ear_conf = keypoints[4][2]  # [2] o'rniga
            
            # MAXSUS: Agar keypoints juda past confidence bo'lsa - bu bosh stolda uxlayotgan bo'lishi mumkin
            low_confidence_count = sum([
                1 for conf in [nose_conf, left_eye_conf, right_eye_conf, left_ear_conf, right_ear_conf]
                if conf < 0.3
            ])
            
            # STRICT: Agar 4+ keypoints past confidence bo'lsa VA qo'llar JUDA yaqin bo'lsa
            if low_confidence_count >= 4:  # 3 dan 4 ga - qattiqroq
                # Qo'llarni tekshirish (agar keypoints 11+ bo'lsa)
                if len(keypoints) >= 11:
                    left_wrist = keypoints[9][:2]
                    right_wrist = keypoints[10][:2]
                    left_wrist_conf = keypoints[9][2]
                    right_wrist_conf = keypoints[10][2]
                    
                    # QATTIQ SHART: Qo'llar yuqori confidence VA yuz JUDA yaqinida
                    if left_wrist_conf > 0.5 or right_wrist_conf > 0.5:  # 0.4 dan 0.5 ga
                        wrist_touching_face = False
                        
                        # CHAP QO'L
                        if left_wrist_conf > 0.5:
                            dist_to_nose = np.sqrt((left_wrist[0] - nose[0])**2 + (left_wrist[1] - nose[1])**2)
                            # Qo'l JUDA yaqin (50px ichida, 100 emas!)
                            if dist_to_nose < 50:
                                wrist_touching_face = True
                        
                        # O'NG QO'L
                        if right_wrist_conf > 0.5:
                            dist_to_nose = np.sqrt((right_wrist[0] - nose[0])**2 + (right_wrist[1] - nose[1])**2)
                            if dist_to_nose < 50:
                                wrist_touching_face = True
                        
                        # Faqat qo'l yuzga tegib tursa
                        if wrist_touching_face:
                            if self.debug_mode:
                                print(f"\n🐛 MAXSUS: Bosh stolda/qo'lda aniqlandi!")
                            return True, "UXLAYAPTI (100%): Bosh qo'lda/stolda"
            
            # Agar yuz keypoints aniq bo'lmasa - xato qaytarish
            if (left_eye_conf < 0.3 or right_eye_conf < 0.3 or 
                left_ear_conf < 0.25 or right_ear_conf < 0.25):
                return False, "Yuz aniq ko'rinmayapti"
            
            # 1. KO'Z HOLATINI TEKSHIRISH
            eye_distance = np.sqrt((left_eye[0] - right_eye[0])**2 + 
                                  (left_eye[1] - right_eye[1])**2)
            
            # Burun va ko'zlar orasidagi vertikal masofa
            nose_to_eye_dist = abs(nose[1] - (left_eye[1] + right_eye[1]) / 2)
            
            # Soddalashtirilgan EAR
            estimated_ear = nose_to_eye_dist / (eye_distance + 0.001)
            
            # 2. BOSH EGILISHI
            head_angle = self.calculate_head_tilt(keypoints)
            
            # 3. UYQU BELGILARI (STRICT - faqat haqiqiy uyqu)
            sleep_indicators = 0
            reasons = []
            
            # BELGI 1: Ko'zlar ANIQ burundan pastda (STRICT)
            eyes_below_nose = ((left_eye[1] + right_eye[1]) / 2) > nose[1] + 10
            if eyes_below_nose:
                sleep_indicators += 2
                reasons.append("Ko'zlar aniq pastda")
            
            # BELGI 2: Bosh sezilarli egilgan (STRICT threshold)
            head_tilted_down = head_angle > self.HEAD_TILT_THRESHOLD  # 20 gradus
            if head_tilted_down:
                sleep_indicators += 3
                reasons.append(f"Bosh egilgan ({int(head_angle)}°)")
            
            # BELGI 3: Bosh JUDA egilgan (haqiqiy uyqu belgisi)
            head_very_tilted = head_angle > 40
            if head_very_tilted:
                sleep_indicators += 5
                reasons.append(f"Bosh juda egilgan ({int(head_angle)}°)")
            
            # BELGI 4: Quloqlar aniq yuqorida
            ears_high = ((left_ear[1] + right_ear[1]) / 2) < ((left_eye[1] + right_eye[1]) / 2) - 25
            if ears_high:
                sleep_indicators += 2
                reasons.append("Bosh pastga qaragan")
            
            # BELGI 5: Ko'zlar JUDA yopiq (STRICT)
            if estimated_ear < 0.15:
                sleep_indicators += 3
                reasons.append("Ko'zlar yopiq")
            
            # BELGI 6: MAXSUS - Burun juda pastda (bosh stolda holat) - QATTIQ SHART
            # Faqat agar burun JUDA pastda bo'lsa (50px+)
            if nose[1] > (left_eye[1] + right_eye[1]) / 2 + 50:  # 30 o'rniga 50
                sleep_indicators += 3
                reasons.append("Burun juda pastda (stolda)")
            
            # STRICT THRESHOLD: Kamida 4 ta ball kerak
            is_sleeping = sleep_indicators >= self.MIN_SLEEP_INDICATORS
            
            # DEBUG rejimi
            if self.debug_mode:
                print(f"\n🐛 DEBUG:")
                print(f"   Head angle: {head_angle:.1f}°")
                print(f"   EAR: {estimated_ear:.3f}")
                print(f"   Sleep indicators: {sleep_indicators}/{self.MIN_SLEEP_INDICATORS}")
                print(f"   Low conf keypoints: {low_confidence_count}")
                print(f"   Is sleeping: {is_sleeping}")
                if reasons:
                    print(f"   Reasons: {', '.join(reasons)}")
            
            if is_sleeping:
                reason = " + ".join(reasons)
                confidence_score = min(100, sleep_indicators * 15)
                return True, f"UXLAYAPTI ({confidence_score}%): {reason}"
            else:
                return False, "Hushyor"
            
        except Exception as e:
            return False, f"Xato: {e}"
    
    def update_sleep_status(self, person_id, is_sleeping):
        """Uyqu holatini yangilash va stabillashtirish - BALANSLI"""
        self.sleep_history[person_id].append(is_sleeping)
        
        # BALANSLI THRESHOLD: Oxirgi 30 frameda 15 tasi uxlayotgan bo'lsa - aniq uxlayapti
        # (8 dan 15 ga oshirildi - false positive kamayadi)
        recent_sleep_count = sum(self.sleep_history[person_id])
        is_stable_sleeping = recent_sleep_count >= self.SLEEP_CONFIRM_FRAMES
        
        return is_stable_sleeping
    
    def get_statistics(self):
        """Uyqu statistikasini olish"""
        stats = {
            'total_persons': len(self.sleep_history),
            'sleeping_now': 0,
            'awake_now': 0
        }
        
        for person_id, history in self.sleep_history.items():
            recent_count = sum(history)
            if recent_count >= 12:
                stats['sleeping_now'] += 1
            else:
                stats['awake_now'] += 1
        
        return stats
    
    def reset_person(self, person_id):
        """Shaxsning tarixini tozalash"""
        if person_id in self.sleep_history:
            del self.sleep_history[person_id]
    
    def reset_all(self):
        """Barcha tarixni tozalash"""
        self.sleep_history.clear()
        self.eye_aspect_ratio_history.clear()
        self.head_tilt_history.clear()


# Test funksiyasi
if __name__ == "__main__":
    print("Sleep Detection Module - Test Mode")
    detector = SleepDetector()
    
    # Test keypoints (mock data)
    test_keypoints = np.array([
        [320, 240, 0.9],  # nose
        [300, 220, 0.9],  # left_eye
        [340, 220, 0.9],  # right_eye
        [280, 200, 0.9],  # left_ear
        [360, 200, 0.9],  # right_ear
    ])
    
    is_sleeping, reason = detector.detect_sleep(test_keypoints)
    print(f"Uxlayaptimi: {is_sleeping}")
    print(f"Sabab: {reason}")
    
    # Status update
    stable = detector.update_sleep_status(person_id=1, is_sleeping=is_sleeping)
    print(f"Barqaror holat: {stable}")
    
    # Statistika
    stats = detector.get_statistics()
    print(f"Statistika: {stats}")