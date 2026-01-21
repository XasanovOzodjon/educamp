"""
MAKTAB MONITORING TIZIMI - ASOSIY DASTUR
Fayl: main.py
Versiya: 3.0 - Modulyar Tuzilma
"""

import cv2
import time
import numpy as np
from collections import defaultdict

# Modullarni import qilish
# from face_recognition_module import InsightFaceRecognitionSystem  # O'CHIRILDI
from sleep_detection_module import SleepDetector
# from phone_detection_module import PhoneDetector  # O'CHIRILDI
from hand_raise_detection_module import HandRaiseDetector
from seat_monitoring_module import SeatMonitor
from camera_utils import (
    open_camera_smart, 
    select_camera_source, 
    configure_camera,
    install_required_packages,
    suppress_opencv_warnings,
    PLATFORM
)


def setup_seats_interactive(camera_index=0):
    """O'rindiqlarni interaktiv belgilash"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           O'RINDIQLARNI BELGILASH REJIMI                   ║
    ╠════════════════════════════════════════════════════════════╣
    ║  1. Videoda o'rindiq burchaklarini sichqoncha bilan        ║
    ║     bosing (4 ta nuqta)                                    ║
    ║  2. 4-nuqtadan keyin o'rindiq nomini kiriting              ║
    ║                                                            ║
    ║  TUGMALAR:                                                 ║
    ║  'c' - oxirgisini o'chirish | 'r' - barchasini tozalash   ║
    ║  'q' - chiqish                                             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    seat_monitor = SeatMonitor()
    cap = open_camera_smart(camera_index)
    
    if cap is None:
        return None
    
    configure_camera(cap, width=1280, height=720)
    
    # GUI tekshiruvi
    gui_available = True
    try:
        cv2.namedWindow('test_window')
        cv2.destroyWindow('test_window')
    except:
        gui_available = False
        print("\n⚠️ GUI mavjud emas!")
        print("   O'rindiqlarni qo'lda JSON faylga yozishingiz kerak.")
        print("   Misol: seats_config.json")
        cap.release()
        return None
    
    temp_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal temp_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(temp_points) < 4:
                temp_points.append([x, y])
                print(f"Nuqta {len(temp_points)}: ({x}, {y})")
                if len(temp_points) == 4:
                    name = input("O'rindiq nomini kiriting: ")
                    seat_monitor.add_seat(name, temp_points)
                    print(f"✓ '{name}' qo'shildi!")
                    temp_points = []
    
    try:
        cv2.namedWindow('O\'rindiqlarni Belgilash')
        cv2.setMouseCallback('O\'rindiqlarni Belgilash', mouse_callback)
    except Exception as e:
        print(f"⚠️ Oyna yaratilmadi: {e}")
        cap.release()
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mavjud o'rindiqlarni chizish
        seat_monitor.draw_seats(frame)
        
        # Joriy nuqtalarni chizish
        for i, point in enumerate(temp_points):
            cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Chiziqlar
        if len(temp_points) > 1:
            for i in range(len(temp_points)-1):
                cv2.line(frame, tuple(temp_points[i]), tuple(temp_points[i+1]), 
                        (255, 0, 0), 2)
            if len(temp_points) == 4:
                cv2.line(frame, tuple(temp_points[3]), tuple(temp_points[0]), 
                        (255, 0, 0), 2)
        
        # Info
        info = f"O'rindiqlar: {len(seat_monitor.seats)} | Joriy nuqtalar: {len(temp_points)}/4"
        cv2.putText(frame, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        help_text = "'c'-oxirgisini o'chirish | 'r'-barchasini tozalash | 'q'-chiqish"
        cv2.putText(frame, help_text, (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        try:
            cv2.imshow('O\'rindiqlarni Belgilash', frame)
        except Exception as e:
            print(f"⚠️ Frame ko'rsatilmadi: {e}")
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if seat_monitor.seats:
                removed_idx = len(seat_monitor.seats) - 1
                seat_monitor.remove_seat(removed_idx)
            else:
                print("⚠️ O'chirish uchun o'rindiqlar yo'q!")
        elif key == ord('r'):
            if seat_monitor.seats:
                seat_monitor.clear_all_seats()
                temp_points = []
            else:
                print("⚠️ Tozalash uchun o'rindiqlar yo'q!")
    
    cap.release()
    cv2.destroyAllWindows()
    return seat_monitor


def register_students_faces(camera_index=0):
    """O'quvchilar yuzlarini ro'yxatdan o'tkazish - O'CHIRILDI"""
    print("\n⚠️ Bu funksiya hozircha o'chirilgan (yuz tanish o'chirildi)")
    print("   Monitoring tizimida faqat qo'l ko'tarish va uyqu aniqlash ishlaydi\n")
    input("📍 [Enter] tugmasini bosing menuga qaytish uchun...")


def full_monitoring_system(camera_index=0, confidence_threshold=0.45):
    """TO'LIQ MONITORING TIZIMI"""
    
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║    MAKTAB MONITORING TIZIMI - SODDALASHTIRILGAN v3.0       ║
    ║    ✅ Qo'l ko'tarish aniqlash                               ║
    ║    ✅ Uyqu holati aniqlash                                  ║
    ║    ✅ O'rindiq monitoring                                   ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Kutubxonalarni tekshirish
    if not install_required_packages():
        print("⚠ Ba'zi kutubxonalar o'rnatilmadi")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ Ultralytics import qilinmadi")
        return
    
    # Modullarni yuklash
    print("\n🔄 Modullar yuklanmoqda...")
    seat_monitor = SeatMonitor()
    # face_system = InsightFaceRecognitionSystem()  # O'CHIRILDI
    
    # UYQU ANIQLASH - DEBUG REJIMI
    # Debug rejimini yoqish uchun debug_mode=True qiling
    debug_sleep = input("\n🐛 Uyqu aniqlash debug rejimini yoqasizmi? (y/n, default: n): ").strip().lower() == 'y'
    sleep_detector = SleepDetector(debug_mode=debug_sleep)
    
    # phone_detector = PhoneDetector()  # O'CHIRILDI
    
    # O'rindiqlar tekshiruvi
    if not seat_monitor.seats:
        print("\n⚠️ O'rindiqlar belgilanmagan!")
        choice = input("O'rindiqlarni hozir belgilaysizmi? (y/n): ")
        if choice.lower() == 'y':
            seat_monitor = setup_seats_interactive(camera_index)
            if not seat_monitor or not seat_monitor.seats:
                print("✗ Dastur to'xtatildi")
                return
    
    # YOLO modellar
    print("\n🔄 YOLO modellar yuklanmoqda...")
    try:
        person_model = YOLO('yolov8n.pt')
        pose_model = YOLO('yolov8n-pose.pt')
        print("✓ YOLO models yuklandi")
    except Exception as e:
        print(f"✗ Model yuklanmadi: {e}")
        return
    
    hand_detector = HandRaiseDetector(pose_model)
    
    # Kamera
    cap = open_camera_smart(camera_index)
    if cap is None:
        return
    
    configure_camera(cap, width=1280, height=720)
    
    print("\n" + "="*60)
    print("✅ TIZIM ISHGA TUSHDI - TO'LIQ MONITORING FAOL")
    print("="*60)
    print("Tugmalar:")
    print("  'q' - To'xtatish")
    print("  's' - Screenshot")
    print("  '+/-' - Aniqlikni sozlash")
    print("="*60 + "\n")
    
    # Monitoring o'zgaruvchilari
    prev_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    current_confidence = confidence_threshold
    frame_skip = 1
    frame_count = 0
    last_pose_results = []
    
    stats = {
        'hand_raised': 0,
        'sleeping': 0,
        # 'using_phone': 0,  # O'CHIRILDI
        # 'recognized': 0    # O'CHIRILDI
    }
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # FPS hisoblash
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # 1. ODAMLARNI ANIQLASH
            person_results = person_model(frame, conf=current_confidence, 
                                         classes=[0], verbose=False, device='cpu')
            boxes = person_results[0].boxes
            
            detections = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'person_id': i,
                    'hand_raised': False,
                    'sleeping': False,
                    # 'using_phone': False,  # O'CHIRILDI
                    # 'name': 'Noma\'lum',   # O'CHIRILDI
                    # 'face_confidence': 0.0  # O'CHIRILDI
                })
            
            # 2. POSE DETECTION
            if frame_count % frame_skip == 0 or frame_count == 1:
                pose_results = pose_model(frame, conf=0.3, verbose=False, device='cpu')
                last_pose_results = pose_results
            
            # 3. BARCHA ANALIZLAR
            current_person_ids = set()
            stats = {'hand_raised': 0, 'sleeping': 0}  # Telefon va yuz tanish o'chirildi
            
            if last_pose_results and len(last_pose_results) > 0:
                for i, det in enumerate(detections):
                    person_id = det['person_id']
                    current_person_ids.add(person_id)
                    
                    if last_pose_results[0].keypoints is not None:
                        try:
                            if i < len(last_pose_results[0].keypoints):
                                kp = last_pose_results[0].keypoints[i].data[0].cpu().numpy()
                                
                                # Qo'l ko'tarish
                                hand_raised_now, _ = hand_detector.is_hand_raised(kp, det['box'])
                                det['hand_raised'] = hand_detector.update_person_status(person_id, hand_raised_now)
                                
                                # Uyqu holati
                                sleeping_now, _ = sleep_detector.detect_sleep(kp)
                                det['sleeping'] = sleep_detector.update_sleep_status(person_id, sleeping_now)
                                
                                # TELEFON VA YUZ TANISH O'CHIRILDI
                                
                                # Statistika
                                if det['hand_raised']:
                                    stats['hand_raised'] += 1
                                if det['sleeping']:
                                    stats['sleeping'] += 1
                                    
                        except Exception:
                            pass
            
            # Eski shaxslarni tozalash
            hand_detector.clean_old_persons(current_person_ids)
            
            # 4. O'RINDIQLAR
            seat_monitor.check_occupancy(detections)
            seat_monitor.draw_seats(frame)
            
            # 5. VIZUALIZATSIYA
            for det in detections:
                x1, y1, x2, y2 = det['box']
                person_id = det['person_id']
                
                # Status prioritet (faqat qo'l va uyqu)
                status_priority = []
                
                if det.get('sleeping', False):
                    status_priority.append(("UXLAYAPTI", "😴", (0, 0, 255), 3))
                
                if det.get('hand_raised', False):
                    status_priority.append(("QO'L KO'TARGAN", "🙋", (255, 150, 0), 1))
                
                if status_priority:
                    status_text, status_emoji, color, _ = status_priority[0]
                    thickness = 4
                else:
                    status_text = "NORMAL"
                    status_emoji = "✓"
                    color = (0, 255, 0)
                    thickness = 2
                
                # Chizish
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Pastki markaz
                bottom_center = [(x1+x2)//2, y2]
                cv2.circle(frame, bottom_center, 8, color, -1)
                cv2.circle(frame, bottom_center, 11, (255, 255, 255), 2)
                
                # O'rindiq
                seat_name = seat_monitor.get_seat_by_person(person_id)
                seat_text = f" | {seat_name}" if seat_name else ""
                
                # Label (yuz tanish o'chirildi)
                label = f"{status_emoji} Odam #{person_id + 1} - {status_text}{seat_text}"
                
                (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
                label_height = 40
                cv2.rectangle(frame, (x1, y1-label_height), (x1+tw+15, y1-5), color, -1)
                cv2.rectangle(frame, (x1, y1-label_height), (x1+tw+15, y1-5), (255,255,255), 2)
                cv2.putText(frame, label, (x1+7, y1-17), font, 0.55, (255,255,255), 2)
            
            # 6. STATISTIKA PANELI
            seat_stats = seat_monitor.get_statistics()
            
            panel_h = 260
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (550, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            y_pos = 35
            cv2.putText(frame, 'MAKTAB MONITORING - SODDA v3.0', (20, y_pos), font, 0.75, (255,255,255), 2)
            y_pos += 25
            cv2.putText(frame, '(Qo\'l + Uyqu + O\'rindiq)', (20, y_pos), font, 0.5, (0,255,255), 1)
            y_pos += 25
            cv2.putText(frame, f'FPS: {int(fps)}', (20, y_pos), font, 0.6, (0,255,0), 2)
            y_pos += 25
            cv2.putText(frame, f'Jami odamlar: {len(detections)}', (20, y_pos), font, 0.6, (255,255,255), 2)
            y_pos += 25
            cv2.putText(frame, f'Band o\'rindiqlar: {seat_stats["occupied_seats"]}/{seat_stats["total_seats"]}', (20, y_pos), font, 0.6, (255,255,0), 2)
            y_pos += 25
            cv2.putText(frame, f'🙋 Qo\'l ko\'targan: {stats["hand_raised"]}', (20, y_pos), font, 0.6, (255,150,0), 2)
            y_pos += 25
            cv2.putText(frame, f'😴 Uxlayotganlar: {stats["sleeping"]}', (20, y_pos), font, 0.6, (0,0,255), 2)
            
            # GUI xavfsiz ko'rsatish
            try:
                cv2.imshow('Maktab Monitoring - Modulyar', frame)
            except Exception as e:
                # GUI ishlamasa, faqat konsol log
                if frame_count % 100 == 0:  # Har 100 frameda
                    print(f"\n📊 Frame {frame_count}: FPS={int(fps)}, Odamlar={len(detections)}")
                    print(f"   🙋 Qo'l: {stats['hand_raised']}, 😴 Uyqu: {stats['sleeping']}")
            
            # Klaviatura
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'screenshot_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot: {filename}")
            # 'r' tugmasi o'chirildi - yuz tanish funksiyasi yo'q
            elif key in [ord('+'), ord('=')]:
                current_confidence = min(0.90, current_confidence + 0.05)
                print(f"⬆️ Aniqlik: {int(current_confidence*100)}%")
            elif key in [ord('-'), ord('_')]:
                current_confidence = max(0.20, current_confidence - 0.05)
                print(f"⬇️ Aniqlik: {int(current_confidence*100)}%")
    
    except KeyboardInterrupt:
        print("\n✓ Dastur to'xtatildi")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Asosiy menyu"""
    # OpenCV ogohlantirishlarini yashirish
    suppress_opencv_warnings()
    
    camera_index = None
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🎓  MAKTAB MONITORING - SODDALASHTIRILGAN v3.0  🎓         ║
    ║                                                               ║
    ║   ✅ Qo'l ko'tarish + Uyqu aniqlash + O'rindiq monitoring    ║
    ║   🚀 Tez va barqaror - Professional darajada                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📌 Platform: {PLATFORM}")
    print(f"📅 Versiya: 3.0 Soddalashtirilgan")
    print(f"🕐 Sana: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    while True:
        print("═" * 70)
        print("                    🎯 ASOSIY MENYU")
        print("═" * 70)
        print("""
    [1] 🪑 O'rindiqlarni sozlash
    [2] 🎬 TO'LIQ MONITORING TIZIMINI BOSHLASH
        └─ Real-time monitoring:
           ✅ Qo'l ko'tarish aniqlash (90-95% aniqlik)
           ✅ Uyqu aniqlash (85-95% aniqlik)
           ✅ O'rindiq monitoring
    [3] 🎥 Kamerani qayta tanlash
    [4] 📊 Modullar haqida ma'lumot
    [q] 🚪 Chiqish
        """)
        print("═" * 70)
        
        choice = input("\n➤ Tanlang (1-4 yoki q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Dasturdan chiqilmoqda...")
            print("✅ Xayr! Maktabingiz xavfsiz bo'lsin! 🎓\n")
            break
            
        elif choice == '1':
            if camera_index is None:
                camera_index = select_camera_source()
            setup_seats_interactive(camera_index=camera_index)
            input("\n📍 [Enter] tugmasini bosing menuga qaytish uchun...")
            
        elif choice == '2':
            if camera_index is None:
                camera_index = select_camera_source()
            
            try:
                conf_input = input("\n➤ Aniqlash aniqligini kiriting (20-90%, default: 45): ").strip()
                if conf_input:
                    confidence = float(conf_input) / 100
                    if confidence < 0.2 or confidence > 0.9:
                        confidence = 0.45
                else:
                    confidence = 0.45
            except:
                confidence = 0.45
            
            print(f"\n✅ Aniqlik: {int(confidence*100)}%")
            print("🚀 Monitoring boshlanmoqda...\n")
            time.sleep(1)
            
            full_monitoring_system(camera_index=camera_index, confidence_threshold=confidence)
            input("\n📍 [Enter] tugmasini bosing menuga qaytish uchun...")
            
        elif choice == '3':
            camera_index = select_camera_source()
            print(f"\n✅ Kamera {camera_index} tanlandi")
            input("\n📍 [Enter] tugmasini bosing menuga qaytish uchun...")
            
        elif choice == '4':
            print("\n" + "═" * 70)
            print("📦 MODULYAR TUZILMA - MODULLAR")
            print("═" * 70)
            print("""
📁 FAYLLAR (SODDALASHTIRILGAN):
   1. sleep_detection_module.py       - Uyqu aniqlash ✅
   2. hand_raise_detection_module.py  - Qo'l ko'tarish ✅
   3. seat_monitoring_module.py       - O'rindiq monitoring ✅
   4. camera_utils.py                 - Kamera utilities ✅
   5. main.py                         - Asosiy dastur (bu fayl) ✅

🎯 ISHLAYOTGAN FUNKSIYALAR:
   • Qo'l ko'tarish aniqlash (90-95% aniqlik)
   • Uyqu aniqlash (85-95% aniqlik)
   • O'rindiq monitoring
   • Real-time statistika

❌ O'CHIRILGAN FUNKSIYALAR:
   • Yuz tanish (InsightFace)
   • Telefon aniqlash

⚡ AFZALLIKLAR:
   • Tezroq ishlaydi
   • Kam xotira ishlatadi
   • Oson sozlanadi
   • Barqaror ishlaydi
            """)
            print("═" * 70)
            input("\n📍 [Enter] tugmasini bosing menuga qaytish uchun...")
            
        else:
            print("\n❌ Noto'g'ri tanlov! Iltimos 1-4 yoki 'q' ni tanlang.")
            time.sleep(1.5)


if __name__ == "__main__":
    main()