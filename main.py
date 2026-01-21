import cv2
import numpy as np
import time
import os
import json
import platform
from collections import deque

# OpenCV xato xabarlarini yashirish
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Platform aniqlash
PLATFORM = platform.system()

def get_camera_backend():
    """Platformaga mos camera backend ni qaytarish"""
    if PLATFORM == 'Windows':
        return cv2.CAP_DSHOW
    elif PLATFORM == 'Darwin':
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_V4L2

def install_ultralytics():
    """Ultralytics YOLO kutubxonasini o'rnatish"""
    try:
        import ultralytics
        print("✓ Ultralytics allaqachon o'rnatilgan")
        return True
    except ImportError:
        print("\nUltralytics YOLO o'rnatilmoqda...")
        import subprocess
        try:
            subprocess.check_call(['pip', 'install', 'ultralytics'])
            print("✓ Ultralytics muvaffaqiyatli o'rnatildi!")
            return True
        except:
            print("✗ O'rnatib bo'lmadi. Qo'lda o'rnating: pip install ultralytics")
            return False

def select_camera_source():
    """Kamera manbani tanlash"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              KAMERA MANBANI TANLANG                        ║
    ╠════════════════════════════════════════════════════════════╣
    ║  [1] Oddiy kamera (Laptop yoki USB kamera)                 ║
    ║  [2] OBS Virtual Camera                                    ║
    ║  [3] Mavjud kameralarni ko'rish                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        choice = input("Tanlang (1-3): ").strip()
        if choice == '1':
            return 0
        elif choice == '2':
            try:
                camera_index = int(input("Kamera indexini kiriting (odatda 1 yoki 2): "))
                return camera_index
            except:
                print("✗ Noto'g'ri format!")
        elif choice == '3':
            available, names = list_cameras()
            if available:
                try:
                    idx = int(input("\nQaysi kamerani ishlatmoqchisiz? Index: "))
                    if idx in available:
                        return idx
                except:
                    pass
        else:
            print("✗ Noto'g'ri tanlov!")

def list_cameras():
    """Mavjud kameralarni ko'rsatish"""
    import contextlib
    print("\n" + "="*60)
    print(f"MAVJUD KAMERALAR ({PLATFORM}):")
    print("="*60)
    
    available = []
    names = {}
    backend = get_camera_backend()
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            for i in range(10):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                        names[i] = f"Kamera {i}"
                        print(f"  [{i}] {names[i]}")
                    cap.release()
    
    if not available:
        print("  Hech qanday kamera topilmadi!")
    print("="*60 + "\n")
    return available, names


class HandRaiseDetector:
    """Qo'l ko'tarish aniqlagichi - YAXSHILANGAN versiya"""
    
    def __init__(self, pose_model):
        self.pose_model = pose_model
        self.hand_raise_history = {}  # {person_id: deque([True/False])}
        self.history_length = 5  # 5 frame tarix (10 dan kamaytirildi)
        self.min_confidence = 3  # Kamida 3 frame (5 dan kamaytirildi)
        self.debug_mode = False  # Debug rejimi
    
    def is_hand_raised(self, keypoints, box):
        """Qo'l ko'tarilganligini aniqlash - YUMSHATILGAN algoritm"""
        try:
            # YOLO Pose keypoints: 17 ta nuqta
            # 0: burun, 5: chap elka, 6: o'ng elka, 
            # 7: chap tirsak, 8: o'ng tirsak, 9: chap bilek, 10: o'ng bilek
            
            if len(keypoints) < 11:
                return False, "Keypoints yetarli emas"
            
            # Asosiy nuqtalar
            nose = keypoints[0][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            
            # Confidence'larni olish
            left_wrist_conf = keypoints[9][2]
            right_wrist_conf = keypoints[10][2]
            left_elbow_conf = keypoints[7][2]
            right_elbow_conf = keypoints[8][2]
            
            # Elka balandligi
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Bosh balandligi (elkadan yuqori)
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            head_level = shoulder_y - shoulder_width * 0.5  # Yumshatildi
            
            # CHAP QO'L tekshirish - YUMSHATILGAN
            left_raised = False
            left_reason = ""
            if left_wrist_conf > 0.2:  # 0.3 dan 0.2 ga tushirildi
                # 1. Bilek elkadan yuqorida
                if left_wrist[1] < shoulder_y:
                    left_raised = True
                    left_reason = "Bilek elkadan yuqori"
                    
                # 2. Yoki bilek boshga yaqin
                elif left_wrist[1] < head_level + 50:  # 50px yumshatma
                    left_raised = True
                    left_reason = "Bilek boshga yaqin"
                    
                # 3. Yoki tirsak juda yuqori
                elif left_elbow_conf > 0.2 and left_elbow[1] < shoulder_y - 30:
                    left_raised = True
                    left_reason = "Tirsak yuqori"
            
            # O'NG QO'L tekshirish - YUMSHATILGAN
            right_raised = False
            right_reason = ""
            if right_wrist_conf > 0.2:
                if right_wrist[1] < shoulder_y:
                    right_raised = True
                    right_reason = "Bilek elkadan yuqori"
                elif right_wrist[1] < head_level + 50:
                    right_raised = True
                    right_reason = "Bilek boshga yaqin"
                elif right_elbow_conf > 0.2 and right_elbow[1] < shoulder_y - 30:
                    right_raised = True
                    right_reason = "Tirsak yuqori"
            
            # Natija
            is_raised = left_raised or right_raised
            no_text = "Yo'q"
            left_status = left_reason if left_raised else no_text
            right_status = right_reason if right_raised else no_text
            reason = f"CHAP: {left_status} | O'NG: {right_status}"
            
            return is_raised, reason
            
        except Exception as e:
            return False, f"Xato: {str(e)}"
    
    def update_person_status(self, person_id, hand_raised):
        """Shaxsning qo'l ko'tarish holatini yangilash - YUMSHATILGAN"""
        if person_id not in self.hand_raise_history:
            self.hand_raise_history[person_id] = deque(maxlen=self.history_length)
        
        self.hand_raise_history[person_id].append(hand_raised)
        
        # Oxirgi N ta frame'dan kamida M tasida qo'l ko'tarilgan bo'lishi kerak
        recent_count = sum(self.hand_raise_history[person_id])
        is_stable = recent_count >= self.min_confidence
        
        # Debug
        if self.debug_mode:
            history_str = ''.join(['✓' if h else '✗' for h in self.hand_raise_history[person_id]])
            print(f"  Person {person_id}: {history_str} ({recent_count}/{self.history_length}) -> {'FAOL' if is_stable else 'oddiy'}")
        
        return is_stable
    
    def clean_old_persons(self, current_ids):
        """Eski shaxslarni tarixdan o'chirish"""
        to_remove = [pid for pid in self.hand_raise_history if pid not in current_ids]
        for pid in to_remove:
            del self.hand_raise_history[pid]


class SeatMonitor:
    """O'rindiqlarni kuzatish tizimi"""
    
    def __init__(self, config_file='seats_config.json'):
        self.seats = []
        self.config_file = config_file
        self.load_seats()
    
    def load_seats(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.seats = data.get('seats', [])
                print(f"✓ {len(self.seats)} ta o'rindiq yuklandi")
            except Exception as e:
                print(f"✗ Konfiguratsiya yuklanmadi: {e}")
                self.seats = []
    
    def save_seats(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'seats': self.seats}, f, ensure_ascii=False, indent=2)
            print(f"✓ {len(self.seats)} ta o'rindiq saqlandi")
        except Exception as e:
            print(f"✗ Saqlashda xato: {e}")
    
    def add_seat(self, name, points):
        seat = {'name': name, 'points': points, 'occupied': False, 'person_id': None}
        self.seats.append(seat)
        self.save_seats()
    
    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def check_occupancy(self, detections):
        for seat in self.seats:
            seat['occupied'] = False
            seat['person_id'] = None
        
        for det_id, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            bottom_center = [(x1+x2)//2, y2]
            
            for seat in self.seats:
                if self.point_in_polygon(bottom_center, seat['points']):
                    seat['occupied'] = True
                    seat['person_id'] = det_id
                    break
    
    def draw_seats(self, frame):
        for seat in self.seats:
            points = np.array(seat['points'], np.int32)
            color = (0, 255, 0) if seat['occupied'] else (0, 0, 255)
            status = "BAND" if seat['occupied'] else "BO'SH"
            
            cv2.polylines(frame, [points], True, color, 3)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            cx = int(np.mean([p[0] for p in seat['points']]))
            cy = int(np.mean([p[1] for p in seat['points']]))
            label = f"{seat['name']}: {status}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
            cv2.rectangle(frame, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), (0,0,0), -1)
            cv2.putText(frame, label, (cx-tw//2, cy), font, 0.7, (255,255,255), 2)


def setup_seats_interactive(camera_index=0):
    """O'rindiqlarni belgilash"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           O'RINDIQLARNI BELGILASH REJIMI                   ║
    ╠════════════════════════════════════════════════════════════╣
    ║  QANDAY ISHLAYDI:                                          ║
    ║                                                            ║
    ║  1. Videoda o'rindiq burchaklarini sichqoncha bilan        ║
    ║     bosing (4 ta nuqta)                                    ║
    ║                                                            ║
    ║  2. 4-nuqtadan keyin o'rindiq nomini kiriting              ║
    ║     (masalan: "1-qator 1-o'rin")                           ║
    ║                                                            ║
    ║  TUGMALAR:                                                 ║
    ║  's' - saqlash (saqlanadi avtomatik)                       ║
    ║  'c' - oxirgi o'rindiqni o'chirish                         ║
    ║  'r' - BARCHASINI tozalash                                 ║
    ║  'q' - chiqish                                             ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    seat_monitor = SeatMonitor()
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"✗ Kamera {camera_index} ochilmadi!")
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
    
    cv2.namedWindow('O\'rindiqlarni Belgilash')
    cv2.setMouseCallback('O\'rindiqlarni Belgilash', mouse_callback)
    
    print("\n✓ O'rindiqlarni belgilashni boshlang...")
    print("Har bir o'rindiq uchun 4 ta burchakni belgilang\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        seat_monitor.draw_seats(frame)
        
        # Joriy belgilanayotgan nuqtalarni ko'rsatish
        for i, point in enumerate(temp_points):
            cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Chiziqlarni ko'rsatish
        if len(temp_points) > 1:
            for i in range(len(temp_points)-1):
                cv2.line(frame, tuple(temp_points[i]), tuple(temp_points[i+1]), 
                        (255, 0, 0), 2)
            if len(temp_points) == 4:
                cv2.line(frame, tuple(temp_points[3]), tuple(temp_points[0]), 
                        (255, 0, 0), 2)
        
        # Ma'lumot paneli
        info = f"O'rindiqlar: {len(seat_monitor.seats)} | Joriy nuqtalar: {len(temp_points)}/4"
        cv2.putText(frame, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Tugmalar haqida ma'lumot
        help_text = "'c'-oxirgisini o'chirish | 'r'-barchasini tozalash | 'q'-chiqish"
        cv2.putText(frame, help_text, (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('O\'rindiqlarni Belgilash', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if seat_monitor.seats:
                removed = seat_monitor.seats.pop()
                seat_monitor.save_seats()
                print(f"✓ '{removed['name']}' o'chirildi")
            else:
                print("⚠️ O'chiriladigan o'rindiq yo'q!")
        elif key == ord('r'):
            if seat_monitor.seats:
                confirm = input("\n⚠️ BARCHA o'rindiqlarni o'chirmoqchimisiz? (ha/yo'q): ").strip().lower()
                if confirm in ['ha', 'yes', 'y']:
                    seat_monitor.seats = []
                    seat_monitor.save_seats()
                    temp_points = []
                    print("✓ Barcha o'rindiqlar tozalandi!")
                else:
                    print("✗ Bekor qilindi")
            else:
                print("⚠️ O'rindiqlar allaqachon bo'sh!")
        elif key == ord('s'):
            print(f"\n✓ {len(seat_monitor.seats)} ta o'rindiq saqlandi!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return seat_monitor


def monitoring_with_hand_detection(camera_index=0, confidence_threshold=0.45):
    """QO'L KO'TARISH bilan monitoring - YANGI YECHIM"""
    
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║    MAKTAB XAVFSIZLIK TIZIMI - QO'L KO'TARISH ANIQLASH      ║
    ║    🙋 Faol o'quvchilar KO'K rangda belgilanadi             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    if not install_ultralytics():
        return
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ Ultralytics import qilinmadi.")
        return
    
    # O'rindiqlar
    seat_monitor = SeatMonitor()
    if not seat_monitor.seats:
        print("\n⚠️ O'rindiqlar belgilanmagan!")
        choice = input("O'rindiqlarni hozir belgilaysizmi? (y/n): ")
        if choice.lower() == 'y':
            seat_monitor = setup_seats_interactive(camera_index)
            if not seat_monitor or not seat_monitor.seats:
                print("✗ Dastur to'xtatildi.")
                return
        else:
            print("✗ O'rindiqlar kerak!")
            return
    
    # Modellarni yuklash
    print("\nModellar yuklanmoqda...")
    try:
        person_model = YOLO('yolov8n.pt')  # Odamlarni aniqlash
        pose_model = YOLO('yolov8n-pose.pt')  # Qo'l pozitsiyasini aniqlash
        print("✓ Person model: yolov8n.pt")
        print("✓ Pose model: yolov8n-pose.pt")
    except Exception as e:
        print(f"✗ Model yuklanmadi: {e}")
        return
    
    # Qo'l ko'tarish aniqlagichi
    hand_detector = HandRaiseDetector(pose_model)
    
    # Kamera
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"✗ Kamera {camera_index} ochilmadi!")
        return
    
    print("\n" + "="*60)
    print("✓ TIZIM ISHGA TUSHDI - QO'L KO'TARISH ANIQLASH FAOL")
    print("="*60)
    print("Tugmalar:")
    print("  'q' - To'xtatish")
    print("  's' - Screenshot")
    print("  '+/-' - Aniqlikni sozlash")
    print("  'd' - DEBUG rejimini yoqish/o'chirish")
    print("  'p' - Pose detection ko'rsatish")
    print("="*60 + "\n")
    
    prev_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    current_confidence = confidence_threshold
    frame_skip = 1  # HAR FRAMEDA pose detection (2 dan 1 ga o'zgartirildi)
    frame_count = 0
    last_pose_results = []
    show_pose_points = False  # Pose nuqtalarini ko'rsatish
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # 1. ODAMLARNI ANIQLASH (har doim)
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
                    'hand_raised': False
                })
            
            # 2. POSE DETECTION (har N-frameda)
            if frame_count % frame_skip == 0 or frame_count == 1:
                pose_results = pose_model(frame, conf=0.3, verbose=False, device='cpu')
                last_pose_results = pose_results
            
            # 3. QO'L KO'TARISHNI TEKSHIRISH
            current_person_ids = set()
            if last_pose_results and len(last_pose_results) > 0:
                if hand_detector.debug_mode:
                    print(f"\n=== Frame {frame_count} - Pose Detection ===")
                
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['box']
                    person_id = det['person_id']
                    current_person_ids.add(person_id)
                    
                    # Bu odam uchun pose topish
                    if last_pose_results[0].keypoints is not None:
                        try:
                            if i < len(last_pose_results[0].keypoints):
                                kp = last_pose_results[0].keypoints[i].data[0].cpu().numpy()
                                
                                # Qo'l ko'tarilganligini tekshirish
                                hand_raised_now, reason = hand_detector.is_hand_raised(kp, det['box'])
                                
                                if hand_detector.debug_mode:
                                    status_text = "✓ Ko'tarilgan" if hand_raised_now else "✗ Pastda"
                                    print(f"Odam #{i+1}: {reason} -> {status_text}")
                                
                                # Stabillashtirish - tarixga qo'shish
                                is_stable = hand_detector.update_person_status(person_id, hand_raised_now)
                                det['hand_raised'] = is_stable
                                
                                # Pose nuqtalarini saqlash (vizualizatsiya uchun)
                                if show_pose_points:
                                    det['keypoints'] = kp
                        except Exception as e:
                            if hand_detector.debug_mode:
                                print(f"Odam #{i+1}: Xato - {e}")
            
            # Eski shaxslarni tozalash
            hand_detector.clean_old_persons(current_person_ids)
            
            # 4. O'RINDIQLARNI TEKSHIRISH
            seat_monitor.check_occupancy(detections)
            seat_monitor.draw_seats(frame)
            
            # 5. ODAMLARNI CHIZISH
            hand_raised_count = 0
            for det in detections:
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                hand_raised = det['hand_raised']
                
                if hand_raised:
                    hand_raised_count += 1
                
                # Rangni aniqlash
                if hand_raised:
                    color = (255, 150, 0)  # KO'K - qo'l ko'tarilgan
                    label_prefix = "🙋 FAOL"
                    thickness = 4
                else:
                    # O'rindiqda bo'lsa yashil, yo'qsa sariq
                    in_seat = any(s['person_id'] == det['person_id'] for s in seat_monitor.seats)
                    color = (0, 255, 0) if in_seat else (0, 255, 255)
                    label_prefix = "O'quvchi"
                    thickness = 2
                
                # Odam atrofini chizish
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # POSE NUQTALARINI KO'RSATISH (agar yoqilgan bo'lsa)
                if show_pose_points and 'keypoints' in det:
                    kp = det['keypoints']
                    # Asosiy nuqtalar: 0-burun, 5,6-elkalar, 7,8-tirsaklar, 9,10-bilaklar
                    important_points = [0, 5, 6, 7, 8, 9, 10]
                    for idx in important_points:
                        if idx < len(kp) and kp[idx][2] > 0.2:  # Confidence > 0.2
                            x, y = int(kp[idx][0]), int(kp[idx][1])
                            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                            cv2.putText(frame, str(idx), (x+12, y), 
                                       font, 0.5, (255, 255, 255), 2)
                    
                    # Elkalarni chiziq bilan bog'lash
                    if kp[5][2] > 0.2 and kp[6][2] > 0.2:
                        cv2.line(frame, 
                                (int(kp[5][0]), int(kp[5][1])),
                                (int(kp[6][0]), int(kp[6][1])),
                                (0, 255, 255), 2)
                    
                    # Qo'llarni chizish
                    # Chap qo'l: 5->7->9
                    if kp[5][2] > 0.2 and kp[7][2] > 0.2:
                        cv2.line(frame,
                                (int(kp[5][0]), int(kp[5][1])),
                                (int(kp[7][0]), int(kp[7][1])),
                                (0, 255, 0), 2)
                    if kp[7][2] > 0.2 and kp[9][2] > 0.2:
                        cv2.line(frame,
                                (int(kp[7][0]), int(kp[7][1])),
                                (int(kp[9][0]), int(kp[9][1])),
                                (0, 255, 0), 2)
                    
                    # O'ng qo'l: 6->8->10
                    if kp[6][2] > 0.2 and kp[8][2] > 0.2:
                        cv2.line(frame,
                                (int(kp[6][0]), int(kp[6][1])),
                                (int(kp[8][0]), int(kp[8][1])),
                                (255, 0, 0), 2)
                    if kp[8][2] > 0.2 and kp[10][2] > 0.2:
                        cv2.line(frame,
                                (int(kp[8][0]), int(kp[8][1])),
                                (int(kp[10][0]), int(kp[10][1])),
                                (255, 0, 0), 2)
                
                # Pastki nuqta
                bottom_center = [(x1+x2)//2, y2]
                cv2.circle(frame, bottom_center, 10, color, -1)
                cv2.circle(frame, bottom_center, 12, (255, 255, 255), 2)
                
                # Label
                seat_name = ""
                for seat in seat_monitor.seats:
                    if seat['person_id'] == det['person_id']:
                        seat_name = f" - {seat['name']}"
                        break
                
                label = f"{label_prefix} #{det['person_id']+1}{seat_name} ({int(conf*100)}%)"
                
                # Label foni
                bg_color = (200, 100, 0) if hand_raised else (0, 150, 0)
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-35), (x1+tw+10, y1-5), bg_color, -1)
                cv2.putText(frame, label, (x1+5, y1-15), font, 0.6, (255,255,255), 2)
            
            # 6. STATISTIKA PANELI
            occupied = sum(1 for s in seat_monitor.seats if s['occupied'])
            empty = len(seat_monitor.seats) - occupied
            
            panel_h = 170
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, 'XAVFSIZLIK + FAOLLIK TIZIMI', (20, 35), font, 0.7, (255,255,255), 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 60), font, 0.6, (0,255,0), 2)
            cv2.putText(frame, f'Band: {occupied} | Bo\'sh: {empty}', (20, 85), font, 0.6, (255,255,0), 2)
            cv2.putText(frame, f'Jami: {len(detections)} kishi', (20, 110), font, 0.6, (255,100,100), 2)
            cv2.putText(frame, f'🙋 Faol: {hand_raised_count} ta', (20, 135), font, 0.6, (255,150,0), 2)
            cv2.putText(frame, f'Aniqlik: {int(current_confidence*100)}%', (20, 160), font, 0.6, (0,255,255), 2)
            
            cv2.imshow('Maktab Tizimi - Qo\'l Ko\'tarish', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'screenshot_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot: {filename}")
            elif key in [ord('+'), ord('=')]:
                current_confidence = min(0.90, current_confidence + 0.05)
                print(f"⬆️ Aniqlik: {int(current_confidence*100)}%")
            elif key in [ord('-'), ord('_')]:
                current_confidence = max(0.20, current_confidence - 0.05)
                print(f"⬇️ Aniqlik: {int(current_confidence*100)}%")
    
    except KeyboardInterrupt:
        print("\n✓ Dastur to'xtatildi.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_index = None
    
    while True:
        print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║    MAKTAB XAVFSIZLIK TIZIMI - QO'L KO'TARISH VERSIYA       ║
    ║    Platform: {PLATFORM:<46} ║
    ╠════════════════════════════════════════════════════════════╣
    ║  [1] O'rindiqlarni sozlash                                 ║
    ║  [2] Monitoring + Qo'l Ko'tarish Aniqlash                  ║
    ║  [3] Kamerani qayta tanlash                                ║
    ║  [q] Chiqish                                               ║
    ╚════════════════════════════════════════════════════════════╝
    """)
        
        choice = input("\nTanlang (1-3 yoki q): ").strip().lower()
        
        if choice == 'q':
            print("✓ Dastur to'xtatildi. Xayr!")
            break
            
        elif choice == '1':
            if camera_index is None:
                camera_index = select_camera_source()
            setup_seats_interactive(camera_index=camera_index)
            print("\n✓ O'rindiqlar saqlandi! Endi [2] ni tanlang - Monitoring boshlash")
            input("\n[Enter] tugmasini bosing menuga qaytish uchun...")
            
        elif choice == '2':
            if camera_index is None:
                camera_index = select_camera_source()
            monitoring_with_hand_detection(camera_index=camera_index, confidence_threshold=0.45)
            print("\n✓ Monitoring to'xtatildi.")
            input("\n[Enter] tugmasini bosing menuga qaytish uchun...")
            
        elif choice == '3':
            camera_index = select_camera_source()
            print(f"✓ Kamera {camera_index} tanlandi")
            input("\n[Enter] tugmasini bosing menuga qaytish uchun...")
            
        else:
            print("✗ Noto'g'ri tanlov! 1, 2, 3 yoki q kiriting.")
            time.sleep(1)