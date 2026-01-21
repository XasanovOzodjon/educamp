import cv2
import numpy as np
import time
import os
import glob
import json

# OpenCV xato xabarlarini yashirish
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

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
    ║                                                            ║
    ║  [1] Oddiy kamera (Laptop yoki USB kamera)                 ║
    ║  [2] OBS Virtual Camera (OBS orqali)                       ║
    ║  [3] Mavjud kameralarni ko'rish                            ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        choice = input("Tanlang (1-3): ").strip()
        
        if choice == '1':
            print("\n✓ Oddiy kamera tanlandi (0-index)")
            return 0
        
        elif choice == '2':
            print("\n✓ OBS Virtual Camera tanlandi")
            print("   OBS'da Settings → Video → Virtual Camera ishga tushganiga ishonch hosil qiling!")
            
            # OBS odatda 1 yoki 2 indexda bo'ladi
            available, names = list_cameras()
            if len(available) > 1:
                camera_name = names.get(available[1], "Noma'lum")
                print(f"\n   OBS uchun tavsiya: Kamera [{available[1]}] - {camera_name}")
                use_suggested = input(f"   Kamera {available[1]} ishlatilsinmi? (y/n): ").strip().lower()
                if use_suggested == 'y':
                    return available[1]
            
            # Qo'lda tanlash
            try:
                camera_index = int(input("   Kamera indexini kiriting (odatda 1 yoki 2): "))
                return camera_index
            except:
                print("✗ Noto'g'ri format!")
                continue
        
        elif choice == '3':
            available, names = list_cameras()
            if available:
                try:
                    camera_index = int(input("\nQaysi kamerani ishlatmoqchisiz? Index kiriting: "))
                    if camera_index in available:
                        print(f"✓ Kamera {camera_index} tanlandi: {names.get(camera_index, 'Kamera')}")
                        return camera_index
                    else:
                        print("✗ Noto'g'ri index!")
                except:
                    print("✗ Noto'g'ri format!")
            continue
        
        else:
            print("✗ Noto'g'ri tanlov! 1, 2 yoki 3 kiriting.")


def list_cameras():
    """Mavjud kameralarni ko'rsatish"""
    import contextlib
    
    print("\n" + "="*60)
    print("MAVJUD KAMERALAR:")
    print("="*60)
    
    available_cameras = []
    camera_names = {}
    
    # Xato xabarlarini vaqtincha o'chirish
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                        # Kamera nomini taxmin qilish
                        if i == 0:
                            camera_names[i] = "Asosiy kamera (Laptop/USB)"
                        elif i == 1:
                            camera_names[i] = "OBS Virtual Camera (ehtimol)"
                        else:
                            camera_names[i] = f"Kamera {i}"
                        print(f"  [{i}] {camera_names[i]}")
                    cap.release()
    
    if not available_cameras:
        print("  Hech qanday kamera topilmadi!")
    
    print("="*60 + "\n")
    return available_cameras, camera_names


class SeatMonitor:
    """O'rindiqlarni kuzatish tizimi"""
    
    def __init__(self, config_file='seats_config.json'):
        self.seats = []
        self.config_file = config_file
        self.setup_mode = False
        self.temp_points = []
        self.current_seat_name = ""
        self.load_seats()
    
    def load_seats(self):
        """Saqlangan o'rindiqlarni yuklash"""
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
        """O'rindiqlarni saqlash"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'seats': self.seats}, f, ensure_ascii=False, indent=2)
            print(f"✓ {len(self.seats)} ta o'rindiq saqlandi")
        except Exception as e:
            print(f"✗ Saqlashda xato: {e}")
    
    def add_seat(self, name, points):
        """Yangi o'rindiq qo'shish"""
        seat = {
            'name': name,
            'points': points,
            'occupied': False,
            'person_id': None
        }
        self.seats.append(seat)
        self.save_seats()
    
    def point_in_polygon(self, point, polygon):
        """Nuqta polygon ichidami tekshirish"""
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
        """Har bir o'rindiqni tekshirish - DEBUG bilan"""
        # Avval hamma o'rindiqni bo'sh deb belgilaymiz
        for seat in self.seats:
            seat['occupied'] = False
            seat['person_id'] = None
        
        # Har bir aniqlangan odamni tekshirish
        for det_id, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            
            # MUHIM: Odamning pastki markazini tekshirish (oyoqlari bo'lgan joy)
            bottom_center = [(x1+x2)//2, y2]
            
            # Qo'shimcha nuqtalar - kengroq tekshirish
            bottom_left = [x1 + (x2-x1)//4, y2]
            bottom_right = [x2 - (x2-x1)//4, y2]
            mid_bottom = [(x1+x2)//2, y2 - (y2-y1)//4]
            
            # DEBUG: Tekshirilayotgan nuqtalarni saqlash
            det['check_points'] = [bottom_center, bottom_left, bottom_right, mid_bottom]
            
            # Qaysi o'rindiqda ekanligini topish
            found_seat = False
            for seat in self.seats:
                if (self.point_in_polygon(bottom_center, seat['points']) or 
                    self.point_in_polygon(bottom_left, seat['points']) or
                    self.point_in_polygon(bottom_right, seat['points']) or
                    self.point_in_polygon(mid_bottom, seat['points'])):
                    seat['occupied'] = True
                    seat['person_id'] = det_id
                    found_seat = True
                    print(f"✓ Odam #{det_id+1} -> {seat['name']}")
                    break
            
            if not found_seat:
                print(f"⚠️ Odam #{det_id+1} hech qaysi o'rindiqda emas! Nuqta: {bottom_center}")
    
    def draw_seats(self, frame, debug=True):
        """O'rindiqlarni rasmda ko'rsatish"""
        for i, seat in enumerate(self.seats):
            points = np.array(seat['points'], np.int32)
            
            # Rang tanlash
            if seat['occupied']:
                color = (0, 255, 0)  # Yashil - band
                status = "BAND"
            else:
                color = (0, 0, 255)  # Qizil - bo'sh
                status = "BO'SH"
            
            # Polygon chizish - QALINROQ
            cv2.polylines(frame, [points], True, color, 3)
            
            # Shaffof to'ldirish
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # DEBUG: O'rindiq burchaklarini ko'rsatish
            if debug:
                for idx, point in enumerate(seat['points']):
                    cv2.circle(frame, tuple(point), 6, (255, 255, 0), -1)
                    cv2.circle(frame, tuple(point), 8, (255, 255, 255), 1)
                    cv2.putText(frame, str(idx+1), (point[0]+10, point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # O'rindiq nomi va holati
            cx = int(np.mean([p[0] for p in seat['points']]))
            cy = int(np.mean([p[1] for p in seat['points']]))
            
            label = f"{seat['name']}: {status}"
            
            # Matn fonini chizish
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(label, font, 0.7, 2)
            cv2.rectangle(frame, (cx - text_width//2 - 5, cy - text_height - 5),
                         (cx + text_width//2 + 5, cy + 5), (0, 0, 0), -1)
            
            # Matn
            cv2.putText(frame, label, (cx - text_width//2, cy), 
                       font, 0.7, (255, 255, 255), 2)


def setup_seats_interactive(camera_index=0):
    """Interaktiv o'rindiqlarni sozlash"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           O'RINDIQLARNI BELGILASH REJIMI                   ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  QANDAY ISHLAYDI:                                          ║
    ║                                                            ║
    ║  1. Videoda o'rindiq burchaklarini sichqoncha bilan        ║
    ║     bosing (4 ta nuqta)                                    ║
    ║                                                            ║
    ║  2. 4-nuqtadan keyin o'rindiq nomini kiriting              ║
    ║     (masalan: "1-qator 1-o'rin")                           ║
    ║                                                            ║
    ║  3. Keyingi o'rindiqni belgilang                           ║
    ║                                                            ║
    ║  4. 's' tugmasini bosing - saqlash                         ║
    ║     'c' tugmasini bosing - oxirgi o'rindiqni o'chirish     ║
    ║     'r' tugmasini bosing - barchasini tozalash             ║
    ║     'q' tugmasini bosing - chiqish                         ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    seat_monitor = SeatMonitor()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # MUHIM: Monitoring bilan bir xil o'lcham
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"✗ Kamera {camera_index} ochilmadi!")
        return None
    
    temp_points = []
    current_name = ""
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal temp_points, current_name
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(temp_points) < 4:
                temp_points.append([x, y])
                print(f"Nuqta {len(temp_points)}: ({x}, {y})")
                
                if len(temp_points) == 4:
                    print("\n4 ta nuqta belgilandi!")
                    current_name = input("O'rindiq nomini kiriting: ")
                    seat_monitor.add_seat(current_name, temp_points)
                    print(f"✓ '{current_name}' qo'shildi!")
                    temp_points = []
    
    cv2.namedWindow('O\'rindiqlarni Belgilash')
    cv2.setMouseCallback('O\'rindiqlarni Belgilash', mouse_callback)
    
    print("\n✓ O'rindiqlarni belgilashni boshlang...")
    print("Har bir o'rindiq uchun 4 ta burchakni belgilang\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Saqlangan o'rindiqlarni ko'rsatish
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
        
        cv2.imshow('O\'rindiqlarni Belgilash', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if seat_monitor.seats:
                removed = seat_monitor.seats.pop()
                seat_monitor.save_seats()
                print(f"✓ '{removed['name']}' o'chirildi")
        elif key == ord('r'):
            seat_monitor.seats = []
            seat_monitor.save_seats()
            temp_points = []
            print("✓ Barcha o'rindiqlar tozalandi")
        elif key == ord('s'):
            print(f"\n✓ {len(seat_monitor.seats)} ta o'rindiq saqlandi!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return seat_monitor


def monitoring_with_seats(camera_index=0, models_folder='models', confidence_threshold=0.45):
    """O'rindiqlarni kuzatish bilan monitoring"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         MAKTAB VIDEO XAVFSIZLIK TIZIMI                     ║
    ║         O'rindiq Kuzatuvi + AI Detection                   ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Ultralytics o'rnatish
    if not install_ultralytics():
        return
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ Ultralytics import qilinmadi.")
        return
    
    # O'rindiqlar kuzatuvchisini yuklash
    seat_monitor = SeatMonitor()
    
    if not seat_monitor.seats:
        print("\n⚠️ O'rindiqlar belgilanmagan!")
        choice = input("O'rindiqlarni hozir belgilaysizmi? (y/n): ")
        if choice.lower() == 'y':
            seat_monitor = setup_seats_interactive(camera_index)
            if not seat_monitor or not seat_monitor.seats:
                print("✗ O'rindiqlar belgilanmadi!")
                return
        else:
            print("✗ O'rindiqlar kerak! Dastur to'xtatildi.")
            return
    
    # Modellarni yuklash
    print("\nModellar yuklanmoqda...")
    models = {}
    
    # Default model
    try:
        models['yolov8n'] = YOLO('yolov8n.pt')
        print("✓ yolov8n.pt yuklandi")
    except Exception as e:
        print(f"✗ Model yuklanmadi: {e}")
        return
    
    # Kamerani ochish
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"✗ Kamera {camera_index} ochilmadi!")
        return
    
    print("\n" + "="*60)
    print("✓ MAKTAB XAVFSIZLIK TIZIMI ISHGA TUSHDI!")
    print("="*60)
    print(f"O'rindiqlar: {len(seat_monitor.seats)} ta")
    print(f"Kamera: {camera_index}")
    print("\nTugmalar:")
    print("  'q' - To'xtatish")
    print("  's' - Screenshot olish")
    print("  'e' - O'rindiqlarni qayta sozlash")
    print("="*60 + "\n")
    
    prev_time = 0
    model = models['yolov8n']
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS hisoblash
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # YOLO detection
            results = model(frame, conf=confidence_threshold, classes=[0], 
                           verbose=False, device='cpu')
            boxes = results[0].boxes
            
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                center = [(x1+x2)//2, (y1+y2)//2]
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'center': center
                })
            
            # O'rindiqlarni tekshirish
            seat_monitor.check_occupancy(detections)
            
            # O'rindiqlarni chizish
            seat_monitor.draw_seats(frame)
            
            # Aniqlangan odamlarni chizish
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                
                # Pastki markazni hisoblash
                bottom_center = [(x1+x2)//2, y2]
                
                # Rangni aniqlash (o'rindiqda bo'lsa yashil, bo'lmasa sariq)
                in_seat = False
                seat_name = ""
                for seat in seat_monitor.seats:
                    if seat['person_id'] == i:
                        in_seat = True
                        seat_name = seat['name']
                        break
                
                color = (0, 255, 0) if in_seat else (0, 255, 255)
                
                # Odam atrofini chizish - QALIN
                thickness = 3 if in_seat else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # DEBUG: Barcha tekshirish nuqtalarini ko'rsatish
                if 'check_points' in det:
                    for idx, point in enumerate(det['check_points']):
                        point_color = (0, 255, 0) if in_seat else (0, 0, 255)
                        cv2.circle(frame, point, 5, point_color, -1)
                        cv2.circle(frame, point, 7, (255, 255, 255), 1)
                        # Nuqta raqami
                        cv2.putText(frame, str(idx+1), (point[0]+10, point[1]), 
                                   font, 0.4, (255, 255, 255), 1)
                
                # Asosiy pastki nuqta - KATTAROQ
                cv2.circle(frame, bottom_center, 10, color, -1)
                cv2.circle(frame, bottom_center, 12, (255, 255, 255), 2)
                
                # Label
                if in_seat:
                    label = f"#{i+1} - {seat_name} ✓"
                    bg_color = (0, 200, 0)
                else:
                    label = f"#{i+1} - JOYDA EMAS ✗"
                    bg_color = (0, 100, 200)
                
                # Label foni
                (text_width, text_height), _ = cv2.getTextSize(label, font, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-35), (x1 + text_width + 10, y1-5), bg_color, -1)
                cv2.putText(frame, label, (x1+5, y1-15), font, 0.6, (255, 255, 255), 2)
            
            # Statistika paneli
            occupied_seats = sum(1 for seat in seat_monitor.seats if seat['occupied'])
            empty_seats = len(seat_monitor.seats) - occupied_seats
            
            panel_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, 'XAVFSIZLIK TIZIMI', (20, 35), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 60), font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Band: {occupied_seats} | Bo\'sh: {empty_seats}', 
                       (20, 85), font, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f'Jami: {len(detections)} kishi', 
                       (20, 110), font, 0.6, (255, 100, 100), 2)
            
            cv2.imshow('Maktab Xavfsizlik Tizimi', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'screenshot_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saqlandi: {filename}")
            elif key == ord('e'):
                cap.release()
                cv2.destroyAllWindows()
                seat_monitor = setup_seats_interactive(camera_index)
                if seat_monitor and seat_monitor.seats:
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print("✓ O'rindiqlar yangilandi, monitoring davom etmoqda...")
                else:
                    print("✗ O'rindiqlar belgilanmadi!")
                    return
    
    except KeyboardInterrupt:
        print("\n✓ Dastur to'xtatildi.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║       MAKTAB VIDEO XAVFSIZLIK TIZIMI - MENYU               ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  [1] O'rindiqlarni sozlash/belgilash                       ║
    ║  [2] Monitoring boshlash (o'rindiqlar bilan)               ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    choice = input("\nTanlang (1-2): ").strip()
    
    if choice == '1':
        # Kamera tanlash
        camera_index = select_camera_source()
        setup_seats_interactive(camera_index=camera_index)
        
    elif choice == '2':
        # Kamera tanlash
        camera_index = select_camera_source()
        monitoring_with_seats(camera_index=camera_index, confidence_threshold=0.45)
        
    else:
        print("✗ Noto'g'ri tanlov!")