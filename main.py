import cv2
import numpy as np
import time
import os
import glob

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

def find_all_models(models_folder='models'):
    """
    models/ papkasidagi barcha .pt modellarni topish
    """
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"✓ '{models_folder}' papkasi yaratildi")
        return []
    
    # Barcha .pt fayllarni topish
    model_files = glob.glob(os.path.join(models_folder, '*.pt'))
    
    models_info = []
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        models_info.append({
            'name': model_name,
            'path': model_path,
            'size': os.path.getsize(model_path) / (1024 * 1024)  # MB
        })
    
    return models_info

def list_cameras():
    """Mavjud kameralarni ko'rsatish"""
    import sys
    import contextlib
    
    print("\n" + "="*60)
    print("MAVJUD KAMERALAR:")
    print("="*60)
    
    available_cameras = []
    
    # Xato xabarlarini vaqtincha o'chirish
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow backend
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"  [{i}] Kamera topildi")
                    cap.release()
    
    if not available_cameras:
        print("  Hech qanday kamera topilmadi!")
    
    print("="*60 + "\n")
    return available_cameras




def train_new_model(base_model='yolov8n.pt', dataset_path=None, epochs=50, model_name='custom'):
    """
    Yangi model train qilish
    
    Parameters:
    - base_model: Boshlang'ich model (yolov8n.pt, yolov8s.pt, ...)
    - dataset_path: Dataset yo'li (YOLO formatida)
    - epochs: Necha marta o'rganish
    - model_name: Model nomi
    """
    try:
        from ultralytics import YOLO
    except:
        print("✗ Ultralytics yuklanmadi")
        return None
    
    print("\n" + "="*60)
    print("YANGI MODEL TRAIN QILISH")
    print("="*60)
    
    if dataset_path is None:
        print("\n✗ Dataset yo'li ko'rsatilmagan!")
        print("\nDataset formati (YOLO):")
        print("  dataset/")
        print("    ├── images/")
        print("    │   ├── train/")
        print("    │   └── val/")
        print("    ├── labels/")
        print("    │   ├── train/")
        print("    │   └── val/")
        print("    └── data.yaml")
        return None
    
    print(f"\nBase model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Model nomi: {model_name}")
    
    try:
        # Modelni yuklash
        model = YOLO(base_model)
        
        # Train qilish
        print("\n🚀 Training boshlandi...")
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=model_name,
            patience=10,
            save=True,
            device='cpu'  # GPU bo'lsa 'cuda' yoki 0
        )
        
        # Modelni saqlash
        output_path = f'models/{model_name}.pt'
        os.makedirs('models', exist_ok=True)
        
        best_model_path = f'runs/detect/{model_name}/weights/best.pt'
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, output_path)
            print(f"\n✓ Model saqlandi: {output_path}")
            return output_path
        
    except Exception as e:
        print(f"\n✗ Training xatosi: {e}")
        return None


def auto_ensemble_detection(camera_index=None, models_folder='models', confidence_threshold=0.45, max_models=5):
    """
    AVTOMATIK ENSEMBLE - KUCHAYTIRILGAN VERSIYA
    Ko'proq model + Optimizatsiya + GPU support = Kuchliroq AI!
    
    Parameters:
    - camera_index: Kamera raqami
    - models_folder: Modellar saqlanadigan papka
    - confidence_threshold: Aniqlik chegarasi (0.45 - kam xato, faqat odamlar)
    - max_models: Maksimal ishlatish uchun modellar (5-7 optimal)
    """
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     AUTO-ENSEMBLE AI - KUCHAYTIRILGAN TIZIM ⚡️        ║
    ║     (5+ Model + GPU + HD = SUPER AI! 🚀)              ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Ultralytics o'rnatish
    if not install_ultralytics():
        return
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ Ultralytics import qilinmadi.")
        return
    
    # Barcha modellarni topish
    print(f"\n{'='*60}")
    print(f"'{models_folder}/' PAPKASIDAGI MODELLAR:")
    print(f"{'='*60}")
    
    custom_models = find_all_models(models_folder)
    
    if not custom_models:
        print(f"\n⚠️ '{models_folder}/' papkasida hech qanday model topilmadi!")
        print("\nStandard YOLOv8 modellaridan foydalanamiz...")
        
        # Default modellar (turli o'lchamlarda - kuchli ensemble uchun)
        default_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        print(f"\nDefault modellar yuklanmoqda: {', '.join(default_models[:max_models])}")
        
        models = {}
        for model_file in default_models[:max_models]:
            try:
                model_name = model_file.replace('.pt', '')
                models[model_name] = YOLO(model_file)
                print(f"✓ {model_file} yuklandi")
            except Exception as e:
                print(f"✗ {model_file} yuklanmadi: {e}")
    else:
        print(f"\n✓ {len(custom_models)} ta custom model topildi!")
        
        # HAR XIL O'LCHAMDAGI modellarni tanlash (kuchli ensemble uchun)
        # Kichik, o'rta va katta modellarni aralashtirish
        custom_models_sorted = sorted(custom_models, key=lambda x: x['size'])
        
        # Strategik tanlash: kichik, o'rta, katta modellar
        selected_models = []
        if len(custom_models_sorted) <= max_models:
            selected_models = custom_models_sorted
        else:
            # Kichikdan kattaga teng taqsimlash
            step = len(custom_models_sorted) / max_models
            for i in range(max_models):
                idx = int(i * step)
                selected_models.append(custom_models_sorted[idx])
        
        custom_models = selected_models
        print(f"✓ KUCHLI ENSEMBLE: {len(custom_models)} xil o'lchamdagi model:\n")
        
        for i, model_info in enumerate(custom_models, 1):
            print(f"  [{i}] {model_info['name']}")
            print(f"      O'lcham: {model_info['size']:.2f} MB")
            print()
        
        # Modellarni yuklash
        print("Modellar yuklanmoqda...")
        models = {}
        
        for model_info in custom_models:
            try:
                model_name = model_info['name'].replace('.pt', '')
                models[model_name] = YOLO(model_info['path'])
                print(f"✓ {model_info['name']} yuklandi")
            except Exception as e:
                print(f"✗ {model_info['name']} yuklanmadi: {e}")
    
    if not models:
        print("\n✗ Hech qanday model yuklanmadi!")
        return
    
    print(f"\n✓ Jami {len(models)} ta model tayyor!")
    print(f"✓ Ensemble aqli: {len(models)} x kuchaytirilgan!\n")
    
    # Kameralarni ko'rsatish
    available = list_cameras()
    
    if camera_index is None:
        if not available:
            print("✗ Kamera topilmadi!")
            return
        
        print("Qaysi kamerani ishlatmoqchisiz?")
        for cam in available:
            print(f"  {cam} - Kamera {cam}")
        
        try:

            camera_index = int(input("\nKamera raqamini kiriting (OBS uchun 1 yoki 2): "))
        except:
            camera_index = available[0]
    
    # GPU mavjudligini tekshirish
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_info = f" [{torch.cuda.get_device_name(0)}]" if device == 'cuda' else ""
    
    # Image size - FPS uchun optimizatsiya qilingan
    img_size = 480 if device == 'cuda' else 384
    
    print(f"\n⚡️ Device: {device.upper()}{gpu_info}")
    
    # Kamerani ochish (DirectShow backend - Windows uchun optimal)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # OPTIMAL O'LCHAM - FPS va sifat balansi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer kichraytirish
    
    if not cap.isOpened():
        print(f"\n✗ Kamera {camera_index} ochilmadi!")
        return
    
    print("\n" + "="*60)
    print("✓ KUCHAYTIRILGAN AUTO-ENSEMBLE AI! ⚡️🚀")
    print("="*60)
    print(f"Modellar soni: {len(models)} ta {'⭐️' * min(len(models), 5)}")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"Image Size: {img_size}px (aniqlik: HIGH)")
    print(f"Confidence: {int(confidence_threshold * 100)}%")
    print(f"Device: {device.upper()}{gpu_info}")
    print(f"Rejim: NAVBAT ENSEMBLE (har model navbatda)")
    print("\nTugmalar:")
    print("  'q' - To'xtatish")
    print("  '+/-' - Aniqlik sozlash")
    print("  'i' - Info panel (on/off)")
    print("  'r' - Reload modellar")
    print("  'f' - Frame skip (NORMAL/FAST/ULTRA)")
    print("="*60 + "\n")
    
    prev_time = 0
    show_info = True
    frame_count = 0
    
    # TEZLASHTIRISH: Frame skipping (1 = har frame)
    skip_frames = 1  # Har frameni tekshirish (yuqori sifat)
    frame_skip_counter = 0
    
    # Modellar navbati (4 ta yengil model - tezlik uchun)
    model_keys = list(models.keys())[:4]  # Faqat 4 ta eng yengil
    current_model_index = 0
    
    # Har bir modeldan oxirgi natijalarni saqlash
    model_results = {key: [] for key in model_keys}
    
    # Har bir detection uchun timestamp
    detection_timestamps = {key: 0 for key in model_keys}
    detection_timeout = 0.2  # 0.2 sekund - agar detection yangilanmasa, o'chirish
    
    # Oxirgi natija
    last_merged = []
    
    # Ranglar
    np.random.seed(42)
    colors = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_skip_counter += 1
            height, width = frame.shape[:2]
            
            # FPS hisoblash
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # FRAME SKIPPING - har N-framedan 1 tasini tekshirish
            if frame_skip_counter >= skip_frames:
                frame_skip_counter = 0
                
                # FAQAT bitta modelni ishlatish (navbat bilan)
                current_model_key = model_keys[current_model_index]
                current_model = models[current_model_key]
                
                # YUQORI SIFAT - faqat ishonchli detectionlar
                results = current_model(frame, conf=confidence_threshold, classes=[0], 
                                       verbose=False, imgsz=img_size, device=device,
                                       iou=0.45, agnostic_nms=True, max_det=50)
                boxes = results[0].boxes
                
                # Shu modelning natijalarini yangilash
                model_results[current_model_key] = []
                detection_timestamps[current_model_key] = curr_time  # Timestamp yangilash
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    model_results[current_model_key].append({
                        'box': [x1, y1, x2, y2],
                        'confidence': conf,
                        'model': current_model_key,
                        'center': [(x1+x2)//2, (y1+y2)//2],
                        'timestamp': curr_time
                    })
                
                # Keyingi modelga o'tish
                current_model_index = (current_model_index + 1) % len(model_keys)
                
                # ESKI NATIJALARNI TOZALASH - faqat yangi detectionlarni qoldirish
                all_detections = []
                for model_key in model_keys:
                    # Agar model oxirgi 0.5 sekundda yangilangan bo'lsa
                    if curr_time - detection_timestamps.get(model_key, 0) < detection_timeout:
                        all_detections.extend(model_results[model_key])
                    else:
                        # Eski natijalarni tozalash
                        model_results[model_key] = []
                
                # Birlashtirilgan natija
                if all_detections:
                    last_merged = merge_detections(all_detections)
            
            people_count = len(last_merged)
            
            # Aniqlangan odamlarni chizish
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, det in enumerate(last_merged):
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                votes = det.get('votes', 1)
                
                # Rang
                if i not in colors:
                    colors[i] = (
                        np.random.randint(100, 255),
                        np.random.randint(100, 255),
                        np.random.randint(100, 255)
                    )
                color = colors[i]
                
                # To'rtburchak - qalinlik ovoz soniga bog'liq
                thickness = min(2 + votes, 6)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label
                if votes > 1:
                    label = f"#{i+1} {int(conf * 100)}% [{votes} models]"
                else:
                    label = f"#{i+1} {int(conf * 100)}%"
                
                # Label fon
                label_size, _ = cv2.getTextSize(label, font, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label matn
                cv2.putText(frame, label, (x1 + 5, y1 - 6), font, 0.5, (255, 255, 255), 2)
                
                # Markazga nuqta - o'lcham ovoz soniga bog'liq
                point_size = 3 + votes
                cv2.circle(frame, det['center'], point_size, color, -1)
            
            # Info panel (sodda versiya - tezroq)
            if show_info:
                panel_width = 420
                panel_height = 180
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                y_pos = 35
                cv2.putText(frame, 'AUTO-ENSEMBLE', 
                           (20, y_pos), font, 0.7, (255, 255, 255), 2)
                y_pos += 32
                
                # FPS
                fps_color = (0, 255, 0) if fps > 15 else (0, 255, 255) if fps > 10 else (255, 100, 100)
                cv2.putText(frame, f'FPS: {int(fps)}', 
                           (20, y_pos), font, 0.9, fps_color, 2)
                y_pos += 32
                
                # Odamlar
                cv2.putText(frame, f'Odamlar: {people_count}', 
                           (20, y_pos), font, 0.8, (0, 255, 0), 2)
                y_pos += 28
                
                # Modellar
                stars = '⭐️' * min(len(models), 5)
                cv2.putText(frame, f'{stars} {len(models)} models',

                           (20, y_pos), font, 0.6, (255, 215, 0), 2)
                y_pos += 25
                
                # Device
                device_text = f'GPU: {device.upper()}' if device == 'cuda' else 'CPU Mode'
                device_color = (0, 255, 0) if device == 'cuda' else (100, 150, 255)
                cv2.putText(frame, device_text, 
                           (20, y_pos), font, 0.5, device_color, 1)
                y_pos += 20
                
                # Skip mode
                skip_mode = "NORMAL" if skip_frames == 1 else "FAST" if skip_frames == 2 else "ULTRA"
                cv2.putText(frame, f'Mode: {skip_mode} | ImgSz: {img_size}', 
                           (20, y_pos), font, 0.5, (100, 200, 255), 1)
            else:
                cv2.putText(frame, f'FPS: {int(fps)} | Odamlar: {people_count} | Models: {len(models)}', 
                           (10, 30), font, 0.6, (0, 255, 0), 2)
            
            # Oyna
            window_title = f'SUPER AI ⚡️ - {len(models)} Models | {device.upper()} | HD'
            cv2.imshow(window_title, frame)
            
            # Tugmalar
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Dastur to'xtatildi.")
                break
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.02)
                print(f"Aniqlik oshirildi: {int(confidence_threshold * 100)}%")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.20, confidence_threshold - 0.02)
                print(f"Aniqlik kamaytirildi: {int(confidence_threshold * 100)}%")
            elif key == ord('i'):
                show_info = not show_info
            elif key == ord('r'):
                print("\n♻️ Modellar qayta yuklanmoqda...")
                cap.release()
                cv2.destroyAllWindows()
                auto_ensemble_detection(camera_index, models_folder, confidence_threshold, max_models)
                return
            elif key == ord('f'):
                # Frame skip rejimini o'zgartirish
                if skip_frames == 1:
                    skip_frames = 2
                    print("Rejim: FAST (har 2-frame) - 2x tezroq")
                elif skip_frames == 2:
                    skip_frames = 3
                    print("Rejim: ULTRA (har 3-frame) - 3x tezroq")
                else:
                    skip_frames = 1
                    print("Rejim: NORMAL (har frame) - yuqori sifat")
    
    except KeyboardInterrupt:
        print("\n✓ Dastur to'xtatildi.")
    except Exception as e:
        print(f"\n✗ XATOLIK: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Resurslar bo'shatildi.")


def merge_detections(detections, iou_threshold=0.50):
    """Detections ni birlashtirish - aniqroq
    
    IoU threshold 0.50 - faqat juda yaqin detectionlarni birlashtirish,
    har bir odam alohida = aniqroq hisob
    """
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    merged = []
    used = [False] * len(detections)
    
    for i, det1 in enumerate(detections):
        if used[i]:
            continue
        
        group = [det1]
        used[i] = True
        
        for j, det2 in enumerate(detections[i+1:], i+1):
            if used[j]:
                continue
            
            iou = calculate_iou(det1['box'], det2['box'])
            
            if iou > iou_threshold:
                group.append(det2)
                used[j] = True
        
        if group:
            avg_box = average_boxes([d['box'] for d in group])
            avg_conf = sum([d['confidence'] for d in group]) / len(group)
            
            merged.append({
                'box': avg_box,
                'confidence': avg_conf,
                'votes': len(group),
                'center': [(avg_box[0]+avg_box[2])//2, (avg_box[1]+avg_box[3])//2]
            })
    
    return merged

def calculate_iou(box1, box2):
    """IoU hisoblash"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def average_boxes(boxes):
    """O'rtacha box"""
    x1 = int(sum([b[0] for b in boxes]) / len(boxes))
    y1 = int(sum([b[1] for b in boxes]) / len(boxes))
    x2 = int(sum([b[2] for b in boxes]) / len(boxes))
    y2 = int(sum([b[3] for b in boxes]) / len(boxes))
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         AUTO-ENSEMBLE AI - O'ZINI O'RGANGAN TIZIM          ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  QANDAY ISHLAYDI:                                          ║
    ║                                                            ║
    ║  1. 'models/' papkasiga .pt modellarni qo'shing            ║
    ║  2. Dastur avtomatik barcha modellarni topadi              ║
    ║  3. Har bir model navbat bilan ishlaydi                    ║
    ║  4. Natijalar birlashtiriladi                              ║
    ║  5. Ko'proq model = aqlliroq AI! ⭐️⭐️⭐️⭐️⭐️                 ║
    ║                                                            ║
    ║  O'Z MODELINGIZNI QANDAY YARATISH:                         ║
    ║                                                            ║
    ║  1. Dataset tayyorlang (YOLO formatida):                   ║
    ║     • Rasmlar va labellar                                  ║
    ║     • train/ va val/ bo'limlari                            ║
    ║                                                            ║
    ║  2. Model train qiling:                                    ║
    ║     from main import train_new_model                       ║
    ║     train_new_model(                                       ║
    ║         base_model='yolov8n.pt',                           ║
    ║         dataset_path='data.yaml',                          ║
    ║         epochs=50,                                         ║
    ║         model_name='my_custom_model'                       ║
    ║     )                                                      ║
    ║                                                            ║
    ║  3. Yangi model avtomatik 'models/' ga saqlanadi           ║
    ║                                                            ║
    ║  4. Dastur qayta ishga tushganda yangi modelni            ║
    ║     avtomatik topadi va ishlatadi!                         ║
    ║                                                            ║
    ║  MASLAHAT:                                                 ║
    ║  • Har xil datasetlarda train qiling                       ║
    ║  • Har xil base modellardan foydalaning                    ║
    ║  • 3-5 ta model = optimal                                  ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Auto-ensemble tizimini ishga tushirish
    # max_models=5 - 5 ta har xil o'lchamdagi model (kuchli ensemble uchun)
    # confidence=0.45 - faqat ishonchli detectionlar (kam xato)
    auto_ensemble_detection(models_folder='models', confidence_threshold=0.45, max_models=5)
    
    #salom