# 🎓 MAKTAB MONITORING TIZIMI - MODULYAR VERSIYA 3.0

## 📦 MODULYAR TUZILMA

Bu versiyada kod **7 ta alohida modul**ga bo'lingan - har bir modul o'z vazifasini bajaradi va mustaqil test qilinadi.

## 📁 FAYLLAR STRUKTURASI

```
maktab-monitoring/
│
├── 📄 main.py                           # Asosiy dastur
├── 📄 face_recognition_module.py        # Yuz tanish (InsightFace)
├── 📄 sleep_detection_module.py         # Uyqu aniqlash
├── 📄 phone_detection_module.py         # Telefon aniqlash
├── 📄 hand_raise_detection_module.py    # Qo'l ko'tarish
├── 📄 seat_monitoring_module.py         # O'rindiq monitoring
├── 📄 camera_utils.py                   # Kamera utilities
│
├── 📄 requirements.txt                  # Python kutubxonalar
├── 📄 README.md                         # Bu fayl
│
├── 📄 students_faces_insightface.json   # Yuz ma'lumotlari (avtomatik)
└── 📄 seats_config.json                 # O'rindiqlar (avtomatik)
```

---

## 🎯 HAR BIR MODULNING VAZIFASI

### 1️⃣ `face_recognition_module.py`
**Vazifa:** Yuzlarni tanish (InsightFace SOTA)

**Asosiy funksiyalar:**
- `InsightFaceRecognitionSystem` - Yuz tanish tizimi
- `register_face()` - Yangi yuz ro'yxatdan o'tkazish
- `recognize_face()` - Yuzni tanish
- `load_database()` - Ma'lumotlar bazasini yuklash
- `save_database()` - Ma'lumotlar bazasini saqlash

**Test:**
```bash
python face_recognition_module.py
```

---

### 2️⃣ `sleep_detection_module.py`
**Vazifa:** Uyqu holatini aniqlash

**Asosiy funksiyalar:**
- `SleepDetector` - Uyqu aniqlagich
- `detect_sleep()` - Uyquni aniqlash (10+ parametr)
- `calculate_head_tilt()` - Bosh egilish burchagi
- `update_sleep_status()` - Holatni yangilash
- `get_statistics()` - Statistika

**Test:**
```bash
python sleep_detection_module.py
```

---

### 3️⃣ `phone_detection_module.py`
**Vazifa:** Telefon ishlatishni aniqlash

**Asosiy funksiyalar:**
- `PhoneDetector` - Telefon aniqlagich
- `detect_phone_usage()` - Telefon ishlatishni aniqlash
- `update_phone_status()` - Holatni yangilash
- `adjust_sensitivity()` - Sezgirlikni sozlash
- `get_statistics()` - Statistika

**Test:**
```bash
python phone_detection_module.py
```

---

### 4️⃣ `hand_raise_detection_module.py`
**Vazifa:** Qo'l ko'tarishni aniqlash

**Asosiy funksiyalar:**
- `HandRaiseDetector` - Qo'l ko'tarish aniqlagichi
- `is_hand_raised()` - Qo'l ko'tarilganligini tekshirish
- `update_person_status()` - Shaxs holatini yangilash
- `clean_old_persons()` - Eski ma'lumotlarni tozalash
- `adjust_sensitivity()` - Sezgirlikni sozlash

**Test:**
```bash
python hand_raise_detection_module.py
```

---

### 5️⃣ `seat_monitoring_module.py`
**Vazifa:** O'rindiqlarni kuzatish

**Asosiy funksiyalar:**
- `SeatMonitor` - O'rindiq monitoring tizimi
- `add_seat()` - Yangi o'rindiq qo'shish
- `check_occupancy()` - Bandlikni tekshirish
- `draw_seats()` - O'rindiqlarni chizish
- `get_statistics()` - Statistika
- `export_to_json()` / `import_from_json()` - Eksport/Import

**Test:**
```bash
python seat_monitoring_module.py
```

---

### 6️⃣ `camera_utils.py`
**Vazifa:** Kamera va umumiy utilities

**Asosiy funksiyalar:**
- `open_camera_smart()` - Kamerani aqlli ochish
- `list_cameras()` - Mavjud kameralarni sanash
- `select_camera_source()` - Kamera tanlash
- `configure_camera()` - Kamera sozlamalari
- `install_required_packages()` - Kutubxonalarni o'rnatish

**Test:**
```bash
python camera_utils.py
```

---

### 7️⃣ `main.py`
**Vazifa:** Asosiy dastur - barcha modullarni birlashtiradi

**Asosiy funksiyalar:**
- `setup_seats_interactive()` - O'rindiqlarni sozlash
- `register_students_faces()` - Yuzlarni ro'yxatdan o'tkazish
- `full_monitoring_system()` - To'liq monitoring
- `main()` - Asosiy menyu

**Ishga tushirish:**
```bash
python main.py
```

---

## 🚀 TEZKOR BOSHLASH

### 1. O'rnatish

```bash
# Barcha kutubxonalarni o'rnatish
pip install -r requirements.txt

# Yoki qo'lda
pip install insightface onnxruntime ultralytics opencv-python numpy
```

### 2. Ishga tushirish

```bash
python main.py
```

### 3. Menyu

```
[1] O'rindiqlarni sozlash
[2] O'quvchilar yuzlarini ro'yxatdan o'tkazish
[3] TO'LIQ MONITORING TIZIMINI BOSHLASH
[4] Kamerani qayta tanlash
[5] Modullar haqida ma'lumot
[q] Chiqish
```

---

## 🎨 MODULYAR AFZALLIKLAR

### ✅ Oson Saqlanish
- Har bir modul alohida
- Bir modulni o'zgartirish boshqalariga ta'sir qilmaydi
- Git bilan ishlash qulay

### ✅ Test Qilish
- Har bir modulni alohida test qilish mumkin
- Xatolarni tez topish
- Debug oson

### ✅ Qayta Ishlatish
- Modullarni boshqa loyihalarda ishlatish
- Kod takrorlanmasligi
- Professional tuzilma

### ✅ Team Development
- Har bir dasturchi alohida modul ustida ishlashi mumkin
- Merge conflicts kamayadi
- Parallel development

### ✅ Kengaytirish
- Yangi modullar qo'shish oson
- Mavjud modullarni yangilash
- Versiyalarni boshqarish

---

## 📊 MODULLAR O'RTASIDAGI BOG'LANISH

```
main.py
    │
    ├─→ camera_utils.py
    │       └─→ Kamera ochish va sozlash
    │
    ├─→ face_recognition_module.py
    │       └─→ Yuz tanish (InsightFace)
    │
    ├─→ sleep_detection_module.py
    │       └─→ Uyqu aniqlash
    │
    ├─→ phone_detection_module.py
    │       └─→ Telefon aniqlash
    │
    ├─→ hand_raise_detection_module.py
    │       └─→ Qo'l ko'tarish aniqlash
    │
    └─→ seat_monitoring_module.py
            └─→ O'rindiq monitoring
```

---

## 🔧 HAR BIR MODULNI ALOHIDA ISHLATISH

### Misol 1: Faqat yuz tanish

```python
from face_recognition_module import InsightFaceRecognitionSystem

# Tizimni ishga tushirish
face_system = InsightFaceRecognitionSystem()

# Yuzni ro'yxatdan o'tkazish
face_system.register_face(frame, bbox, "Ali Valiyev")

# Yuzni tanish
name, confidence = face_system.recognize_face(frame, bbox)
print(f"Ism: {name}, Aniqlik: {confidence}%")
```

### Misol 2: Faqat uyqu aniqlash

```python
from sleep_detection_module import SleepDetector

# Detektorni yaratish
detector = SleepDetector()

# Uyquni aniqlash
is_sleeping, reason = detector.detect_sleep(keypoints)

# Holatni yangilash
stable = detector.update_sleep_status(person_id=1, is_sleeping=is_sleeping)
```

### Misol 3: O'rindiq monitoring

```python
from seat_monitoring_module import SeatMonitor

# Monitoring tizimini yaratish
monitor = SeatMonitor()

# O'rindiq qo'shish
monitor.add_seat("O'rindiq 1", [[100,100], [200,100], [200,200], [100,200]])

# Bandlikni tekshirish
monitor.check_occupancy(detections)

# Statistika
stats = monitor.get_statistics()
print(f"Band: {stats['occupied_seats']}/{stats['total_seats']}")
```

---

## 💡 YANGI MODUL QO'SHISH

Yangi modul qo'shish uchun:

1. **Yangi fayl yarating:**
```python
# new_module.py
class NewDetector:
    def __init__(self):
        pass
    
    def detect(self, data):
        # Sizning logikangiz
        pass
```

2. **main.py ga import qiling:**
```python
from new_module import NewDetector
```

3. **Ishlatish:**
```python
detector = NewDetector()
result = detector.detect(data)
```

---

## 📋 REQUIREMENTS

```txt
insightface>=0.7.3
onnxruntime>=1.16.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## 🐛 MUAMMOLARNI HAL QILISH

### Modul import qilinmayapti
```bash
# Pythonpath ni sozlang
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Yoki Windows'da
set PYTHONPATH=%PYTHONPATH%;%CD%
```

### Kutubxona topilmayapti
```bash
# Virtual environment yarating
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Kutubxonalarni o'rnating
pip install -r requirements.txt
```

---

## 📞 YORDAM

- 📧 Email: support@example.com
- 💬 Telegram: @support_bot
- 🌐 GitHub: [repository-link]

---

## 📄 LITSENZIYA

MIT License - erkin ishlatish va o'zgartirish

---

## 🙏 MINNATDORCHILIK

- InsightFace - SOTA yuz tanish
- Ultralytics - YOLO modellari
- OpenCV - Computer vision
- Python community

---

**Versiya:** 3.0 Modulyar  
**Sana:** 2025  
**Holat:** Production Ready ✅

🎓 **Maktabingiz xavfsiz bo'lsin!** 🎓
