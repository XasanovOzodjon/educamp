"""
KAMERA VA UMUMIY UTILITY MODULI
Fayl: camera_utils.py
Versiya: 3.0
"""

import cv2
import os
import platform
import contextlib


# Platform aniqlash
PLATFORM = platform.system()


def get_camera_backend():
    """Platform asosida eng mos kamera backend'ini qaytarish"""
    if PLATFORM == 'Windows':
        return cv2.CAP_DSHOW
    elif PLATFORM == 'Darwin':  # macOS
        return cv2.CAP_AVFOUNDATION
    else:  # Linux
        return cv2.CAP_V4L2


def open_camera_smart(camera_index, backend=None):
    """
    Kamerani aqlli ochish - agar berilgan kamera ishlamasa, boshqalarini qidiradi
    
    Args:
        camera_index: Ochish uchun kamera indexi
        backend: Kamera backend (None bo'lsa avtomatik aniqlanadi)
    
    Returns:
        VideoCapture obyekti yoki None
    """
    if backend is None:
        backend = get_camera_backend()
    
    cap = cv2.VideoCapture(camera_index, backend)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"✓ Kamera {camera_index} muvaffaqiyatli ochildi!")
            return cap
        cap.release()
    
    print(f"⚠ Kamera {camera_index} ishlamadi. Boshqa kameralarni qidiryapman...")
    
    # Boshqa kameralarni qidirish
    for idx in [0, 1, 2]:
        if idx == camera_index:
            continue
        
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✓ Kamera {idx} topildi va ishlatilmoqda!")
                return cap
            cap.release()
    
    print("✗ Hech qanday ishlaydigan kamera topilmadi!")
    return None


def list_cameras(max_cameras=10):
    """
    Mavjud kameralarni sanab chiqish
    
    Args:
        max_cameras: Tekshirish uchun maksimal kamera soni
    
    Returns:
        tuple: (mavjud_indexlar_listi, ism_dict)
    """
    print("\n" + "="*60)
    print(f"MAVJUD KAMERALAR ({PLATFORM}):")
    print("="*60)
    
    available = []
    names = {}
    backend = get_camera_backend()
    
    # Xato chiqishlarini yashirish
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            for i in range(max_cameras):
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


def select_camera_source():
    """Foydalanuvchidan kamera manbani tanlash"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              KAMERA MANBANI TANLANG                        ║
    ╠════════════════════════════════════════════════════════════╣
    ║  [1] Oddiy kamera (Laptop yoki USB kamera)                 ║
    ║  [2] OBS Virtual Camera yoki boshqa kamera                 ║
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
            except ValueError:
                print("✗ Noto'g'ri format! Raqam kiriting.")
        
        elif choice == '3':
            available, names = list_cameras()
            if available:
                try:
                    idx = int(input("\nQaysi kamerani ishlatmoqchisiz? Index: "))
                    if idx in available:
                        return idx
                    else:
                        print("✗ Bu index mavjud emas!")
                except ValueError:
                    print("✗ Noto'g'ri format!")
            else:
                print("✗ Kameralar topilmadi!")
        
        else:
            print("✗ Noto'g'ri tanlov! 1, 2 yoki 3 ni tanlang.")


def configure_camera(cap, width=1280, height=720, fps=30):
    """
    Kamera sozlamalarini o'rnatish
    
    Args:
        cap: VideoCapture obyekti
        width: Kenglik (pixel)
        height: Balandlik (pixel)
        fps: Frame per second
    
    Returns:
        bool: Muvaffaqiyat holati
    """
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Haqiqiy qiymatlarni tekshirish
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Kamera sozlamalari:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")
        
        return True
    except Exception as e:
        print(f"⚠ Kamera sozlashda xato: {e}")
        return False


def get_camera_info(cap):
    """
    Kamera ma'lumotlarini olish
    
    Args:
        cap: VideoCapture obyekti
    
    Returns:
        dict: Kamera ma'lumotlari
    """
    if not cap or not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'backend': cap.getBackendName()
    }
    
    return info


def install_required_packages():
    """Zarur kutubxonalarni o'rnatish"""
    required_packages = {
        'ultralytics': 'YOLO modellari uchun',
        'insightface': 'Yuz tanish uchun (SOTA model)',
        'onnxruntime': 'InsightFace backend uchun'
    }
    
    all_installed = True
    
    print("\n--- ASOSIY KUTUBXONALAR ---")
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - o'rnatilgan ({description})")
        except ImportError:
            print(f"\n⚠ {package} o'rnatilmoqda... ({description})")
            import subprocess
            try:
                subprocess.check_call(['pip', 'install', package])
                print(f"✓ {package} muvaffaqiyatli o'rnatildi!")
            except:
                print(f"✗ {package} o'rnatilmadi. Qo'lda o'rnating: pip install {package}")
                all_installed = False
    
    if not all_installed:
        print("\n⚠ Ba'zi kutubxonalar o'rnatilmadi, lekin davom etamiz...")
    else:
        print("\n✅ Barcha kutubxonalar tayyor!")
    
    return all_installed


def suppress_opencv_warnings():
    """OpenCV xato xabarlarini yashirish"""
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


def get_platform_info():
    """Platform ma'lumotlarini olish"""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }


# Test funksiyasi
if __name__ == "__main__":
    print("Camera Utils Module - Test Mode")
    
    # OpenCV ogohlantirishlarini yashirish
    suppress_opencv_warnings()
    
    # Platform ma'lumotlari
    platform_info = get_platform_info()
    print(f"\n📌 Platform: {platform_info['system']}")
    print(f"📌 Machine: {platform_info['machine']}")
    
    # Backend aniqlash
    backend = get_camera_backend()
    print(f"\n🎥 Kamera backend: {backend}")
    
    # Kameralarni sanash
    available, names = list_cameras()
    print(f"\n📹 {len(available)} ta kamera topildi")
    
    # Birinchi kamerani ochish
    if available:
        print(f"\n🔄 Kamera {available[0]} ni ochish...")
        cap = open_camera_smart(available[0])
        
        if cap:
            # Kamera ma'lumotlari
            info = get_camera_info(cap)
            print(f"\n✓ Kamera ma'lumotlari:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # Sozlash
            configure_camera(cap, width=640, height=480)
            
            # Test frame
            ret, frame = cap.read()
            if ret:
                print(f"\n✓ Frame olinadi: {frame.shape}")
            
            cap.release()
            print("\n✓ Kamera yopildi")
    else:
        print("\n✗ Kameralar topilmadi")
    
    print("\n✓ Test tugadi")
