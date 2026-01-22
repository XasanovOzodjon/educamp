"""
O'RINDIQ MONITORING MODULI
Fayl: seat_monitoring_module.py
Versiya: 3.0
"""

import cv2
import numpy as np
import json
import os


class SeatMonitor:
    """O'rindiqlarni kuzatish tizimi"""
    
    def __init__(self, config_file='seats_config.json'):
        self.seats = []
        self.config_file = config_file
        self.load_seats()
    
    def load_seats(self):
        """O'rindiqlar konfiguratsiyasini yuklash"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.seats = data.get('seats', [])
                print(f"✓ {len(self.seats)} ta o'rindiq yuklandi")
            except Exception as e:
                print(f"✗ Konfiguratsiya yuklanmadi: {e}")
                self.seats = []
        else:
            print("ℹ O'rindiqlar konfiguratsiyasi topilmadi")
    
    def save_seats(self):
        """O'rindiqlar konfiguratsiyasini saqlash"""
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
        print(f"✓ O'rindiq '{name}' qo'shildi")
    
    def remove_seat(self, index):
        """O'rindiqni o'chirish"""
        if 0 <= index < len(self.seats):
            removed = self.seats.pop(index)
            self.save_seats()
            print(f"✓ O'rindiq '{removed['name']}' o'chirildi")
            return True
        return False
    
    def clear_all_seats(self):
        """Barcha o'rindiqlarni tozalash"""
        self.seats = []
        self.save_seats()
        print("✓ Barcha o'rindiqlar tozalandi")
    
    def point_in_polygon(self, point, polygon):
        """Nuqta polygon ichida ekanligini tekshirish"""
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
        """O'rindiq bandligini tekshirish - YAXSHILANGAN"""
        # Avval barcha o'rindiqlarni bo'sh qilib olish
        for seat in self.seats:
            seat['occupied'] = False
            seat['person_id'] = None
            seat['overlap_score'] = 0
        
        # Har bir aniqlangan odamni tekshirish
        for det_id, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            
            # Odamning pastki markazi (oyoq joyi) - ENG MUHIM
            bottom_center = [(x1+x2)//2, y2]
            
            # Odamning tanasining markaziy qismi
            body_center = [(x1+x2)//2, (y1+y2*3)//4]
            
            best_seat = None
            best_score = 0
            
            # Qaysi o'rindiqda ekanligini aniqlash
            for seat in self.seats:
                score = 0
                
                # Pastki markaz o'rindiqda ekanligini tekshirish (eng muhim)
                if self.point_in_polygon(bottom_center, seat['points']):
                    score += 100
                
                # Tana markazi ham o'rindiqda bo'lsa qo'shimcha ball
                if self.point_in_polygon(body_center, seat['points']):
                    score += 50
                
                # Agar bu o'rindiqda eng yuqori ball bo'lsa
                if score > best_score:
                    best_score = score
                    best_seat = seat
            
            # Eng mos o'rindiqni belgilash
            if best_seat and best_score >= 100:
                # Agar o'rindiq allaqachon band bo'lsa, yuqori scoreni saqlaymiz
                if not best_seat['occupied'] or best_score > best_seat['overlap_score']:
                    best_seat['occupied'] = True
                    best_seat['person_id'] = det_id
                    best_seat['overlap_score'] = best_score
    
    def draw_seats(self, frame):
        """O'rindiqlarni rasmga chizish"""
        for seat in self.seats:
            points = np.array(seat['points'], np.int32)
            
            # Rang: band - yashil, bo'sh - qizil
            color = (0, 255, 0) if seat['occupied'] else (0, 0, 255)
            status = "BAND" if seat['occupied'] else "BO'SH"
            
            # Polygon chizish
            cv2.polylines(frame, [points], True, color, 3)
            
            # Shaffof to'ldirish
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # O'rindiq nomi va holati
            cx = int(np.mean([p[0] for p in seat['points']]))
            cy = int(np.mean([p[1] for p in seat['points']]))
            label = f"{seat['name']}: {status}"
            
            # Label chizish
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
            cv2.rectangle(frame, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), (0,0,0), -1)
            cv2.putText(frame, label, (cx-tw//2, cy), font, 0.7, (255,255,255), 2)
    
    def get_statistics(self):
        """O'rindiq statistikasini olish"""
        stats = {
            'total_seats': len(self.seats),
            'occupied_seats': sum(1 for s in self.seats if s['occupied']),
            'empty_seats': sum(1 for s in self.seats if not s['occupied']),
            'occupancy_rate': 0.0
        }
        
        if stats['total_seats'] > 0:
            stats['occupancy_rate'] = (stats['occupied_seats'] / stats['total_seats']) * 100
        
        return stats
    
    def get_seat_by_person(self, person_id):
        """Shaxs qaysi o'rindiqda ekanligini topish"""
        for seat in self.seats:
            if seat.get('person_id') == person_id and seat.get('occupied'):
                return seat['name']
        return None
    
    def export_to_json(self, filename='seats_export.json'):
        """O'rindiqlarni JSON faylga eksport qilish"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'seats': self.seats,
                    'statistics': self.get_statistics()
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ O'rindiqlar '{filename}' ga eksport qilindi")
            return True
        except Exception as e:
            print(f"✗ Eksport xato: {e}")
            return False
    
    def import_from_json(self, filename):
        """O'rindiqlarni JSON fayldan import qilish"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.seats = data.get('seats', [])
            self.save_seats()
            print(f"✓ {len(self.seats)} ta o'rindiq '{filename}' dan import qilindi")
            return True
        except Exception as e:
            print(f"✗ Import xato: {e}")
            return False


# Test funksiyasi
if __name__ == "__main__":
    print("Seat Monitoring Module - Test Mode")
    monitor = SeatMonitor(config_file='test_seats.json')
    
    # Test o'rindiq qo'shish
    test_seat_1 = [[100, 100], [200, 100], [200, 200], [100, 200]]
    test_seat_2 = [[300, 100], [400, 100], [400, 200], [300, 200]]
    
    monitor.add_seat("O'rindiq 1", test_seat_1)
    monitor.add_seat("O'rindiq 2", test_seat_2)
    
    # Test detections
    test_detections = [
        {'box': [80, 80, 150, 220], 'person_id': 0},   # O'rindiq 1 da
        {'box': [320, 90, 380, 210], 'person_id': 1},  # O'rindiq 2 da
    ]
    
    # Bandlikni tekshirish
    monitor.check_occupancy(test_detections)
    
    # Statistika
    stats = monitor.get_statistics()
    print(f"\nStatistika:")
    print(f"  Jami o'rindiqlar: {stats['total_seats']}")
    print(f"  Band: {stats['occupied_seats']}")
    print(f"  Bo'sh: {stats['empty_seats']}")
    print(f"  Bandlik: {stats['occupancy_rate']:.1f}%")
    
    # Shaxsning o'rindiqini topish
    seat_name = monitor.get_seat_by_person(person_id=0)
    print(f"\nPerson 0 o'rindigi: {seat_name}")
    
    # Eksport
    monitor.export_to_json('test_export.json')
    
    # Tozalash
    if os.path.exists('test_seats.json'):
        os.remove('test_seats.json')
    if os.path.exists('test_export.json'):
        os.remove('test_export.json')
    
    print("\n✓ Test tugadi")
