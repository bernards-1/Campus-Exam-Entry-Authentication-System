import sqlite3
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import cv2

# 连接数据库
conn = sqlite3.connect("students.db")
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    course TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS student_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(student_id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS student_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(student_id)
)
''')


cursor.execute("""
CREATE TABLE IF NOT EXISTS student_exam_attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    name TEXT NOT NULL,
    exam_time DATETIME,
    status INTEGER DEFAULT 0,
    in_time DATETIME,
    out_time DATETIME
    
)
""")


# create student

# cursor.execute("""
# INSERT INTO students (student_id, name, course) 
# VALUES (?, ?, ?)
# """, ("S00001", "Alice", "RSD"))

# YOLO 人脸检测
yolo_model = YOLO("yolov8n-face.pt")

photos = [f"./personPhoto/photo_{i:02d}.jpg" for i in range(1, 110)]

studentid="S00001"

for p in photos:
    

    try:
        # 读图
        img = cv2.imread(p)
        if img is None:
            print(f"无法读取 {p}")
            continue
            
        results = yolo_model(img, verbose=False)

        # 取第一个人脸框
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            face_crop = img[y1:y2, x1:x2]

            h0, w0 = face_crop.shape[:2]
            max_side = max(h0, w0)

            padded = cv2.copyMakeBorder(
                face_crop,
                (max_side - h0) // 2,
                (max_side - h0 + 1) // 2,
                (max_side - w0) // 2,
                (max_side - w0 + 1) // 2,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            face_resized = cv2.resize(padded, (112, 112))
            
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            #
            embedding = DeepFace.represent(
                face_resized, model_name="ArcFace", detector_backend="skip"
            )[0]["embedding"]

            # save into database
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute("INSERT INTO student_embeddings (student_id, embedding) VALUES (?, ?)",
                           (studentid, embedding_blob))
            print(f"✅ saved {p}  embedding")
        else:
            print(f"⚠️ {p} cant detect face")

    except Exception as e:
        print(f"⚠️ {p} generate embedding fail: {e}")

conn.commit()
conn.close()
print("all embedding generate success")