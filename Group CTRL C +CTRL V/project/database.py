import sqlite3

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
    course_code TEXT NOT NULL,
    semester TEXT NOT NULL,
    exam_time DATETIME,
    seat_number int,
    status INTEGER DEFAULT 0,
    in_time DATETIME,
    out_time DATETIME,
    FOREIGN KEY (course_code) REFERENCES courses(course_code)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS exams (
    exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL,
    semester TEXT NOT NULL,           
    exam_time DATETIME NOT NULL,
    seat_range TEXT NOT NULL,         
    FOREIGN KEY (course_code) REFERENCES courses(course_code)
);
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS student_cards (
    student_id TEXT PRIMARY KEY,
    name TEXT,
    card_uid TEXT UNIQUE 
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS courses (
    course_code TEXT PRIMARY KEY,  
    course_name TEXT NOT NULL
);
""")

###### Student Card ########
cursor.execute("""
INSERT INTO student_cards (student_id, name, card_uid)
VALUES (?, ?, ?)
""", ("S00001", "Alice", "UID000001"))



#### Course ####
cursor.execute("""
INSERT INTO courses (course_code, course_name) 
VALUES (?, ?)
""", ("CS101", "Introduction to Programming"))
conn.commit()

########### Exam info ##########
cursor.execute("""
INSERT INTO exams (course_code, semester, exam_time, seat_range)
VALUES (?, ?, ?, ?)
""", ("CS101", "202501", "2025-09-15 09:00:00", "1-10"))
########### Student exam ##########
cursor.execute("""
INSERT INTO student_exam_attendance (student_id, name, course_code, semester, exam_time) 
VALUES (?, ?, ?, ?, ?)
""", ("S00001", "Alice", "CS101", "202501", "2025-09-15 09:00:00"))
conn.commit()

