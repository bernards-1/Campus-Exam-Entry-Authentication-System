import cvzone
from ultralytics import YOLO
import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
import datetime
import serial
import threading
import queue
import time
import streamlit as st
import random

if "stop" not in st.session_state:
    st.session_state.stop = False

def stop_app():
    st.session_state.stop = True
    st.write("[INFO] Closing resources...")
    try:
        if ser and ser.is_open:
            ser.close()
            st.write("[INFO] The serial port is closed")
    except Exception as e:
        st.write(f"[WARN] Serial port closing error: {e}")

    try:
        if conn:
            conn.close()
            st.write("[INFO] The database is closed")
    except Exception as e:
        st.write(f"[WARN] Database shutdown error: {e}")

    st.stop()  

# ===== Serial ports and queues =====
ser = serial.Serial('COM5', 115200, timeout=1)
uid_queue = queue.Queue()

# ===== database =====
conn = sqlite3.connect("students.db", check_same_thread=False)
cursor = conn.cursor()
# ==================== UI ====================
st.set_page_config(layout="wide")


img_path = None
student_id = "Unknown"
student_name = "Unknown"

# First create two columns on the left and right
col1, col2 = st.columns(2)

col1.header("📷 Camera Feed")
col2.header("ℹ Student Info")

# ====================  CSS ====================
st.markdown(
    """
<style>
    /* Simple rectangular photo style */
    .student-photo img {
        border: 2px solid #ccc;      /* Simple gray border */
        border-radius: 8px;          /* Slight rounded corners */
        width: 150px;
        height: 150px;
        object-fit: cover;
    }

    .student-info {
        font-size: 16px;
        margin-top: 10px;
        font-family: Arial, sans-serif;
    }

    .student-info-text {
        color: #333;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


def assign_seat(course_code, semester):
    cursor.execute("""
        SELECT seat_range
        FROM exams
        WHERE course_code = ? AND semester = ?
    """, (course_code, semester))
    exam_row = cursor.fetchone()

    if not exam_row:
        return None  

    
    seat_start, seat_end = map(int, exam_row[0].split("-"))
    all_seats = set(range(seat_start, seat_end + 1))
    print(f"📘 All seats: {all_seats}")

    
    cursor.execute("""
        SELECT seat_number
        FROM student_exam_attendance
        WHERE course_code = ? AND semester = ? AND exam_time = ?
          AND seat_number IS NOT NULL
    """, (course_code, semester, exam_time))
    used_seats = {row[0] for row in cursor.fetchall()}
    print(f"🪑 Used seats: {used_seats}")

    
    available_seats = list(all_seats - used_seats)
    print(f"✅ Available seats: {available_seats}")
    
    if not available_seats:
        return None 

    return random.choice(available_seats)

result_queue = queue.Queue()
frame_queue = queue.Queue()
seat_result_queue = queue.Queue()

# ===== take attendance =====
def mark_attendance(student_id, student_name,course_code,semester):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        SELECT exam_time, in_time, out_time, status 
        FROM student_exam_attendance
        WHERE student_id = ?
    """, (student_id,))
    record = cursor.fetchone()



    if record is None:
        print(f"⚠️ Not found {student_name} exam")
        return

    exam_time_str, in_time, out_time, status = record
    exam_time = datetime.datetime.strptime(exam_time_str, "%H:%M") if len(exam_time_str) <= 5 \
                else datetime.datetime.strptime(exam_time_str, "%Y-%m-%d %H:%M:%S")

    start = exam_time - datetime.timedelta(minutes=30)
    end = exam_time + datetime.timedelta(minutes=30)

    if in_time is None:
        if start <= now <= end:
            seat=assign_seat(course_code, semester)
            print(f"seat: {seat} ")

            cursor.execute("""
                UPDATE student_exam_attendance
                SET in_time = ?, status = 1,seat_number=?
                WHERE student_id = ? AND in_time IS NULL
            """, (now_str,seat, student_id))
            info="Check-In Success"
            seat_result_queue.put((student_id,course_code))
            print(f"✅ {student_name} Check-in Success {now_str}")
        elif now < start:
            print(f"⏳ {student_name} Too early.Only exam the first 30 minutes Sign Check-in  (Exam {exam_time_str})")
        else:
            print(f"❌ {student_name} Missed the check-in time (Exam {exam_time_str})")

    elif out_time is None:
        in_time_dt = datetime.datetime.strptime(in_time, "%Y-%m-%d %H:%M:%S")
        if (now - in_time_dt).total_seconds() < 300:
            print(f"ℹ️ {student_name} Already Check In at: ({in_time})")
        else:
            if now >= end:
                cursor.execute("""
                    UPDATE student_exam_attendance
                    SET out_time = ?,status=2
                    WHERE student_id = ? AND out_time IS NULL
                """, (now_str, student_id))
                seat_result_queue.put((student_id,course_code))
                print(f"✅ {student_name} Check out Success {now_str}")
            else:
                print(f"⏳ {student_name} It not signout time yet (Exam {exam_time_str})")
    else:
        print(f"ℹ️ {student_name} Completed Sign In & Sign Out")

    conn.commit()


def serial_thread():
    while True:
        if ser.in_waiting > 0:
            uid = ser.readline().decode('utf-8', errors='ignore').strip()
            if uid:
                print(f"Card UID: '{uid}'")
                uid_queue.put(uid)
        time.sleep(0.1)

if "student_id" not in st.session_state:
    st.session_state["student_id"] = "---"
if "student_name" not in st.session_state:
    st.session_state["student_name"] = "---"
if "img_path" not in st.session_state:
    st.session_state["img_path"] = "---"




# ===== cam =====
def camera_thread():
    facemodel = YOLO('yolov8n-face.pt')
    model_name = "ArcFace"
    threshold = 2.5

    cap = cv2.VideoCapture(1)
    current_student_id = None
    current_student_name = "Unknown"
    reference_embeddings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (700, 500))

        # check UID
        while not uid_queue.empty():
            uid = uid_queue.get()
            print(f"Card UID received: {uid}")
            #cursor.execute("SELECT student_id, name FROM student_cards WHERE UPPER(card_uid)=?", (uid.upper(),))
            cursor.execute("""
                SELECT sc.student_id, sc.name, si.image_path
                FROM student_cards sc
                JOIN student_images si ON sc.student_id = si.student_id
                WHERE UPPER(sc.card_uid)=?
            """, (uid.upper(),))
            row = cursor.fetchone()
            if row:
                current_student_id = row[0]
                current_student_name = row[1]
                current_img_path = row[2]

                cursor.execute("""
                    SELECT course_code,semester, exam_time
                    FROM student_exam_attendance
                    WHERE student_id = ?
                """, (current_student_id,))
                exam_row = cursor.fetchone()

                if exam_row:
                    current_course_code = exam_row[0]
                    current_semester = exam_row[1]
                    current_exam_time = exam_row[2]
                    print(f"📘 Current course code: {current_course_code}")
                    cursor.execute("SELECT course_name FROM courses WHERE course_code = ?", (current_course_code,))
                    course_row = cursor.fetchone()
                    if course_row:
                        current_course_name = course_row[0]
                        print(f"📘 Current course: {current_course_code} - {current_course_name}")
                    else:
                        print(f"⚠️ No course found for code {current_course_code}")
                   

                else:
                    current_course_code = None
                    current_semester = None
                    current_exam_time = None
                    current_course_name = None

                result_queue.put((current_student_id, current_student_name,current_img_path,
                                  current_course_code,current_semester,current_exam_time,current_course_name))

                # read embedding
                cursor.execute("""
                    SELECT e.embedding
                    FROM student_embeddings e
                    WHERE e.student_id = ?
                """, (current_student_id,))
                rows = cursor.fetchall()
                reference_embeddings = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
                print(f"找到 {len(reference_embeddings)} 个参考 embeddings")
            else:
                current_student_id = None
                current_student_name = "Unknown"
                reference_embeddings = []
                current_img_path = None
                print("❌ Unknown card")
                result_queue.put((None, "Unknown",None,None,None,None,None))

        # YOLO 
        face_result = facemodel.predict(frame, conf=0.40, verbose=False)
        for info in face_result:
            for box in info.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                face_crop = frame[y1:y2, x1:x2]

                try:
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

                    if reference_embeddings:
                        emb_obj = DeepFace.represent(face_resized, model_name=model_name, detector_backend="skip")
                        face_emb = np.array(emb_obj[0]["embedding"], dtype=np.float32)
                        distances = [np.linalg.norm(face_emb - ref_emb) for ref_emb in reference_embeddings]
                        min_dist = min(distances)
                        verified = min_dist < threshold
                        #text = f"{current_student_name} ({min_dist:.2f})" if verified else f"Unknown ({min_dist:.2f})"
                        text = f"{current_student_name} " if verified else f"Unknown"

                        if verified and current_student_id:
                            mark_attendance(current_student_id, current_student_name,current_course_code,current_semester)
                    else:
                        verified = False
                        text = "Please Scan Student Card"
                except Exception as e:
                    verified = False
                    text = f"Error: {str(e)[:20]}"
                    print(f"Error: {e}")

                color = (0, 255, 0) if verified else (0, 0, 255)
                cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3, colorR=color)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        time.sleep(0.03)

    cap.release()



t1 = threading.Thread(target=serial_thread, daemon=True)
t2 = threading.Thread(target=camera_thread, daemon=True)
t1.start()
t2.start()


# ==================== Streamlit UI ====================
st.set_page_config(page_title="Smart Attendance", layout="wide")
col1, col2 = st.columns(2)
camera_placeholder = col1.empty()
info_placeholder = col2.empty()


current_student = {
    "sid": None,
    "name": None,
    "img_path": None,
    "course_code": None,
    "semester": None,
    "exam_time": None,
    "course_name": None,
    "seat_number": None,
    "in_time": None,
    "out_time": None,
    "status": 0
}

while True:
    
    if not frame_queue.empty():
        frame = frame_queue.get()
        camera_placeholder.image(frame, channels="BGR", use_container_width=True)

    # if result_queue has data will upadate current_student
    if not result_queue.empty():
        try:
            sid, name, img_path, course_code, semester, exam_time, course_name = result_queue.get()
            current_student.update({
                "sid": sid,
                "name": name,
                "img_path": img_path,
                "course_code": course_code,
                "semester": semester,
                "exam_time": exam_time,
                "course_name": course_name
            })
            if sid and course_code:
                cursor.execute("""
                    SELECT seat_number, in_time, out_time, status
                    FROM student_exam_attendance
                    WHERE student_id = ? AND course_code = ?
                """, (sid, course_code))
                row = cursor.fetchone()
                if row:
                    seat_number, in_time, out_time, status = row
                    # current_student
                    current_student.update({
                        "seat_number": seat_number,
                        "in_time": in_time,
                        "out_time": out_time,
                        "status": status
                    })
        except ValueError:
            pass  #  current_student

    # if seat_result_queue has data will chect seat/in/out/status
    if not seat_result_queue.empty():
        sid, course_code = seat_result_queue.get()
        cursor.execute("""
            SELECT seat_number, in_time, out_time, status
            FROM student_exam_attendance
            WHERE student_id = ? AND course_code = ?
        """, (sid, course_code))
        row = cursor.fetchone()
        if row:
            seat_number, in_time, out_time, status = row
            current_student.update({
                "seat_number": seat_number,
                "in_time": in_time,
                "out_time": out_time,
                "status": status
            })

    # show 
    course_display = f"{current_student['course_code']} {current_student['course_name']}" if current_student["course_code"] and current_student["course_name"] else ""

    checkin_info = "Waiting"
    info = ""
    if current_student["status"] == 1:
        checkin_info = "Checked In"
        info = "Check In Success"

    elif current_student["status"] == 2:
        checkin_info = "Checked Out"
        info = "Check Out Success"


    with info_placeholder.container():
        st.markdown('<div class="student-photo">', unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1, 3, 3])
        with col_center:
            if current_student["img_path"]:
                st.image(current_student["img_path"], width=150)
            else:
                st.info("Please Scan Student Card")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<h3>{current_student["name"] or " "}  <span style="color:lightgreen;">{info}</span></h3>', unsafe_allow_html=True)

        if current_student['seat_number']:
            st.markdown(f'<h3>Your Seat is <span style="color:red;">{current_student["seat_number"]}</span></h3>', unsafe_allow_html=True)






        st.markdown(f"""
        <div class="student-card" style="border:1px solid #ccc; padding:10px; border-radius:10px;">
            <b>Student ID:</b> {current_student['sid'] or "N/A"}<br>
            <b>Name:</b> {current_student['name'] or "Unknown"}<br>
            <b>Semester:</b> {current_student['semester'] or " "}<br>
            <b>Course Code:</b>{course_display or " "}<br>
            <b>Exam Time:</b>{current_student['exam_time'] or " "}<br>
            <b>Seat Number:</b> {current_student['seat_number'] or "N/A"}<br>
            <b>In Time:</b> {current_student['in_time'] or " "}<br>
            <b>Out Time:</b> {current_student['out_time'] or " "}<br>
            <b>Status:</b> {checkin_info}<br>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(0.01)  # 
