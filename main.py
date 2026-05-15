import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# CRICKET BOWLING KINEMATICS ANALYZER
# =========================================================
# Features:
# - Pose Estimation using MediaPipe
# - Elbow / Knee / Shoulder Angle Tracking
# - Angular Velocity Analysis
# - Release Frame Detection
# - Bowling Phase Segmentation
# - Biomechanics Dashboard
# - Kinematic Insights
#
# Recommended:
# - Side-angle bowling video
# - Full body visible
# - Portrait videos work well
# =========================================================


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# ANGLE FUNCTION
# =========================================================

def calculate_angle(a, b, c):
    """
    Calculates angle between 3 points.
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)


# =========================================================
# LANDMARK EXTRACTION
# =========================================================

def get_landmarks(results, frame):

    if not results.pose_landmarks:
        return None

    h, w, _ = frame.shape
    lm = results.pose_landmarks.landmark

    def pt(id):
        return [lm[id].x * w, lm[id].y * h]

    return {

        "left_shoulder": pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
        "left_elbow": pt(mp_pose.PoseLandmark.LEFT_ELBOW),
        "left_wrist": pt(mp_pose.PoseLandmark.LEFT_WRIST),

        "left_hip": pt(mp_pose.PoseLandmark.LEFT_HIP),
        "left_knee": pt(mp_pose.PoseLandmark.LEFT_KNEE),
        "left_ankle": pt(mp_pose.PoseLandmark.LEFT_ANKLE),

        "right_hip": pt(mp_pose.PoseLandmark.RIGHT_HIP),

        "nose": pt(mp_pose.PoseLandmark.NOSE)
    }


# =========================================================
# VIDEO INPUT
# =========================================================

video_path = "bowling.mp4"

cap = cv2.VideoCapture(video_path)

# Window setup for portrait videos
cv2.namedWindow("Bowling Analyzer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bowling Analyzer", 500, 900)

# =========================================================
# STORAGE
# =========================================================

frames = []

elbow_angles = []
knee_angles = []
shoulder_angles = []

wrist_y_positions = []
nose_x_positions = []

frame_count = 0

# =========================================================
# MAIN LOOP
# =========================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    landmarks = get_landmarks(results, frame)

    if landmarks:

        # =================================================
        # ANGLE CALCULATIONS
        # =================================================

        elbow_angle = calculate_angle(
            landmarks["left_shoulder"],
            landmarks["left_elbow"],
            landmarks["left_wrist"]
        )

        knee_angle = calculate_angle(
            landmarks["left_hip"],
            landmarks["left_knee"],
            landmarks["left_ankle"]
        )

        shoulder_angle = calculate_angle(
            landmarks["left_elbow"],
            landmarks["left_shoulder"],
            landmarks["left_hip"]
        )

        # =================================================
        # STORE DATA
        # =================================================

        elbow_angles.append(elbow_angle)
        knee_angles.append(knee_angle)
        shoulder_angles.append(shoulder_angle)

        wrist_y_positions.append(
            landmarks["left_wrist"][1]
        )

        nose_x_positions.append(
            landmarks["nose"][0]
        )

        frames.append(frame_count)

        # =================================================
        # DRAW POSE
        # =================================================

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # =================================================
        # DISPLAY INFO
        # =================================================

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Elbow: {int(elbow_angle)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # =====================================================
    # RESIZE FRAME FOR PORTRAIT DISPLAY
    # =====================================================

    scale_percent = 70

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

    resized_frame = cv2.resize(frame, (width, height))

    cv2.imshow("Bowling Analyzer", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame({

    "frame": frames,

    "elbow_angle": elbow_angles,
    "knee_angle": knee_angles,
    "shoulder_angle": shoulder_angles,

    "wrist_y": wrist_y_positions,
    "nose_x": nose_x_positions
})


# =========================================================
# SMOOTHING
# =========================================================

df["elbow_smooth"] = (
    df["elbow_angle"]
    .rolling(window=5, min_periods=1)
    .mean()
)

df["knee_smooth"] = (
    df["knee_angle"]
    .rolling(window=5, min_periods=1)
    .mean()
)

df["shoulder_smooth"] = (
    df["shoulder_angle"]
    .rolling(window=5, min_periods=1)
    .mean()
)


# =========================================================
# ANGULAR VELOCITY
# =========================================================

df["elbow_velocity"] = np.gradient(df["elbow_smooth"])
df["shoulder_velocity"] = np.gradient(df["shoulder_smooth"])


# =========================================================
# RELEASE FRAME DETECTION
# =========================================================
# Approximation:
# Highest wrist point = likely release frame

release_index = df["wrist_y"].idxmin()
release_frame = df.loc[release_index, "frame"]


# =========================================================
# PHASE SEGMENTATION
# =========================================================

total_frames = len(df)

runup_end = int(total_frames * 0.40)
gather_end = int(total_frames * 0.55)
delivery_end = int(total_frames * 0.75)

phases = []

for i in range(total_frames):

    if i < runup_end:
        phases.append("Run-up")

    elif i < gather_end:
        phases.append("Gather")

    elif i < delivery_end:
        phases.append("Delivery")

    else:
        phases.append("Follow-through")

df["phase"] = phases


# =========================================================
# PERFORMANCE INSIGHTS
# =========================================================

print("\n=================================================")
print("🏏 CRICKET BOWLING KINEMATICS REPORT")
print("=================================================")

print(f"\nTotal Frames Analyzed: {total_frames}")

print(f"Detected Release Frame: {release_frame}")

print("\n📊 KINEMATIC INSIGHTS\n")

# =========================================================
# CONSISTENCY ANALYSIS
# =========================================================

elbow_std = np.std(df["elbow_smooth"])

if elbow_std < 8:
    print(
        "✔ Elbow motion demonstrates strong movement consistency "
        "throughout the bowling sequence."
    )
else:
    print(
        "⚠ Elbow extension variability detected, indicating possible "
        "release inconsistency."
    )

# =========================================================
# SHOULDER SEQUENCING
# =========================================================

peak_shoulder_velocity_frame = df.iloc[
    df["shoulder_velocity"].idxmax()
]["frame"]

if peak_shoulder_velocity_frame < release_frame:
    print(
        "✔ Shoulder rotational acceleration occurs before release, "
        "suggesting coordinated kinetic sequencing."
    )
else:
    print(
        "⚠ Delayed shoulder acceleration detected near release phase."
    )

# =========================================================
# KNEE STABILITY
# =========================================================

min_knee = df["knee_smooth"].min()

if min_knee > 135:
    print(
        "✔ Front leg stabilization appears structurally stable "
        "during delivery stride."
    )
else:
    print(
        "⚠ Significant knee collapse observed during delivery stride, "
        "possibly reducing momentum transfer efficiency."
    )

# =========================================================
# FOLLOW THROUGH
# =========================================================

follow_variation = np.std(
    df[df["phase"] == "Follow-through"]["shoulder_velocity"]
)

if follow_variation < 5:
    print(
        "✔ Follow-through motion appears smooth with controlled "
        "deceleration mechanics."
    )
else:
    print(
        "⚠ Abrupt deceleration patterns detected during follow-through."
    )


# =========================================================
# DASHBOARD
# =========================================================

fig = plt.figure(figsize=(15, 10))

# =========================================================
# ANGLE GRAPH
# =========================================================

ax1 = plt.subplot(2, 1, 1)

ax1.plot(
    df["frame"],
    df["elbow_smooth"],
    label="Elbow Angle"
)

ax1.plot(
    df["frame"],
    df["knee_smooth"],
    label="Knee Angle"
)

ax1.plot(
    df["frame"],
    df["shoulder_smooth"],
    label="Shoulder Angle"
)

# Release marker
ax1.axvline(
    release_frame,
    linestyle='--',
    label="Release Frame"
)

# Phase shading
ax1.axvspan(0, runup_end, alpha=0.1)
ax1.axvspan(runup_end, gather_end, alpha=0.2)
ax1.axvspan(gather_end, delivery_end, alpha=0.3)
ax1.axvspan(delivery_end, total_frames, alpha=0.1)

ax1.set_title("Bowling Kinematics Analysis")

ax1.set_xlabel("Frame")

ax1.set_ylabel("Angle (Degrees)")

ax1.legend()

ax1.grid(True)


# =========================================================
# ANGULAR VELOCITY GRAPH
# =========================================================

ax2 = plt.subplot(2, 1, 2)

ax2.plot(
    df["frame"],
    df["elbow_velocity"],
    label="Elbow Angular Velocity"
)

ax2.plot(
    df["frame"],
    df["shoulder_velocity"],
    label="Shoulder Angular Velocity"
)

ax2.axvline(
    release_frame,
    linestyle='--',
    label="Release Frame"
)

ax2.set_title("Angular Velocity Analysis")

ax2.set_xlabel("Frame")

ax2.set_ylabel("Angular Velocity")

ax2.legend()

ax2.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# EXPORT CSV
# =========================================================

df.to_csv("bowling_analysis.csv", index=False)

print("\n📁 Analysis exported to bowling_analysis.csv")

print("\n✅ Analysis Complete.")