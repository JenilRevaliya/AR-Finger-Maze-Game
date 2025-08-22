import sys
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import deque
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFrame, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# Constants
WIDTH, HEIGHT = 1280, 720
MAZE_ROWS, MAZE_COLS = 15, 15
WALL_OPACITY = 0.4
MAX_PINCH_DIST = 0.08
MAX_DIST = 100
CROP_FACTOR = 0.6
MIN_WALL_THICKNESS = 1
MAX_DIFFICULTY = 10

# Game States
GAME_STATE_PLAYING = 0
GAME_STATE_WIN = 1
GAME_STATE_START = 3
GAME_STATE_SOLVING = 4

class Maze:
    """Manages maze generation and drawing."""
    def __init__(self, solved_count, frame_width, frame_height):
        difficulty_level = min(solved_count * 0.5, MAX_DIFFICULTY)
        size_increase = int(difficulty_level * 2)
        self.size = (MAZE_ROWS + size_increase, MAZE_COLS + size_increase)
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.start_cell = (1, 1)
        self.end_cell = (self.size[0] - 2, self.size[1] - 2)
        
        self.grid = self.generate_recursive_backtracker()
        self.cell_width = self.frame_width // self.size[1]
        self.cell_height = self.frame_height // self.size[0]
        
        self.wall_thickness = 1.2
        
        self.solution_path = self.find_solution()
        self.total_solution_len = len(self.solution_path) if self.solution_path else 0

    def generate_recursive_backtracker(self):
        """Generates a maze using the recursive backtracking algorithm,
        guaranteeing a single solution path."""
        rows, cols = self.size
        grid = np.ones(self.size, dtype=np.int8)
        stack = []
        
        start_node = self.start_cell
        grid[start_node] = 0
        stack.append(start_node)

        while stack:
            current = stack[-1]
            y, x = current
            
            neighbors = []
            if y > 1 and grid[y-2, x] == 1:
                neighbors.append((y-2, x))
            if y < rows - 2 and grid[y+2, x] == 1:
                neighbors.append((y+2, x))
            if x > 1 and grid[y, x-2] == 1:
                neighbors.append((y, x-2))
            if x < cols - 2 and grid[y, x+2] == 1:
                neighbors.append((y, x+2))
            
            if neighbors:
                next_node = random.choice(neighbors)
                ny, nx = next_node
                if ny == y - 2:
                    grid[y-1, x] = 0
                elif ny == y + 2:
                    grid[y+1, x] = 0
                elif nx == x - 2:
                    grid[y, x-1] = 0
                elif nx == x + 2:
                    grid[y, x+1] = 0
                
                grid[next_node] = 0
                stack.append(next_node)
            else:
                stack.pop()
        
        grid[self.start_cell[0], self.start_cell[1] - 1] = 0
        grid[self.end_cell[0], self.end_cell[1] + 1] = 0
        
        return grid

    def find_solution(self):
        """Finds the solution path using Breadth-First Search (BFS)."""
        rows, cols = self.size
        queue = deque([(self.start_cell, [self.start_cell])])
        visited = {self.start_cell}
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.end_cell:
                return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(((nr, nc), new_path))
        return None

    def draw(self, frame):
        """Draws the transparent maze walls with thin black borders."""
        overlay = frame.copy()
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.grid[r, c] == 1:
                    cv2.rectangle(overlay, (c * self.cell_width, r * self.cell_height),
                                  ((c + 1) * self.cell_width, (r + 1) * self.cell_height),
                                  (255, 255, 255), -1)
        cv2.addWeighted(overlay, WALL_OPACITY, frame, 1 - WALL_OPACITY, 0, frame)
        
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.grid[r, c] == 1:
                    if r > 0 and self.grid[r-1, c] == 0:
                        cv2.line(frame, (c * self.cell_width, r * self.cell_height), ((c+1) * self.cell_width, r * self.cell_height), (0, 0, 0), int(self.wall_thickness))
                    if r < self.size[0]-1 and self.grid[r+1, c] == 0:
                        cv2.line(frame, (c * self.cell_width, (r+1) * self.cell_height), ((c+1) * self.cell_width, (r+1) * self.cell_height), (0, 0, 0), int(self.wall_thickness))
                    if c > 0 and self.grid[r, c-1] == 0:
                        cv2.line(frame, (c * self.cell_width, r * self.cell_height), (c * self.cell_width, (r+1) * self.cell_height), (0, 0, 0), int(self.wall_thickness))
                    if c < self.size[1]-1 and self.grid[r, c+1] == 0:
                        cv2.line(frame, ((c+1) * self.cell_width, r * self.cell_height), ((c+1) * self.cell_width, (r+1) * self.cell_height), (0, 0, 0), int(self.wall_thickness))
        
        start_pos = (self.start_cell[1] * self.cell_width, self.start_cell[0] * self.cell_height)
        end_pos = (self.end_cell[1] * self.cell_width, self.end_cell[0] * self.cell_height)
        cv2.putText(frame, "START", (start_pos[0] + 5, start_pos[1] + int(self.cell_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "END", (end_pos[0] + 5, end_pos[1] + int(self.cell_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

class PlayerDot:
    """Represents and manages the player's dot."""
    def __init__(self, start_pos, maze_cell_size):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(start_pos, dtype=np.float32)
        self.maze_cell_size = maze_cell_size
        self.smoothing_factor = 0.15

    def update(self, new_target):
        if new_target is not None:
            cell_x = new_target[0] // self.maze_cell_size[0]
            cell_y = new_target[1] // self.maze_cell_size[1]
            self.target_pos = np.array([
                cell_x * self.maze_cell_size[0] + self.maze_cell_size[0] / 2,
                cell_y * self.maze_cell_size[1] + self.maze_cell_size[1] / 2
            ], dtype=np.float32)
        self.pos = self.pos + (self.target_pos - self.pos) * self.smoothing_factor

    def draw(self, frame):
        cv2.circle(frame, tuple(self.pos.astype(int)), 10, (0, 255, 255), -1)

class HandDetector:
    """Detects and tracks the hand and finger landmarks."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.last_pos = None

    def detect_hand(self, frame):
        """Processes the frame to find hand landmarks and applies a Kalman filter."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            pos = np.array([[index_tip.x * frame.shape[1]], [index_tip.y * frame.shape[0]]], dtype=np.float32)
            
            if self.last_pos is None:
                self.kf.statePost = np.array([[pos[0, 0]], [pos[1, 0]], [0], [0]], np.float32)
                self.last_pos = pos
            
            self.kf.predict()
            
            measurement = np.array([[pos[0,0]], [pos[1,0]]], dtype=np.float32)
            estimated = self.kf.correct(measurement)
            
            estimated_landmarks = landmarks
            estimated_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x = estimated[0, 0] / frame.shape[1]
            estimated_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y = estimated[1, 0] / frame.shape[0]
            
            return estimated_landmarks
        
        self.last_pos = None
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        
        return None

class GameLogic:
    """Manages the game state, path tracking, and movement rules."""
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.game_state = GAME_STATE_START
        self.is_pinching = False
        self.player_dot = PlayerDot(
            (self.maze.start_cell[1] * self.maze.cell_width + self.maze.cell_width / 2,
             self.maze.start_cell[0] * self.maze.cell_height + self.maze.cell_height / 2),
            (self.maze.cell_width, self.maze.cell_height)
        )
        self.last_cell = None
        self.show_solution = False

    def map_coords_to_cell(self, coords):
        """Maps pixel coordinates to maze grid cell."""
        if coords is None:
            return None
        x, y = coords
        col = x // self.maze.cell_width
        row = y // self.maze.cell_height
        row = max(0, min(row, self.maze.size[0] - 1))
        col = max(0, min(col, self.maze.size[1] - 1))
        return (row, col)

    def get_progress(self):
        if len(self.current_path) == 0 or not self.maze.solution_path:
            return 0
        current_cell = self.current_path[-1]
        solution_index = next((i for i, cell in enumerate(self.maze.solution_path) if cell == current_cell), -1)
        if solution_index == -1:
            return 0
        progress = (solution_index / (len(self.maze.solution_path) - 1)) * 100 if len(self.maze.solution_path) > 1 else 100
        return int(progress)

    def update(self, landmarks, is_pinching, index_coords_transformed):
        self.is_pinching = is_pinching
        
        if self.game_state == GAME_STATE_START:
            if is_pinching and self.map_coords_to_cell(index_coords_transformed) == self.maze.start_cell:
                self.game_state = GAME_STATE_PLAYING
                self.current_path.append(self.maze.start_cell)
            self.player_dot.update(index_coords_transformed)
            return

        if self.game_state == GAME_STATE_WIN:
            return

        if self.is_pinching:
            if len(self.current_path) > 0 and np.linalg.norm(np.array(index_coords_transformed) - self.player_dot.pos) > 50:
                self.player_dot.update(None)
                return

            self.player_dot.update(index_coords_transformed)
            
            current_cell_tuple = self.map_coords_to_cell(tuple(self.player_dot.pos))

            if current_cell_tuple is None or current_cell_tuple == self.last_cell:
                return

            current_cell_int = (int(current_cell_tuple[0]), int(current_cell_tuple[1]))
            
            if not (0 <= current_cell_int[0] < self.maze.size[0] and 0 <= current_cell_int[1] < self.maze.size[1]):
                return
            
            if self.maze.grid[current_cell_int] == 1:
                return

            if self.last_cell and abs(current_cell_int[0] - self.last_cell[0]) + abs(current_cell_int[1] - self.last_cell[1]) > 1:
                return

            self.last_cell = current_cell_int
            self.current_path.append(current_cell_int)

            if current_cell_int == self.maze.end_cell:
                self.game_state = GAME_STATE_WIN
        else:
            self.player_dot.update(None)

    def draw_path(self, frame):
        overlay = frame.copy()
        
        if self.show_solution and self.maze.solution_path:
            for i in range(len(self.maze.solution_path) - 1):
                cell1 = self.maze.solution_path[i]
                cell2 = self.maze.solution_path[i+1]
                center1 = (int(cell1[1] * self.maze.cell_width + self.maze.cell_width / 2),
                           int(cell1[0] * self.maze.cell_height + self.maze.cell_height / 2))
                center2 = (int(cell2[1] * self.maze.cell_width + self.maze.cell_width / 2),
                           int(cell2[0] * self.maze.cell_height + self.maze.cell_height / 2))
                cv2.line(overlay, center1, center2, (255, 0, 0), 2)
        
        if self.current_path:
            for i in range(len(self.current_path) - 1):
                cell1 = self.current_path[i]
                cell2 = self.current_path[i+1]
                center1 = (int(cell1[1] * self.maze.cell_width + self.maze.cell_width / 2),
                           int(cell1[0] * self.maze.cell_height + self.maze.cell_height / 2))
                center2 = (int(cell2[1] * self.maze.cell_width + self.maze.cell_width / 2),
                           int(cell2[0] * self.maze.cell_height + self.maze.cell_height / 2))
                cv2.line(overlay, center1, center2, (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

class WorkerThread(QThread):
    """Handles the video stream, game logic, and hand tracking."""
    image_updated = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)
    fps_updated = pyqtSignal(int)
    message_updated = pyqtSignal(str, str)
    solved_count_updated = pyqtSignal(int)
    time_updated = pyqtSignal(str)
    complexity_updated = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.hand_detector = HandDetector()
        
        self.cropped_w = int(WIDTH * CROP_FACTOR)
        self.cropped_h = int(HEIGHT * CROP_FACTOR)
        
        self.mazes_solved = 0
        self.maze = Maze(solved_count=self.mazes_solved, frame_width=self.cropped_w, frame_height=self.cropped_h)
        self.game_logic = GameLogic(self.maze)
        self.is_running = True
        
        self.start_time = time.time()
        self.time_taken = 0

    def run(self):
        prev_time = time.time()
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            landmarks = self.hand_detector.detect_hand(frame)
            
            frame_h, frame_w, _ = frame.shape
            start_x = (frame_w - self.cropped_w) // 2
            start_y = (frame_h - self.cropped_h) // 2
            
            cropped_frame = frame[start_y:start_y + self.cropped_h, start_x:start_x + self.cropped_w].copy()
            
            cropped_frame = cv2.flip(cropped_frame, 1)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            self.fps_updated.emit(int(fps))
            
            is_pinching = False
            index_coords_transformed = None
            if landmarks:
                h_full, w_full = frame.shape[:2]
                
                index_tip_lm = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip_lm = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
                dist = np.sqrt((index_tip_lm.x - thumb_tip_lm.x)**2 + (index_tip_lm.y - thumb_tip_lm.y)**2)
                if dist < MAX_PINCH_DIST:
                    is_pinching = True

                line_color = (255, 255, 255) if is_pinching else (0, 0, 255)
                
                def get_flipped_cropped_coords(landmark):
                    original_x_px = int(landmark.x * w_full)
                    original_y_px = int(landmark.y * h_full)
                    
                    flipped_x_px = w_full - original_x_px
                    
                    cropped_x_px = flipped_x_px - start_x
                    cropped_y_px = original_y_px - start_y
                    return (cropped_x_px, cropped_y_px)

                p1 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP])
                p2 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP])
                p3 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP])
                p4 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP])
                p5 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.THUMB_IP])
                p6 = get_flipped_cropped_coords(landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP])
                
                index_coords_transformed = p1
                
                for p in [p1, p2, p3, p4, p5, p6]:
                    cv2.circle(cropped_frame, p, 5, (255, 0, 0), -1)
                
                cv2.line(cropped_frame, p1, p2, line_color, 2)
                cv2.line(cropped_frame, p2, p3, line_color, 2)
                cv2.line(cropped_frame, p3, p4, line_color, 2)
                cv2.line(cropped_frame, p4, p5, line_color, 2)
                cv2.line(cropped_frame, p5, p6, line_color, 2)
            else:
                index_coords_transformed = None
            
            old_game_state = self.game_logic.game_state
            self.game_logic.update(landmarks, is_pinching, index_coords_transformed)
            
            if self.game_logic.game_state == GAME_STATE_PLAYING:
                self.time_taken = time.time() - self.start_time
                self.message_updated.emit("", "white")
            elif self.game_logic.game_state == GAME_STATE_WIN and old_game_state != GAME_STATE_WIN:
                self.mazes_solved += 1
                self.solved_count_updated.emit(self.mazes_solved)
                self.time_taken = time.time() - self.start_time
                self.message_updated.emit("YOU WIN!", "lightgreen")
            
            if self.game_logic.get_progress() == 0 and self.game_logic.current_path and self.game_logic.game_state != GAME_STATE_START:
                self.message_updated.emit("Restart to try again!", "red")
            
            if self.game_logic.game_state == GAME_STATE_START:
                self.message_updated.emit("Place finger at START!", "cyan")
                self.start_time = time.time()
                self.time_taken = 0
            elif self.game_logic.game_state == GAME_STATE_PLAYING:
                if landmarks and not is_pinching:
                    self.message_updated.emit("Pinch to Move!", "yellow")
                else:
                    self.message_updated.emit("", "white")
            elif self.game_logic.game_state == GAME_STATE_SOLVING:
                 self.message_updated.emit("SOLUTION", "lightblue")
            
            self.time_updated.emit(f"{self.time_taken:.2f} s")
            self.complexity_updated.emit(int(min(self.mazes_solved * 0.5, MAX_DIFFICULTY)))
            
            self.maze.draw(cropped_frame)
            self.game_logic.draw_path(cropped_frame)
            self.game_logic.player_dot.draw(cropped_frame)

            self.image_updated.emit(cropped_frame)
            self.progress_updated.emit(self.game_logic.get_progress())

    def reset_game(self):
        # Reset the logic but keep the same maze
        self.game_logic = GameLogic(self.maze)
        self.game_logic.game_state = GAME_STATE_START
        self.game_logic.show_solution = False
        self.start_time = time.time()
        self.time_taken = 0
        self.message_updated.emit("Maze reset!", "yellow")

    def new_maze(self):
        if self.game_logic.game_state == GAME_STATE_WIN:
            self.mazes_solved += 1
            self.message_updated.emit("Complexity Increased!", "orange")
        
        self.maze = Maze(solved_count=self.mazes_solved, frame_width=self.cropped_w, frame_height=self.cropped_h)
        self.game_logic = GameLogic(self.maze)
        self.game_logic.game_state = GAME_STATE_START
        self.game_logic.show_solution = False
        self.start_time = time.time()
        self.time_taken = 0
        self.message_updated.emit("New maze generated!", "cyan")
        
    def show_solution(self):
        if self.game_logic.game_state != GAME_STATE_SOLVING:
            self.game_logic.show_solution = True
            self.game_logic.game_state = GAME_STATE_SOLVING
            self.message_updated.emit("SOLUTION", "lightblue")
        else:
            self.game_logic.show_solution = False
            self.game_logic.game_state = GAME_STATE_START
            self.message_updated.emit("Solution cleared", "yellow")

    def stop(self):
        self.is_running = False
        self.cap.release()

class MainWindow(QMainWindow):
    """The main PyQt5 window displaying the video feed."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR Finger Maze Game")
        self.setGeometry(100, 100, WIDTH, HEIGHT + 100)
        self.setFixedSize(WIDTH, HEIGHT + 100)
        self.setStyleSheet("background-color: #222; color: white;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.top_bar = QFrame()
        self.top_bar.setStyleSheet("background-color: #111; border: 1px solid #555; padding: 5px;")
        self.top_bar_layout = QHBoxLayout(self.top_bar)
        
        self.solved_label = QLabel("Solved: 0")
        self.solved_label.setFont(QFont('Arial', 14))
        self.time_label = QLabel("Time: 0.00 s")
        self.time_label.setFont(QFont('Arial', 14))
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont('Arial', 14))
        self.complexity_label = QLabel("Complexity: 0")
        self.complexity_label.setFont(QFont('Arial', 14))
        
        self.top_bar_layout.addWidget(self.solved_label)
        self.top_bar_layout.addWidget(self.time_label)
        self.top_bar_layout.addWidget(self.complexity_label)
        self.top_bar_layout.addWidget(self.fps_label)
        self.main_layout.addWidget(self.top_bar)

        self.message_label = QLabel("")
        self.message_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.message_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.message_label)
        
        self.video_container = QFrame()
        self.video_container.setFrameShape(QFrame.StyledPanel)
        self.video_container.setLineWidth(2)
        self.video_container.setStyleSheet("border: 2px solid #555;")
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.video_label)
        self.main_layout.addWidget(self.video_container)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { height: 20px; border: 2px solid #555; border-radius: 5px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #00ff00; }
        """)
        self.main_layout.addWidget(self.progress_bar)

        self.bottom_bar = QFrame()
        self.bottom_bar.setStyleSheet("background-color: #111; border: 1px solid #555; padding: 5px;")
        self.bottom_bar_layout = QHBoxLayout(self.bottom_bar)
        
        self.reset_btn = QPushButton("RESET")
        self.new_maze_btn = QPushButton("NEW MAZE")
        self.solution_btn = QPushButton("SOLUTION")
        self.exit_btn = QPushButton("EXIT")
        
        self.set_button_style(self.reset_btn, "#4CAF50")
        self.set_button_style(self.new_maze_btn, "#2196F3")
        self.set_button_style(self.solution_btn, "#FFC107")
        self.set_button_style(self.exit_btn, "#f44336")
        
        self.bottom_bar_layout.addWidget(self.reset_btn)
        self.bottom_bar_layout.addWidget(self.new_maze_btn)
        self.bottom_bar_layout.addWidget(self.solution_btn)
        self.bottom_bar_layout.addWidget(self.exit_btn)
        self.main_layout.addWidget(self.bottom_bar)

        self.worker_thread = WorkerThread()
        self.worker_thread.image_updated.connect(self.update_image)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.fps_updated.connect(lambda fps: self.fps_label.setText(f"FPS: {fps}"))
        self.worker_thread.solved_count_updated.connect(lambda count: self.solved_label.setText(f"Solved: {count}"))
        self.worker_thread.time_updated.connect(lambda time_str: self.time_label.setText(f"Time: {time_str}"))
        self.worker_thread.message_updated.connect(self.update_message)
        self.worker_thread.complexity_updated.connect(lambda complexity: self.complexity_label.setText(f"Complexity: {complexity}"))
        self.worker_thread.start()

        self.reset_btn.clicked.connect(self.worker_thread.reset_game)
        self.new_maze_btn.clicked.connect(self.worker_thread.new_maze)
        self.solution_btn.clicked.connect(self.worker_thread.show_solution)
        self.exit_btn.clicked.connect(self.close)

    def set_button_style(self, button, color):
        button.setFont(QFont('Arial', 12, QFont.Bold))
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 5px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {color};
                opacity: 0.8;
            }}
        """)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_message(self, text, color):
        self.message_label.setText(text)
        self.message_label.setStyleSheet(f"color: {color};")
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = self.video_label.size()
        p = convert_to_qt_format.scaled(label_size.width(), label_size.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.worker_thread.stop()
        self.worker_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())