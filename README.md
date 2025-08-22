🤖 AR Finger Maze Game

An interactive Augmented Reality (AR) game where you navigate a dynamic maze using your hand and a pinch gesture. The game uses real-time hand tracking via your webcam to create a fun, intuitive, and challenging experience.

✨ Features

	•	Real-time Hand Tracking: Uses MediaPipe to accurately track your index finger and thumb.
	•	Pinch-to-Move: The player dot only moves when you perform a "pinch" gesture, providing precise control.
	•	Dynamic Maze Generation: The maze is randomly generated for each new game, ensuring a unique challenge every time.
	•	Progressive Difficulty: As you solve more mazes, the game becomes more complex with larger mazes and narrower paths.
	•	Interactive UI: A clean, modern UI shows your progress, time taken, and difficulty level, with buttons for easy control.
	•	Visual Feedback: The pinch gesture is visualized with a connected line on your fingers, and your path through the maze is highlighted.

🚀 Getting Started


Prerequisites

You'll need Python 3.11+ and a few libraries to run this project.

	1	Install Python: Make sure you have a compatible version of Python installed.
 
	2	Create a Virtual Environment (recommended):
 
    python3 -m venv venv
    source venv/bin/activate	   
    
	3	Install Libraries: 
 
    Use pip to install the required packages. Bash  pip install PyQt5 opencv-python mediapipe numpy

Running the Game

After installing the dependencies, simply run the main Python file.
Bash

python main.py
This will open a window showing your webcam feed.

🎮 How to Play

	1	Start the Game: Place your hand in the camera's view. A message will prompt you to "Place finger at START!".
	2	Move the Dot: The yellow player dot will appear at the start of the maze. To move it, perform a pinch gesture with your index finger and thumb. The dot will follow your pinched fingers.
	3	Stop the Dot: To stop, simply unpinch your fingers. The dot will freeze in place, waiting for your next move.
	4	Navigate the Maze: Guide the dot from the green "START" label to the red "END" label.
	5	Winning: When the dot reaches the "END" cell, you win! The maze will reset, and the difficulty will increase for the next round.

⚙️ Game Controls

The game's UI, including all controls, is built directly into the main window.
	•	RESET: Clears your current path and resets the timer, allowing you to restart the same maze.
	•	NEW MAZE: Generates a new, random maze and restarts the game from the beginning.
	•	SOLUTION: Displays the shortest possible path to solve the current maze. Click again to hide it.
	•	EXIT: Closes the game window.

🔧 Project Structure

The code is organized into a single main.py file using an object-oriented approach. This makes it easy to understand and modify the different components:
	•	Maze: Handles maze generation, drawing, and difficulty scaling.
	•	PlayerDot: Manages the player's position and smooth movement.
	•	HandDetector: Uses MediaPipe and a Kalman Filter for accurate and stable hand tracking.
	•	GameLogic: Contains the core game rules, state management, and win conditions.
	•	WorkerThread: A PyQt5 thread that handles the camera feed and processing to keep the UI responsive.
	•	MainWindow: The main PyQt5 window that displays all UI elements.
