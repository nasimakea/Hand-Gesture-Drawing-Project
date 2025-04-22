from flask import Flask, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/start_drawing')
def start_drawing():
    try:
        script_path = os.path.join("src", "hand_gesture_drawing.py")
        subprocess.Popen(["python", script_path])  # Run the hand tracking script
        return jsonify({"message": "Drawing started! Look at the camera."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
