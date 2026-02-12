from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data (like a mini database)
tasks = [
    {"id": 1, "task": "Learn Flask", "done": False},
    {"id": 2, "task": "Build REST API", "done": False},
]

@app.route("/")
def hello():
    return jsonify("Hello, World!")

# GET all tasks
@app.route("/tasks", methods=["GET"])
def get_tasks():
    return jsonify(tasks)

# GET a task by id
@app.route("/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id):
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404

# POST a new task
@app.route("/tasks", methods=["POST"])
def create_task():
    data = request.get_json()
    new_task = {
        "id": len(tasks) + 1,
        "task": data.get("task", ""),
        "done": False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

# PUT to update a task
@app.route("/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id):
    data = request.get_json()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task:
        task["task"] = data.get("task", task["task"])
        task["done"] = data.get("done", task["done"])
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404

# DELETE a task
@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    global tasks
    tasks = [t for t in tasks if t["id"] != task_id]
    return jsonify({"message": "Task deleted"}), 200

if __name__ == "__main__":
    app.run(debug=True)
