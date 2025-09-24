# Settings page for user profile and preferences
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    conn = get_postgres_conn()
    cur = conn.cursor()
    if request.method == "POST":
        # Update profile info
        username = request.form.get("username")
        email = request.form.get("email")
        age = request.form.get("age")
        preferred_model = request.form.get("preferred_model")
        notify = 1 if request.form.get("notify") else 0

        # Update user info
        cur.execute("UPDATE users SET username=%s, email=%s, age=%s WHERE id=%s", (username, email, age, current_user.id))

        # Optionally update preferences (add columns if needed)
        try:
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_model TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS notify BOOLEAN DEFAULT FALSE")
        except Exception:
            pass
        cur.execute("UPDATE users SET preferred_model=%s, notify=%s WHERE id=%s", (preferred_model, notify, current_user.id))

        # Change password if requested
        current_password = request.form.get("current_password")
        new_password = request.form.get("new_password")
        confirm_new_password = request.form.get("confirm_new_password")
        if current_password and new_password and new_password == confirm_new_password:
            cur.execute("SELECT password_hash FROM users WHERE id=%s", (current_user.id,))
            row = cur.fetchone()
            if row and check_password_hash(row[0], current_password):
                new_hash = generate_password_hash(new_password)
                cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, current_user.id))
                flash("Password updated.", "success")
            else:
                flash("Current password incorrect.", "error")
        elif new_password or confirm_new_password:
            flash("To change password, fill all password fields and ensure new passwords match.", "error")

        conn.commit()
        flash("Settings updated.", "success")
        cur.close()
        conn.close()
        return redirect(url_for('settings'))

    # GET: fetch user info for form
    cur.execute("SELECT username, email, age, preferred_model, notify FROM users WHERE id=%s", (current_user.id,))
    row = cur.fetchone()
    user = {
        'username': row[0],
        'email': row[1],
        'age': row[2],
        'preferred_model': row[3] or 'gemini-1.5-flash',
        'notify': row[4] if row[4] is not None else False
    }
    cur.close()
    conn.close()
    # Patch current_user for template
    for k, v in user.items():
        setattr(current_user, k, v)
    return render_template("settings.html")
# app.py
from flask import Flask, request, render_template, session, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from dotenv import load_dotenv
import os
import traceback
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2.extras

# Load
# .env (if present)
load_dotenv()

# Import new db connections and utils
from db import get_postgres_conn, get_redis_conn, get_neo4j_driver
from utils import choose_model_heuristic, unified_call, FAST_MODEL, BIG_MODEL
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
token_serializer = URLSafeTimedSerializer(app.secret_key)

# Initialize Redis
redis_conn = get_redis_conn()

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email, age, password_hash, role="user"):
        self.id = id
        self.username = username
        self.email = email
        self.age = age
        self.password_hash = password_hash
        self.role = role

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    conn = get_postgres_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()
    cur.close()
    conn.close()
    if user_data:
        return User(
            id=user_data['id'],
            username=user_data['username'],
            email=user_data['email'],
            age=user_data['age'],
            password_hash=user_data['password_hash'],
            role=user_data.get('role', 'user') if isinstance(user_data, dict) else user_data[5] if len(user_data) > 5 else 'user'
        )
    return None

# Role check decorator
def roles_required(*required_roles):
    def wrapper(fn):
        @login_required
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in required_roles:
                flash("You do not have permission to access this resource.", "error")
                return redirect(url_for('home'))
            return fn(*args, **kwargs)
        # Preserve function name for Flask routing
        decorated.__name__ = fn.__name__
        return decorated
    return wrapper

# Token utilities
def generate_token(user_id, role, expires_minutes=60):
    payload = {"user_id": str(user_id), "role": role}
    return token_serializer.dumps(payload)

def verify_token(token, max_age_seconds=3600):
    try:
        data = token_serializer.loads(token, max_age=max_age_seconds)
        return data
    except SignatureExpired:
        return None
    except BadSignature:
        return None

def token_required(fn):
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
        if not token:
            token = request.args.get('token')
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        data = verify_token(token)
        if not data:
            return jsonify({"error": "Token is invalid or expired"}), 401
        request.user_from_token = data
        return fn(*args, **kwargs)
    decorated.__name__ = fn.__name__
    return decorated

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        conn = get_postgres_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cur.fetchone()
        cur.close()
        conn.close()
        
        if user_data:
            user = User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                age=user_data['age'],
                password_hash=user_data['password_hash'],
                role=user_data.get('role', 'user')
            )
            if user.check_password(password):
                login_user(user)
                # optionally issue token for API usage
                session['api_token'] = generate_token(user.id, user.role)
                return redirect(url_for('chat'))
        
        flash("Invalid username or password", "error")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        age = request.form.get("age")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        conn = get_postgres_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            flash("Username already exists", "error")
        elif password != confirm_password:
            flash("Passwords do not match", "error")
        else:
            password_hash = generate_password_hash(password)
            role = 'user'
            if request.form.get("role") in ("user", "admin"):
                role = request.form.get("role")
            cur.execute(
                "INSERT INTO users (username, email, age, password_hash, role) VALUES (%s, %s, %s, %s, %s)",
                (username, email, int(age) if age else None, password_hash, role)
            )
            conn.commit()
            
            # Also create a user node in Neo4j
            driver = get_neo4j_driver()
            with driver.session() as session:
                session.run("CREATE (u:User {username: $username})", username=username)
            driver.close()

            flash("Account created successfully! Please login.", "success")
            return redirect(url_for('login'))
        
        cur.close()
        conn.close()
    
    return render_template("signup.html")

# Simple admin page protected by role
@app.route("/admin")
@roles_required("admin")
def admin_dashboard():
    return render_template("index.html")

# API to get current token for logged-in users
@app.route("/api/token", methods=["GET"])
@login_required
def get_token():
    token = session.get('api_token') or generate_token(current_user.id, getattr(current_user, 'role', 'user'))
    session['api_token'] = token
    return jsonify({"token": token})

# Example user-protected API endpoint
@app.route("/api/me", methods=["GET"])
@token_required
def api_me():
    data = getattr(request, 'user_from_token', {})
    return jsonify({"user_id": data.get("user_id"), "role": data.get("role")})

# Example admin-protected API endpoint
@app.route("/api/admin/overview", methods=["GET"])
@token_required
def api_admin_overview():
    data = getattr(request, 'user_from_token', {})
    if data.get('role') != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    return jsonify({"status": "ok", "message": "Admin access granted"})

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('home'))

@app.route("/chat")
@login_required
def chat():
    conversation_id = session.get('current_conversation_id')
    conn = get_postgres_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if not conversation_id:
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
            (current_user.id, "New Chat")
        )
        conversation_id = cur.fetchone()['id']
        conn.commit()
        session['current_conversation_id'] = conversation_id

    cur.execute("SELECT * FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, current_user.id))
    conversation = cur.fetchone()

    if not conversation:
        # If conversation doesn't exist or belong to user, create a new one
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
            (current_user.id, "New Chat")
        )
        conversation_id = cur.fetchone()['id']
        conn.commit()
        session['current_conversation_id'] = conversation_id
        cur.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
        conversation = cur.fetchone()

    # Fetch messages for the conversation
    cur.execute("SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC", (conversation_id,))
    messages = cur.fetchall()

    # Get recent conversations from Redis cache or DB
    recent_conversations_cache_key = f"user:{current_user.id}:recent_conversations"
    recent_conversations = redis_conn.get(recent_conversations_cache_key)
    if recent_conversations:
        try:
            # Try to decode from JSON first
            recent_conversations = json.loads(recent_conversations.decode('utf-8'))
        except:
            # If JSON fails, clear cache and fetch from DB
            redis_conn.delete(recent_conversations_cache_key)
            recent_conversations = None
    
    if not recent_conversations:
        cur.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = %s ORDER BY updated_at DESC LIMIT 5",
            (current_user.id,)
        )
        # Convert to list of dictionaries for consistent access
        recent_conversations = []
        for row in cur.fetchall():
            # Convert datetime to string for JSON serialization
            updated_at = row[2]
            if hasattr(updated_at, 'strftime'):
                updated_at = updated_at.strftime('%Y-%m-%d %H:%M:%S')
            
            recent_conversations.append({
                'id': row[0],
                'title': row[1],
                'updated_at': updated_at
            })
        # Cache as JSON
        redis_conn.set(recent_conversations_cache_key, json.dumps(recent_conversations), ex=3600)

    cur.close()
    conn.close()
    
    return render_template(
        "chat.html",
        conversation=conversation,
        messages=messages,
        recent_conversations=recent_conversations
    )

@app.route("/chat/send", methods=["POST"])
@login_required
def send_message():
    conversation_id = session.get('current_conversation_id')
    if not conversation_id:
        flash("No active conversation", "error")
        return redirect(url_for('chat'))

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return redirect(url_for('chat'))

    conn = get_postgres_conn()
    cur = conn.cursor()

    # Verify conversation ownership
    cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
    owner = cur.fetchone()
    if not owner or owner[0] != current_user.id:
        flash("Invalid conversation", "error")
        cur.close()
        conn.close()
        return redirect(url_for('chat'))

    try:
        # History command: return recent messages without calling model
        if prompt.lower() in ("history", "/history"):
            cur.execute(
                "SELECT prompt, response, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 10",
                (conversation_id,)
            )
            rows = cur.fetchall()
            rows = rows[::-1]  # chronological order
            history_lines = []
            for row in rows:
                p = row[0]
                r = row[1]
                ts = row[2].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[2], 'strftime') else str(row[2])
                history_lines.append(f"[{ts}] User: {p}")
                history_lines.append(f"[{ts}] Assistant: {r}")

            history_text = "No previous messages." if not history_lines else "\n".join(history_lines)

            # Save a system-style message that returns history
            cur.execute(
                "INSERT INTO messages (conversation_id, prompt, response, model_used, complexity, task_type, confidence) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (conversation_id, prompt, history_text, 'system-history', 0.0, 'Other', 1.0)
            )
            cur.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s", (conversation_id,))
            conn.commit()

            # Invalidate recent conversations cache
            redis_conn.delete(f"user:{current_user.id}:recent_conversations")

            return redirect(url_for('load_conversation', conversation_id=conversation_id))

        # Get user personalization info
        user_info = f"Username: {current_user.username}"
        if current_user.age:
            user_info += f", Age: {current_user.age}"

        # Fetch last 10 messages for context
        cur.execute(
            "SELECT prompt, response FROM messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 10",
            (conversation_id,)
        )
        previous = cur.fetchall()[::-1]  # chronological
        transcript_lines = []
        for row in previous:
            prior_prompt = row[0]
            prior_response = row[1]
            if prior_prompt:
                transcript_lines.append(f"User: {prior_prompt}")
            if prior_response:
                transcript_lines.append(f"Assistant: {prior_response}")
        conversation_history = "\n".join(transcript_lines)

        # Build contextual prompt
        context_block = (
            f"[Personalization Context: {user_info}]\n" +
            (f"[Conversation History (most recent first limited to 10 messages)]:\n{conversation_history}\n\n" if conversation_history else "")
        )
        personalized_prompt = f"{context_block}Current User Message: {prompt}"

        model_name, model_used = choose_model_heuristic(personalized_prompt)
        assessment, response = unified_call(model_name, personalized_prompt)

        complexity = assessment.get('complexity', 0.5)
        task_type = assessment.get('task_type', 'Other')
        confidence = assessment.get('confidence', 0.5)

        # Save message to PostgreSQL
        cur.execute(
            "INSERT INTO messages (conversation_id, prompt, response, model_used, complexity, task_type, confidence) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (conversation_id, prompt, response, model_used, complexity, task_type, confidence)
        )

        # Update conversation title and timestamp
        cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (conversation_id,))
        message_count = cur.fetchone()[0]
        if message_count == 1:
            new_title = prompt[:50] + ("..." if len(prompt) > 50 else "")
            cur.execute("UPDATE conversations SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s", (new_title, conversation_id))
        else:
            cur.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s", (conversation_id,))

        conn.commit()

        # Invalidate recent conversations cache
        redis_conn.delete(f"user:{current_user.id}:recent_conversations")

        # Add concepts to Neo4j
        driver = get_neo4j_driver()
        with driver.session() as neo4j_session:
            # Simple concept extraction (can be improved with NLP)
            concepts = [word for word in prompt.split() if len(word) > 4] 
            for concept in concepts:
                neo4j_session.run(
                    "MERGE (c:Concept {name: $concept}) "
                    "MERGE (u:User {username: $username}) "
                    "MERGE (u)-[:DISCUSSED]->(c)",
                    concept=concept.lower(), username=current_user.username
                )
        driver.close()

    except Exception as e:
        error_msg = f"Error during inference: {e}\n\n{traceback.format_exc()}"
        flash(error_msg, "error")
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('load_conversation', conversation_id=conversation_id))

@app.route("/chat/<int:conversation_id>")
@login_required
def load_conversation(conversation_id):
    session['current_conversation_id'] = conversation_id
    return redirect(url_for('chat'))

@app.route("/chat/new")
@login_required
def new_conversation():
    session.pop('current_conversation_id', None)
    return redirect(url_for('chat'))

@app.route("/chat/delete/<int:conversation_id>", methods=["POST"])
@login_required
def delete_conversation(conversation_id):
    conn = get_postgres_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, current_user.id))
    conn.commit()
    cur.close()
    conn.close()

    if session.get('current_conversation_id') == str(conversation_id):
        session.pop('current_conversation_id', None)

    # Invalidate recent conversations cache
    redis_conn.delete(f"user:{current_user.id}:recent_conversations")

    flash("Conversation deleted.", "success")
    return redirect(url_for('chat'))

if __name__ == "__main__":
    # Run this once to set up the database
    from db import create_tables, create_neo4j_constraints
    create_tables()
    create_neo4j_constraints()
    app.run(debug=True, host='0.0.0.0', port=5000)