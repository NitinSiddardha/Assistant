import os
import time
import psycopg2
import redis
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# PostgreSQL Connection
def get_postgres_conn():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=int(os.getenv("POSTGRES_PORT"))  # Convert port to int
    )
    return conn

# Redis Connection
def get_redis_conn():
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),  # Convert port to int
        db=int(os.getenv("REDIS_DB"))
    )
    return r

# Neo4j Connection
def get_neo4j_driver():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    return driver

# Create PostgreSQL tables
def create_tables():
    conn = get_postgres_conn()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            age INTEGER,
            password_hash VARCHAR(200) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'user',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Ensure role column exists for existing installations
    cur.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'user';
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(200) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            model_used VARCHAR(100) NOT NULL,
            complexity FLOAT,
            task_type VARCHAR(100),
            confidence FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# Create Neo4j constraints
def create_neo4j_constraints(max_attempts: int = 30, delay_seconds: float = 2.0):
    """Create constraints with retry until Neo4j is ready."""
    attempt = 0
    last_error = None
    while attempt < max_attempts:
        try:
            driver = get_neo4j_driver()
            with driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            driver.close()
            return
        except ServiceUnavailable as e:
            last_error = e
            time.sleep(delay_seconds)
            attempt += 1
        except Exception as e:
            # For any other transient startup errors, retry as well
            last_error = e
            time.sleep(delay_seconds)
            attempt += 1
    raise RuntimeError(f"Neo4j not ready after {max_attempts} attempts: {last_error}")

if __name__ == '__main__':
    create_tables()
    create_neo4j_constraints()
    print("PostgreSQL tables and Neo4j constraints created successfully.")
