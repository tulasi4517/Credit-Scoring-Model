import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/credit_scoring_db")
DB_NAME = os.getenv("DB_NAME", "credit_scoring_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "credit_assessments")

# In-memory storage fallback if MongoDB Atlas/local server is offline
_in_memory_assessments = []
_db_client = None
_db_collection = None
_is_connected = False
_connection_status_msg = "Initializing database..."
_is_atlas = "mongodb+srv" in MONGO_URI.lower()

try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

def init_db():
    """Initializes MongoDB Atlas / MongoDB client connection."""
    global _db_client, _db_collection, _is_connected, _connection_status_msg, _is_atlas
    
    if not PYMONGO_AVAILABLE:
        _is_connected = False
        _connection_status_msg = "PyMongo package missing. Running in Fallback Mode."
        print(f"[DB WARN] {_connection_status_msg}")
        return False

    try:
        # 3-second timeout for server selection check
        _db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Verify connection by running ping command
        _db_client.admin.command('ping')
        
        db = _db_client[DB_NAME]
        _db_collection = db[COLLECTION_NAME]
        # Create index on created_at for fast retrieval
        _db_collection.create_index([("created_at", -1)])
        
        _is_connected = True
        target_type = "MongoDB Atlas Cloud" if _is_atlas else "MongoDB Server"
        _connection_status_msg = f"Connected to {target_type} ({DB_NAME})"
        print(f"[DB SUCCESS] {_connection_status_msg}")
        return True
    except Exception as e:
        _is_connected = False
        _db_client = None
        _db_collection = None
        _connection_status_msg = f"Atlas Offline (Using Memory Store): {str(e)[:80]}"
        print(f"[DB WARN] Could not connect to MongoDB ({MONGO_URI}): {e}")
        print("[DB INFO] System will safely persist assessments in memory store.")
        return False

def save_assessment(assessment_data):
    """Saves a credit risk assessment document to MongoDB Atlas or fallback store."""
    record = dict(assessment_data)
    record["created_at"] = datetime.utcnow().isoformat()
    record["timestamp_epoch"] = time.time()
    
    if _is_connected and _db_collection is not None:
        try:
            res = _db_collection.insert_one(record)
            record["_id"] = str(res.inserted_id)
            return record
        except Exception as e:
            print(f"[DB ERROR] Failed to write to MongoDB: {e}")
            
    # Fallback in-memory persistence
    record["_id"] = f"mem_{int(time.time() * 1000)}"
    _in_memory_assessments.insert(0, record)
    return record

def get_assessments(limit=50, risk_filter=None):
    """Retrieves recent credit assessments from MongoDB Atlas or fallback store."""
    if _is_connected and _db_collection is not None:
        try:
            query = {}
            if risk_filter:
                query["risk_band"] = risk_filter
                
            cursor = _db_collection.find(query).sort("created_at", -1).limit(limit)
            records = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                records.append(doc)
            return records
        except Exception as e:
            print(f"[DB ERROR] Failed to fetch from MongoDB: {e}")
            
    # Fallback read from memory
    filtered = _in_memory_assessments
    if risk_filter:
        filtered = [r for r in _in_memory_assessments if r.get("risk_band") == risk_filter]
    return filtered[:limit]

def delete_assessment(record_id):
    """Deletes an assessment record by ID from MongoDB Atlas or memory store."""
    if _is_connected and _db_collection is not None:
        try:
            if PYMONGO_AVAILABLE and ObjectId.is_valid(record_id):
                res = _db_collection.delete_one({"_id": ObjectId(record_id)})
            else:
                res = _db_collection.delete_one({"_id": record_id})
            return res.deleted_count > 0
        except Exception as e:
            print(f"[DB ERROR] Failed to delete record from MongoDB: {e}")
            
    # Fallback memory delete
    global _in_memory_assessments
    initial_len = len(_in_memory_assessments)
    _in_memory_assessments = [r for r in _in_memory_assessments if r.get("_id") != record_id]
    return len(_in_memory_assessments) < initial_len

def get_db_status():
    """Returns database health diagnostics and record statistics."""
    count = 0
    if _is_connected and _db_collection is not None:
        try:
            count = _db_collection.count_documents({})
        except Exception:
            count = len(_in_memory_assessments)
    else:
        count = len(_in_memory_assessments)

    return {
        "is_connected": _is_connected,
        "is_atlas": _is_atlas,
        "database_name": DB_NAME,
        "collection_name": COLLECTION_NAME,
        "status_message": _connection_status_msg,
        "total_records": count,
        "mongo_uri_masked": MONGO_URI.split("@")[-1] if "@" in MONGO_URI else MONGO_URI
    }

# Run initialization upon module load
init_db()
