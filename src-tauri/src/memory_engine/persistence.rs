//! SQLite Persistence Layer — Single Source of Truth Storage
//! Sarathi owns 100% of storage, schema migrations, vector blobs, and transactions.

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeRecord {
    pub id: String,
    pub memory_type: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub content: String,
    pub importance_score: f64,
    pub recency_timestamp: i64,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfileRecord {
    pub key: String,
    pub value: String,
    pub category: String,
    pub confidence: f64,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryRecord {
    pub project_id: String,
    pub core_persona: String,
    pub user_facts_summary: String,
    pub active_task: Option<String>,
    pub updated_at: String,
}

pub struct PersistenceManager {
    db_path: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl PersistenceManager {
    pub fn new(app_data_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let db_path = app_data_dir.join("sarathi.db");
        let conn = Connection::open(&db_path)?;

        // Ensure Phase 6 schema migrations exist
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                session_id TEXT,
                content TEXT NOT NULL,
                importance_score REAL DEFAULT 0.5,
                recency_timestamp INTEGER NOT NULL,
                embedding_blob BLOB,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                summary_text TEXT NOT NULL,
                turn_start INTEGER NOT NULL,
                turn_end INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS working_memory (
                project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                core_persona TEXT NOT NULL,
                user_facts_summary TEXT NOT NULL,
                active_task TEXT,
                updated_at TEXT NOT NULL
            );

            INSERT OR IGNORE INTO projects (id, name, description, created_at, updated_at)
            VALUES ('proj_general', 'General', 'Default workspace project', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        ")?;

        Ok(Self {
            db_path,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Inserts or updates a memory node
    pub fn save_memory_node(&self, record: &MemoryNodeRecord) -> Result<()> {
        let start_time = std::time::Instant::now();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memory_nodes (id, memory_type, project_id, session_id, content, importance_score, recency_timestamp, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
             content=excluded.content,
             importance_score=excluded.importance_score,
             recency_timestamp=excluded.recency_timestamp,
             updated_at=excluded.updated_at",
            params![
                record.id,
                record.memory_type,
                record.project_id,
                record.session_id,
                record.content,
                record.importance_score,
                record.recency_timestamp,
                record.metadata,
                record.created_at,
                record.updated_at
            ],
        )?;
        log::info!(
            "[MEMORY_PERSISTENCE] Table 'memory_nodes' INSERT/UPDATE succeeded in {}ms: ID='{}', Type='{}', Project={:?}, Importance={:.2}",
            start_time.elapsed().as_millis(), record.id, record.memory_type, record.project_id, record.importance_score
        );
        Ok(())
    }

    /// Queries memory nodes filtered by project ID
    pub fn get_memories_by_project(&self, project_id: Option<&str>, limit: usize) -> Result<Vec<MemoryNodeRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = if let Some(pid) = project_id {
            conn.prepare(
                "SELECT id, memory_type, project_id, session_id, content, importance_score, recency_timestamp, metadata, created_at, updated_at
                 FROM memory_nodes WHERE project_id = ?1 OR project_id IS NULL ORDER BY recency_timestamp DESC LIMIT ?2"
            )?
        } else {
            conn.prepare(
                "SELECT id, memory_type, project_id, session_id, content, importance_score, recency_timestamp, metadata, created_at, updated_at
                 FROM memory_nodes ORDER BY recency_timestamp DESC LIMIT ?1"
            )?
        };

        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<MemoryNodeRecord> {
            Ok(MemoryNodeRecord {
                id: row.get(0)?,
                memory_type: row.get(1)?,
                project_id: row.get(2)?,
                session_id: row.get(3)?,
                content: row.get(4)?,
                importance_score: row.get(5)?,
                recency_timestamp: row.get(6)?,
                metadata: row.get(7)?,
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        };

        let mut list = Vec::new();
        if let Some(pid) = project_id {
            for r in stmt.query_map(params![pid, limit as i64], map_row)? {
                list.push(r?);
            }
        } else {
            for r in stmt.query_map(params![limit as i64], map_row)? {
                list.push(r?);
            }
        };

        Ok(list)
    }

    /// User profile CRUD
    pub fn save_user_profile_fact(&self, key: &str, value: &str, category: &str) -> Result<()> {
        let start_time = std::time::Instant::now();
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO user_profile (key, value, category, confidence, updated_at)
             VALUES (?1, ?2, ?3, 1.0, ?4)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            params![key, value, category, now],
        )?;
        log::info!(
            "[MEMORY_PERSISTENCE] Table 'user_profile' UPSERT succeeded in {}ms: Key='{}', Value='{}', Category='{}'",
            start_time.elapsed().as_millis(), key, value, category
        );
        Ok(())
    }

    pub fn get_user_profile(&self) -> Result<Vec<UserProfileRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT key, value, category, confidence, updated_at FROM user_profile")?;
        let rows = stmt.query_map([], |row| {
            Ok(UserProfileRecord {
                key: row.get(0)?,
                value: row.get(1)?,
                category: row.get(2)?,
                confidence: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    /// Projects CRUD
    pub fn create_project(&self, id: &str, name: &str, description: Option<&str>) -> Result<ProjectRecord> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, name, description, now, now],
        )?;
        Ok(ProjectRecord {
            id: id.to_string(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, name, description, created_at, updated_at FROM projects ORDER BY name ASC")?;
        let rows = stmt.query_map([], |row| {
            Ok(ProjectRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn delete_memory_node(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM memory_nodes WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn get_memory_counts(&self) -> Result<(usize, usize, usize)> {
        let conn = self.conn.lock().unwrap();
        let nodes: usize = conn.query_row("SELECT COUNT(*) FROM memory_nodes", [], |r| r.get(0)).unwrap_or(0);
        let profile: usize = conn.query_row("SELECT COUNT(*) FROM user_profile", [], |r| r.get(0)).unwrap_or(0);
        let projects: usize = conn.query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0)).unwrap_or(0);
        Ok((nodes, profile, projects))
    }
}
