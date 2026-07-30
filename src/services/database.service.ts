import Database from '@tauri-apps/plugin-sql';
import { Setting, ActivityLogEntry } from '../types/database';

let dbInstance: Database | null = null;

export async function getDatabase(): Promise<Database> {
  if (!dbInstance) {
    dbInstance = await Database.load('sqlite:sarathi.db');
  }
  return dbInstance;
}

export async function getSetting(key: string): Promise<Setting | null> {
  const db = await getDatabase();
  const results = await db.select<Setting[]>('SELECT * FROM settings WHERE key = $1 LIMIT 1', [key]);
  return results.length > 0 ? results[0] : null;
}

export async function setSetting(key: string, value: string, type: string): Promise<void> {
  const db = await getDatabase();
  await db.execute('INSERT OR REPLACE INTO settings (key, value, type, updated_at) VALUES ($1, $2, $3, datetime("now"))', [key, value, type]);
}

export async function getAllSettings(): Promise<Setting[]> {
  const db = await getDatabase();
  return db.select<Setting[]>('SELECT * FROM settings');
}

export async function logActivity(action: string, category: string, details: string): Promise<void> {
  const db = await getDatabase();
  await db.execute('INSERT INTO activity_log (action, category, details, created_at) VALUES ($1, $2, $3, datetime("now"))', [action, category, details]);
}

export async function getRecentActivity(limit: number): Promise<ActivityLogEntry[]> {
  const db = await getDatabase();
  return db.select<ActivityLogEntry[]>('SELECT * FROM activity_log ORDER BY created_at DESC LIMIT $1', [limit]);
}