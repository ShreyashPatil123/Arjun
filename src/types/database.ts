export interface Setting {
  key: string;
  value: string;
  type: string;
  updated_at: string;
}

export interface ActivityLogEntry {
  id: number;
  action: string;
  category: string;
  details: string;
  created_at: string;
}