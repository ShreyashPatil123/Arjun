/**
 * Sarathi Unified Memory Engine Frontend API Service
 * Exposes reusable functions for Memory Dashboard, User Profile, Projects, Search, and Status.
 */

import { invoke } from '@tauri-apps/api/core';

export interface UserProfileRecord {
  key: string;
  value: string;
  category: string;
  confidence: number;
  updatedAt: string;
}

export interface ProjectRecord {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ScoredCandidate {
  id: string;
  content: string;
  memoryType: string;
  projectId?: string;
  importanceScore: number;
  recencyTimestamp: number;
  similarity: number;
  recencyScore?: number;
  finalScore?: number;
}

export interface ProviderHealthStatus {
  status: string;
  registeredProviders: string[];
  capabilities: Record<string, any>;
}

export const memoryService = {
  /**
   * Get Memory Engine sidecar & provider health status
   */
  async getHealthStatus(): Promise<ProviderHealthStatus> {
    return await invoke<ProviderHealthStatus>('get_memory_health_status');
  },

  /**
   * Get all User Profile Memory facts
   */
  async getUserProfile(): Promise<UserProfileRecord[]> {
    return await invoke<UserProfileRecord[]>('get_user_profile_memory');
  },

  /**
   * Add or update a user profile fact manually
   */
  async updateUserProfileFact(key: string, value: string, category: string = 'general'): Promise<void> {
    await invoke('update_user_profile_fact', { key, value, category });
  },

  /**
   * List all projects
   */
  async listProjects(): Promise<ProjectRecord[]> {
    return await invoke<ProjectRecord[]>('list_memory_projects');
  },

  /**
   * Create a new isolated memory project
   */
  async createProject(name: string, description?: string): Promise<ProjectRecord> {
    return await invoke<ProjectRecord>('create_memory_project', { name, description });
  },

  /**
   * Switch active workspace project
   */
  async switchActiveProject(projectId: string): Promise<string> {
    return await invoke<string>('switch_active_project', { projectId });
  },

  /**
   * Get currently active workspace project ID
   */
  async getActiveProject(): Promise<string> {
    return await invoke<string>('get_active_project');
  },

  /**
   * Search memory nodes by query and optional project filter
   */
  async searchMemories(query: string, projectId?: string): Promise<ScoredCandidate[]> {
    return await invoke<ScoredCandidate[]>('search_memory_nodes', { query, projectId });
  },

  /**
   * Delete a memory node by ID
   */
  async deleteMemoryNode(id: string): Promise<void> {
    await invoke('delete_memory_node_by_id', { id });
  },
};
