// Model browsing IPC.
//
// Mirrors src-tauri/src/commands/catalog.rs.

import { invoke } from '@tauri-apps/api/core';
import type { ModelCard, ModelCategory } from '../types/ai';

export interface CategoryCount {
  category: ModelCategory;
  label: string;
  count: number;
}

export interface CatalogPage {
  cards: ModelCard[];
  /** Only categories that match at least one result, so no filter is a dead end. */
  categories: CategoryCount[];
  /** Memory available for weights; 0 when hardware could not be read. */
  weightBudgetBytes: number;
  /** Explains a partial result, e.g. rate limiting or browsing without a token. */
  notice?: string | null;
}

/**
 * Browses the catalog, optionally filtered by a search term.
 *
 * A search queries HuggingFace directly rather than filtering what is cached —
 * the cache holds only the popular sweep, so filtering it would report that a
 * specific fine-tune does not exist.
 */
export function browseModelCards(query?: string): Promise<CatalogPage> {
  return invoke<CatalogPage>('browse_model_cards', { query: query ?? null });
}

/** Every known category, for a sidebar that stays stable while results load. */
export function listModelCategories(): Promise<CategoryCount[]> {
  return invoke<CategoryCount[]>('list_model_categories');
}

/** Bytes as GB with one decimal, e.g. `4.7 GB`. */
export function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}
