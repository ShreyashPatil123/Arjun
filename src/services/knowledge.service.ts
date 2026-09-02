import { getBackendService } from './api';
import type { Classification } from './registry.service';

/** How a passage was found. Mirrors `knowledge::index::Retrieval`. */
export type Retrieval = 'keyword' | 'vector';

/** One document the index holds. Metadata only — the text is reached by searching. */
export interface IndexedDocument {
  documentSha256: string;
  documentName: string;
  classification: Classification;
  /** How many passages this document was split into. */
  chunks: number;
  /** The highest page number held for it. */
  pages: number;
}

/** One retrieved passage, with everything needed to cite it. */
export interface SearchResult {
  chunkId: string;
  documentSha256: string;
  documentName: string;
  text: string;
  page: number;
  sectionPath: string[];
  classification: Classification;
  /** Lower is a better match. */
  score: number;
  retrieval: Retrieval;
}

/** The index's own state. */
export interface KnowledgeHealth {
  /** Distinct, non-superseded documents in the index. */
  documents: number;
  /** How many of those this person is cleared to see. */
  visibleDocuments: number;
  /** Passages across the documents this person can see. */
  visiblePassages: number;
  /**
   * False when the index could not be opened or counted. Distinct from an
   * index that opened and holds nothing: an empty library and an unreadable
   * one need opposite actions from whoever is looking at the screen.
   */
  readable: boolean;
}

export const knowledgeService = {
  /** Every document the signed-in person may retrieve from. */
  documents(): Promise<IndexedDocument[]> {
    return getBackendService().invoke<IndexedDocument[]>('knowledge_documents');
  },

  /**
   * Searches as the signed-in person.
   *
   * The same call the agent's `knowledge.search_authorized` tool makes, so what
   * this returns is what a run would actually be given rather than an
   * approximation of it. Clearance is applied inside the query, so a passage
   * somebody may not see is never fetched, ranked or counted.
   */
  search(query: string, limit?: number): Promise<SearchResult[]> {
    return getBackendService().invoke<SearchResult[]>('knowledge_search', { query, limit });
  },

  /** Document and passage counts, and whether the index answered at all. */
  health(): Promise<KnowledgeHealth> {
    return getBackendService().invoke<KnowledgeHealth>('knowledge_health');
  },
};
