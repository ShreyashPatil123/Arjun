export async function getProviders() { return ['huggingface', 'ollama_library', 'local']; }
export async function listProviders() { return getProviders(); }
export async function searchProviderModels(providerId: string, query: string) { return []; }