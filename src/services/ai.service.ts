export async function getAIStatus() { return 'stopped'; }
export async function loadModel(modelPath: string) { return; }
export async function unloadModel() { return; }
export async function chat(messages?: unknown[]) { return { message: '' }; }