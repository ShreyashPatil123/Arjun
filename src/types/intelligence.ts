export interface InferenceParameters {
  temperature: number;
  topP: number;
  topK: number;
  repeatPenalty: number;
  maxTokens: number;
  contextLength: number;
  threads: number;
  gpuLayers: number;
}

export interface CapabilityItem {
  capability: string;
  supported: boolean;
  confidence: number;
  description: string;
}

export interface ModelProfile {
  profileVersion: number;
  packageId: string;
  modelId: string;
  modelName: string;
  modelFamily: string;
  architecture: string;
  chatTemplate: string;
  systemPromptFormat: string;
  tokens: {
    bosToken?: string;
    eosToken?: string;
    stopTokens: string[];
    padToken?: string;
  };
  recommendedParams: InferenceParameters;
  activeUserParams?: InferenceParameters;
  capabilityRegistry: {
    capabilities: Record<string, CapabilityItem>;
  };
  provenance: {
    ggufMetadataExtracted: boolean;
    generationConfigExtracted: boolean;
    tokenizerConfigExtracted: boolean;
    configExtracted: boolean;
    modelCardExtracted: boolean;
    sourceSummary: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface AdapterRouteResult {
  intent: string;
  targetCapability: string;
  selectedAdapterName?: string;
  adapterFilePath?: string;
  isAutoRouted: boolean;
  reasoning: string;
}
