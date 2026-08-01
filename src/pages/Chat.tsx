import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Send,
  Square,
  Bot,
  User,
  HardDrive,
  Sparkles,
  ArrowLeft,
  Activity,
  Cpu,
  Zap,
  Sliders,
  ChevronDown,
  ChevronUp,
  Layers,
} from 'lucide-react';
import { Button, Badge, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import {
  getInferenceStatus,
  sendChatMessage,
  stopChatGeneration,
  unloadActiveModel,
  listenInferenceStatus,
  listenInferenceToken,
  listenInferenceError,
} from '../services/ai.service';
import { routePromptCapability } from '../services/intelligence.service';
import type {
  ChatMessage,
  InferenceStatusPayload,
  LoadedModelInfo,
  StreamChunkPayload,
} from '../types/ai';
import styles from './Chat.module.css';

export const Chat: React.FC = () => {
  const navigate = useNavigate();
  const { addToast } = useToast();

  const [status, setStatus] = useState<string>('NotLoaded');
  const [loadingStep, setLoadingStep] = useState<string | null>(null);
  const [loadedModel, setLoadedModel] = useState<LoadedModelInfo | null>(null);
  const [activeAdapter, setActiveAdapter] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [streamingText, setStreamingText] = useState<string>('');

  // Diagnostic Stats
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);
  const [tokensGeneratedCount, setTokensGeneratedCount] = useState<number>(0);
  const [generationStartTime, setGenerationStartTime] = useState<number | null>(null);
  const [tokensPerSecond, setTokensPerSecond] = useState<number>(0);
  const [promptTokenCount, setPromptTokenCount] = useState<number>(0);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const isGeneratingRef = useRef<boolean>(false);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingText]);

  // Query initial status
  const refreshStatus = useCallback(async () => {
    try {
      const payload: InferenceStatusPayload = await getInferenceStatus();
      setStatus(payload.status);
      setLoadingStep(payload.step || null);
      setLoadedModel(payload.model || null);
      if (payload.model?.activeAdapter) {
        setActiveAdapter(payload.model.activeAdapter);
      }
      if (payload.status === 'Generating') {
        setIsGenerating(true);
        isGeneratingRef.current = true;
      }
    } catch (err) {
      console.error('Failed to get inference status:', err);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Listen to Tauri inference events
  useEffect(() => {
    let unlistenStatusFn: (() => void) | null = null;
    let unlistenTokenFn: (() => void) | null = null;
    let unlistenErrorFn: (() => void) | null = null;

    listenInferenceStatus((payload) => {
      setStatus(payload.status);
      setLoadingStep(payload.step || null);
      if (payload.model) {
        setLoadedModel(payload.model);
        if (payload.model.activeAdapter) {
          setActiveAdapter(payload.model.activeAdapter);
        }
      } else {
        setLoadedModel(null);
        setActiveAdapter(null);
      }
      if (payload.status === 'Generating') {
        setIsGenerating(true);
        isGeneratingRef.current = true;
      } else {
        setIsGenerating(false);
        isGeneratingRef.current = false;
      }
    }).then((unlisten) => { unlistenStatusFn = unlisten; });

    listenInferenceToken((payload: StreamChunkPayload) => {
      if (payload.isFinal) {
        setIsGenerating(false);
        isGeneratingRef.current = false;

        setStreamingText((prevFull) => {
          const finalMessage = prevFull + payload.text;
          if (finalMessage.trim().length > 0) {
            setMessages((prevMsgs) => [
              ...prevMsgs,
              { role: 'assistant', content: finalMessage.trim() },
            ]);
          }
          return '';
        });
      } else {
        setStreamingText((prev) => prev + payload.text);
        setTokensGeneratedCount((prev) => {
          const nextCount = prev + 1;
          setGenerationStartTime((startTime) => {
            if (startTime) {
              const elapsedSec = (Date.now() - startTime) / 1000;
              if (elapsedSec > 0) {
                setTokensPerSecond(Number((nextCount / elapsedSec).toFixed(1)));
              }
            }
            return startTime;
          });
          return nextCount;
        });
      }
    }).then((unlisten) => { unlistenTokenFn = unlisten; });

    listenInferenceError((payload) => {
      addToast('error', `Inference Error: ${payload.error}`);
      setIsGenerating(false);
      isGeneratingRef.current = false;
      setStreamingText('');
    }).then((unlisten) => { unlistenErrorFn = unlisten; });

    return () => {
      if (unlistenStatusFn) unlistenStatusFn();
      if (unlistenTokenFn) unlistenTokenFn();
      if (unlistenErrorFn) unlistenErrorFn();
    };
  }, [addToast]);

  const handleSend = async () => {
    const text = inputText.trim();
    if (!text || isGenerating || status !== 'Ready') return;

    const userMessage: ChatMessage = { role: 'user', content: text };
    const newMessages = [...messages, userMessage];

    // Estimate prompt token count
    const totalChars = newMessages.reduce((acc, m) => acc + m.content.length, 0);
    setPromptTokenCount(Math.ceil(totalChars / 4));
    setTokensGeneratedCount(0);
    setGenerationStartTime(Date.now());
    setTokensPerSecond(0);

    setMessages(newMessages);
    setInputText('');
    setStreamingText('');
    setIsGenerating(true);
    isGeneratingRef.current = true;

    try {
      if (loadedModel) {
        routePromptCapability('huggingface', loadedModel.modelId, text).then((route) => {
          if (route.selectedAdapterName) {
            setActiveAdapter(route.selectedAdapterName);
            addToast('info', `⚡ Routed to ${route.targetCapability.toUpperCase()} LoRA adapter (${route.selectedAdapterName})`);
          } else {
            setActiveAdapter(null);
          }
        }).catch(() => {});
      }
      await sendChatMessage(newMessages);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast('error', `Failed to send message: ${msg}`);
      setIsGenerating(false);
      isGeneratingRef.current = false;
    }
  };

  const handleStop = async () => {
    try {
      await stopChatGeneration();
      addToast('info', 'Generation stopped');
    } catch (err) {
      console.error('Failed to stop generation:', err);
    }
  };

  const handleUnload = async () => {
    try {
      await unloadActiveModel();
      addToast('info', 'Model unloaded');
      setStatus('NotLoaded');
      setLoadedModel(null);
      setActiveAdapter(null);
      setMessages([]);
    } catch (err) {
      addToast('error', `Failed to unload: ${String(err)}`);
    }
  };

  const isModelReady = status === 'Ready' || status === 'Generating';

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInfo}>
          <Button variant="ghost" size="sm" onClick={() => navigate('/models')}>
            <ArrowLeft size={16} style={{ marginRight: 4 }} />
            Models & Storage
          </Button>

          <div className={styles.modelBadge}>
            <div
              className={`${styles.statusDot} ${
                status === 'Generating'
                  ? styles.statusDotGenerating
                  : isModelReady
                  ? styles.statusDotReady
                  : ''
              }`}
            />
            {loadedModel ? (
              <div className={styles.modelTitleGroup}>
                <span className={styles.modelTitleName}>{loadedModel.modelName}</span>
                <span className={styles.modelPill}>{loadedModel.quantization}</span>
                <span className={styles.modelPillFamily}>{loadedModel.modelFamily || 'Generic'}</span>
                {activeAdapter && (
                  <span className={styles.modelPillAdapter}>⚡ LoRA: {activeAdapter}</span>
                )}
              </div>
            ) : (
              <span className={styles.noModelText}>No Model Loaded</span>
            )}
          </div>

          {loadedModel && (
            <div className={styles.modelDetails}>
              Context: {loadedModel.contextLength.toLocaleString()} tokens · {loadedModel.backendUsed}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <Button
            variant={showDiagnostics ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            title="Toggle Developer Runtime Diagnostics"
          >
            <Activity size={15} style={{ marginRight: 4 }} />
            Diagnostics {showDiagnostics ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </Button>

          {isModelReady && (
            <Button variant="secondary" size="sm" onClick={handleUnload} disabled={isGenerating}>
              Unload
            </Button>
          )}
          <Badge variant={isModelReady ? 'success' : 'default'}>
            {status === 'Generating' ? 'Generating...' : status === 'Ready' ? '● Ready' : status}
          </Badge>
        </div>
      </header>

      {/* Developer Runtime Diagnostics Collapsible Panel */}
      {showDiagnostics && (
        <div className={styles.diagnosticsPanel}>
          <div className={styles.diagGrid}>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Cpu size={13} /> Active Model</div>
              <div className={styles.diagValue}>{loadedModel ? loadedModel.modelName : 'None'}</div>
            </div>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Layers size={13} /> Family / Template</div>
              <div className={styles.diagValue}>{loadedModel ? `${loadedModel.modelFamily || 'Generic'} (${loadedModel.chatTemplate || 'default'})` : '—'}</div>
            </div>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Zap size={13} /> Active Adapter</div>
              <div className={styles.diagValue}>{activeAdapter ? activeAdapter : 'None (Base Model)'}</div>
            </div>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Cpu size={13} /> Hardware Backend</div>
              <div className={styles.diagValue}>{loadedModel ? loadedModel.backendUsed : '—'}</div>
            </div>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Sliders size={13} /> Sampling Parameters</div>
              <div className={styles.diagValue}>Temp: 0.7 · Top-P: 0.9 · Repeat: 1.1</div>
            </div>
            <div className={styles.diagCard}>
              <div className={styles.diagLabel}><Activity size={13} /> Real-time Throughput</div>
              <div className={styles.diagValue}>
                {isGenerating || tokensGeneratedCount > 0
                  ? `Prompt: ~${promptTokenCount} t | Out: ${tokensGeneratedCount} t | ${tokensPerSecond} t/s`
                  : 'Idle'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <main className={styles.chatArea}>
        {status === 'Loading' ? (
          <div className={styles.emptyState}>
            <Spinner size="lg" />
            <h3 style={{ margin: 0 }}>Loading Local LLM into Memory...</h3>
            <p style={{ fontSize: '13px', color: 'var(--accent)' }}>
              {loadingStep || 'Initializing llama.cpp runtime...'}
            </p>
          </div>
        ) : !isModelReady ? (
          <div className={styles.emptyState}>
            <Bot size={48} color="var(--accent)" />
            <h3>No Local Model Loaded</h3>
            <p style={{ maxWidth: '440px' }}>
              Load a downloaded model from the Models & Storage page to start generating local LLM responses with zero cloud APIs.
            </p>
            <Button variant="primary" onClick={() => navigate('/models')}>
              <HardDrive size={16} style={{ marginRight: 6 }} />
              Go to Models & Storage
            </Button>
          </div>
        ) : messages.length === 0 && !streamingText ? (
          <div className={styles.emptyState}>
            <Sparkles size={40} color="var(--accent)" />
            <h3>Local LLM Ready</h3>
            <p style={{ maxWidth: '400px' }}>
              Model {loadedModel?.modelName} is loaded in local RAM/VRAM. Type a prompt below to test real-time token streaming.
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`${styles.messageRow} ${
                  msg.role === 'user' ? styles.messageRowUser : styles.messageRowAssistant
                }`}
              >
                <div
                  className={`${styles.messageAvatar} ${
                    msg.role === 'user' ? styles.messageAvatarUser : ''
                  }`}
                >
                  {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                </div>
                <div
                  className={`${styles.messageBubble} ${
                    msg.role === 'user' ? styles.messageBubbleUser : styles.messageBubbleAssistant
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {/* Streaming Active Bubble */}
            {isGenerating && (
              <div className={`${styles.messageRow} ${styles.messageRowAssistant}`}>
                <div className={styles.messageAvatar}>
                  <Bot size={18} />
                </div>
                <div className={`${styles.messageBubble} ${styles.messageBubbleAssistant}`}>
                  {streamingText}
                  <span className={styles.streamingCursor} />
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </>
        )}
      </main>

      {/* Input Bar */}
      <footer className={styles.inputBar}>
        <div className={styles.inputContainer}>
          <input
            type="text"
            className={styles.inputField}
            placeholder={
              !isModelReady
                ? 'Load a model first from Models & Storage...'
                : isGenerating
                ? 'Generating local tokens...'
                : 'Type a prompt for the local model (e.g. Write a Python function)...'
            }
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={!isModelReady || isGenerating}
          />

          {isGenerating ? (
            <Button variant="danger" onClick={handleStop} className={styles.sendButton}>
              <Square size={16} style={{ marginRight: 6 }} />
              Stop
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={handleSend}
              disabled={!isModelReady || !inputText.trim()}
              className={styles.sendButton}
            >
              <Send size={16} style={{ marginRight: 6 }} />
              Send
            </Button>
          )}
        </div>
      </footer>
    </div>
  );
};