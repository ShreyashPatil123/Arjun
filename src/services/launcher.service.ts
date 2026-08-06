// Launch section IPC.
//
// Mirrors src-tauri/src/commands/launcher.rs. Errors from those commands are
// written for people, not logs, so they are surfaced verbatim rather than
// replaced with a generic message.

import { invoke } from '@tauri-apps/api/core';
import { listen, type Event, type UnlistenFn } from '@tauri-apps/api/event';

/** Which gateway endpoint a tool speaks. */
export type ToolProtocol = 'openai' | 'anthropic';

/**
 * A tool's current state. The `state` field discriminates the union — it comes
 * from a Rust enum, flattened into the same object as the tool's details.
 */
export type ToolState =
  | { state: 'notInstalled' }
  | { state: 'ready'; version: string }
  | { state: 'running'; pid: number }
  | { state: 'problem'; reason: string };

export type ToolStatus = {
  id: string;
  name: string;
  description: string;
  protocol: ToolProtocol;
  /** True for tools the user added themselves — shown as unverified. */
  userDefined: boolean;
  /** Present only when an install could actually run. */
  installCommand?: string | null;
} & ToolState;

export interface ClientActivity {
  label: string;
  protocol: string;
  requests: number;
  lastSeenMs: number;
}

export interface GatewayStats {
  running: boolean;
  port: number;
  totalRequests: number;
  activeRequests: number;
  queueDepth: number;
  clients: ClientActivity[];
}

export interface LaunchOverview {
  tools: ToolStatus[];
  gateway: GatewayStats;
  /** Problems loading custom tool definitions. */
  warnings: string[];
  /** False when no model is loaded, so Launch would produce a dead tool. */
  canLaunch: boolean;
  blockedReason?: string | null;
}

/** Tools and live server status in one call, so the two cannot disagree. */
export function getLaunchOverview(): Promise<LaunchOverview> {
  return invoke<LaunchOverview>('get_launch_overview');
}

/**
 * The exact command an install would run. Nothing is executed — this exists so
 * the user can see and approve it first.
 */
export function previewToolInstall(toolId: string): Promise<string> {
  return invoke<string>('preview_tool_install', { toolId });
}

export function installTool(toolId: string): Promise<void> {
  return invoke<void>('install_tool', { toolId });
}

export interface ToolInstallProgress {
  toolId: string;
  line: string;
}

/**
 * Each line the package manager prints, as it prints it.
 *
 * An install pulls megabytes and takes a minute; without this the button sits
 * on "Installing…" long enough to look hung.
 */
export function listenToolInstallProgress(
  callback: (progress: ToolInstallProgress) => void
): Promise<UnlistenFn> {
  return listen<ToolInstallProgress>('tool:install-progress', (event: Event<ToolInstallProgress>) => {
    callback(event.payload);
  });
}

/** Starts the tool already connected. Returns its process id. */
export function launchTool(toolId: string): Promise<number> {
  return invoke<number>('launch_tool', { toolId });
}

/**
 * Stops tracking a tool without killing it — it may be a terminal the user is
 * working in. The card returns to Ready.
 */
export function forgetToolProcess(toolId: string): Promise<void> {
  return invoke<void>('forget_tool_process', { toolId });
}

/** Where custom tool definitions live, so the UI can show the path. */
export function userToolsFile(): Promise<string> {
  return invoke<string>('user_tools_file');
}
