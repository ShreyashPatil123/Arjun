export const logger = {
  info: (msg: string, ...args: any[]) => console.info(`[INFO]`, msg, ...args),
  warn: (msg: string, ...args: any[]) => console.warn(`[WARN]`, msg, ...args),
  error: (msg: string, ...args: any[]) => console.error(`[ERROR]`, msg, ...args),
  debug: (msg: string, ...args: any[]) => console.debug(`[DEBUG]`, msg, ...args),
};