export function formatBytes(bytes?: number | null, decimals = 2): string {
  if (bytes === undefined || bytes === null || isNaN(bytes) || bytes <= 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  if (i < 0 || i >= sizes.length) return '0 Bytes';
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function formatFrequency(mhz?: number | null): string {
  if (mhz === undefined || mhz === null || isNaN(mhz) || mhz <= 0) return 'N/A';
  if (mhz >= 1000) {
    return `${(mhz / 1000).toFixed(2)} GHz`;
  }
  return `${mhz} MHz`;
}

export function formatPercentage(used?: number | null, total?: number | null): number {
  if (used === undefined || used === null || total === undefined || total === null || total <= 0) return 0;
  const pct = Math.round((used / total) * 100);
  return Math.min(100, Math.max(0, pct));
}

export function formatNumber(num?: number | null): string {
  if (num === undefined || num === null || isNaN(num)) return 'N/A';
  return new Intl.NumberFormat().format(num);
}

export function formatDate(date?: string | Date | null): string {
  if (!date) return 'N/A';
  try {
    return new Intl.DateTimeFormat('default', {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(typeof date === 'string' ? new Date(date) : date);
  } catch {
    return 'N/A';
  }
}

export function debounce<T extends (...args: any[]) => void>(fn: T, ms: number) {
  let timer: ReturnType<typeof setTimeout>;
  return function(this: any, ...args: Parameters<T>) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

export function classNames(...args: any[]) {
  return args.filter((x): x is string => typeof x === 'string' && x.length > 0).join(' ');
}

export function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}