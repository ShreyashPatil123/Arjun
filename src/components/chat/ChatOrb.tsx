import React, { useEffect, useRef } from 'react';
import styles from './ChatSurface.module.css';

interface ChatOrbProps {
  active: boolean;
  size?: number;
}

export function ChatOrb({ active, size = 36 }: ChatOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf: number;
    let t = 0;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width = size * dpr;
      const h = canvas.height = size * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, size, size);

      const cx = size / 2;
      const cy = size / 2;
      const radius = size * 0.38;

      // Background glow
      const glow = ctx.createRadialGradient(cx, cy, radius * 0.4, cx, cy, radius * 1.4);
      glow.addColorStop(0, 'rgba(139, 92, 246, 0.25)');   // purple center
      glow.addColorStop(0.6, 'rgba(56, 189, 248, 0.15)'); // cyan mid
      glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // Rotating ring
      t += active ? 0.04 : 0.01;
      const segments = 3;
      for (let i = 0; i < segments; i++) {
        const angle = t + (i * Math.PI * 2) / segments;
        const x1 = cx + Math.cos(angle) * radius * 0.85;
        const y1 = cy + Math.sin(angle) * radius * 0.85;
        const x2 = cx + Math.cos(angle + 0.4) * radius;
        const y2 = cy + Math.sin(angle + 0.4) * radius;

        const grad = ctx.createLinearGradient(x1, y1, x2, y2);
        grad.addColorStop(0, '#c084fc'); // purple-400
        grad.addColorStop(1, '#38bdf8'); // sky-400

        ctx.beginPath();
        ctx.arc(cx, cy, radius, angle, angle + 0.5);
        ctx.strokeStyle = grad;
        ctx.lineWidth = active ? 2.5 : 2;
        ctx.lineCap = 'round';
        ctx.shadowColor = '#a855f7';
        ctx.shadowBlur = active ? 12 : 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Inner soft core
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 0.5);
      core.addColorStop(0, 'rgba(168, 85, 247, 0.35)');
      core.addColorStop(1, 'rgba(168, 85, 247, 0)');
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
      ctx.fill();

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, [active, size]);

  return (
    <canvas
      ref={canvasRef}
      className={styles.chatOrb}
      style={{ width: size, height: size }}
      aria-hidden="true"
    />
  );
}