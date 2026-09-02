import React, { Suspense, useMemo } from 'react';
import styles from './ChatSurface.module.css';

/**
 * Loaded on demand, not with the application.
 *
 * three.js and react-three-fiber are ~900 KB of the built bundle — nearly
 * three times everything else put together — for one 36px avatar. Imported
 * statically they were paid for by every screen, including the ones with no
 * chat on them. Split out, they are fetched the first time an orb is actually
 * drawn: from disk, by a desktop application, so the cost is a frame, and
 * until then the CSS orb is already on screen.
 */
const GradientOrb = React.lazy(() => import('../ui/GradientOrb'));

interface ChatOrbProps {
  /** True while a run is streaming — the orb spins faster and brightens. */
  active: boolean;
  size?: number;
}

/**
 * The assistant's avatar orb.
 *
 * A GPU shader orb (`GradientOrb`): noise-mixed blue/purple/orange with a
 * breathing pulse and constant rotation. It streams faster while a run is in
 * flight, which is the only signal the orb carries.
 *
 * ## Why there is still a CSS orb underneath
 *
 * The shader needs WebGL, and this application is aimed at machines whose GPU
 * is already holding a language model — including ones reached over remote
 * desktop, where WebGL is often absent entirely. When the context cannot be
 * created the layered-conic-gradient orb renders instead, which is what shipped
 * before and needs nothing but the compositor. A missing avatar would be a
 * strange way to fail a chat.
 *
 * Only the newest assistant cell draws an orb (`orbMessageId` in
 * `ChatSurface`), so there is one WebGL context on screen, not one per message.
 */

/** Whether this machine can give us a WebGL context at all. Asked once. */
let webglSupport: boolean | null = null;

function hasWebgl(): boolean {
  if (webglSupport !== null) return webglSupport;
  try {
    const canvas = document.createElement('canvas');
    webglSupport = Boolean(
      canvas.getContext('webgl2') ?? canvas.getContext('webgl'),
    );
  } catch {
    webglSupport = false;
  }
  return webglSupport;
}

export function ChatOrb({ active, size = 36 }: ChatOrbProps) {
  const shader = useMemo(hasWebgl, []);

  const cssOrb = (
    <span
      className={styles.chatOrb}
      data-active={active || undefined}
      style={{ '--orb-size': `${size}px` } as React.CSSProperties}
      aria-hidden="true"
    />
  );

  if (!shader) return cssOrb;

  return (
    <span
      className={styles.chatOrbShader}
      data-active={active || undefined}
      style={{ '--orb-size': `${size}px` } as React.CSSProperties}
      aria-hidden="true"
    >
      {/* The CSS orb is the fallback in both senses: it holds the space while
        * the shader chunk loads, and it is what stays if WebGL is missing. */}
      <Suspense fallback={cssOrb}>
        <GradientOrb
          config={{
            background: 'transparent',
            // Streaming turns faster. The numbers are the only difference
            // between the two states, so "it is working" reads as motion
            // rather than as a second element appearing.
            rotationSpeed: active ? 0.85 : 0.28,
            noiseScale: 0.65,
            innerRadius: 0.1,
          }}
        />
      </Suspense>
    </span>
  );
}
