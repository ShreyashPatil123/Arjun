import React from 'react';
import { mathToPlain, parseTex, type MathNode } from './texMath';
import styles from './ChatSurface.module.css';

/**
 * Math a model wrote in LaTeX, set as math: `\frac{bar}{psi}` as a fraction,
 * `x^2` raised, `\times` as ×. See `texMath.ts` for what is read and why.
 *
 * Everything is rendered as React text; nothing from the model becomes markup.
 * Screen readers get the one-line reading (`150 psi × 0.06894757 bar/psi`).
 */
export function MathView({ tex, display }: { tex: string; display: boolean }) {
  const nodes = React.useMemo(() => parseTex(tex), [tex]);
  const label = React.useMemo(() => mathToPlain(nodes), [nodes]);
  const Tag = display ? 'div' : 'span';
  return (
    <Tag className={display ? styles.mdMathBlock : styles.mdMath} role="math" aria-label={label}>
      <span aria-hidden="true">{renderNodes(nodes, 'm')}</span>
    </Tag>
  );
}

function renderNodes(nodes: MathNode[], key: string): React.ReactNode[] {
  return nodes.map((node, i) => {
    const k = `${key}-${i}`;
    switch (node.kind) {
      case 'text':
        return <React.Fragment key={k}>{node.text}</React.Fragment>;
      case 'sup':
        return <sup key={k}>{renderNodes(node.body, k)}</sup>;
      case 'sub':
        return <sub key={k}>{renderNodes(node.body, k)}</sub>;
      case 'frac':
        return (
          <span key={k} className={styles.mdFrac}>
            <span className={styles.mdFracNum}>{renderNodes(node.num, `${k}-n`)}</span>
            <span className={styles.mdFracDen}>{renderNodes(node.den, `${k}-d`)}</span>
          </span>
        );
      case 'sqrt':
        return (
          <span key={k} className={styles.mdSqrt}>
            √<span className={styles.mdSqrtBody}>{renderNodes(node.body, k)}</span>
          </span>
        );
    }
    return null;
  });
}
