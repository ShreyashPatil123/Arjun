import React from 'react';
import styles from './ChatSurface.module.css';

/**
 * Minimal Markdown renderer for the chat surface.
 *
 * Why hand-rolled:
 * - The Arjun project has an egress gate that forbids adding a runtime
 *   markdown library at this layer (the only chokepoint is the broker).
 * - A model response typically only needs: paragraphs, inline code,
 *   fenced code blocks, bold, italic, unordered lists, headings, and
 *   links. That's a small, well-bounded set we can render correctly
 *   without pulling in a parser.
 *
 * Streaming-safe: rendering is a pure function of `content`, so each
 * new chunk produces a deterministic, valid React tree.
 */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Apply inline markdown transforms to a single line of text.
 * The order of operations matters: code spans first (to protect their
 * contents from later transforms), then bold, then italic, then links.
 */
function renderInline(line: string, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let i = 0;
  let key = 0;
  let buf = '';

  const flushBuf = () => {
    if (buf) {
      out.push(<span key={`${keyPrefix}-t-${key++}`}>{buf}</span>);
      buf = '';
    }
  };

  while (i < line.length) {
    // Inline code: `...`
    if (line[i] === '`') {
      const end = line.indexOf('`', i + 1);
      if (end !== -1) {
        flushBuf();
        out.push(
          <code key={`${keyPrefix}-c-${key++}`} className={styles.mdCode}>
            {line.slice(i + 1, end)}
          </code>,
        );
        i = end + 1;
        continue;
      }
    }
    // Bold: **...**
    if (line[i] === '*' && line[i + 1] === '*') {
      const end = line.indexOf('**', i + 2);
      if (end !== -1) {
        flushBuf();
        out.push(
          <strong key={`${keyPrefix}-b-${key++}`}>{line.slice(i + 2, end)}</strong>,
        );
        i = end + 2;
        continue;
      }
    }
    // Italic: *...* (single asterisk, but not double)
    if (line[i] === '*' && line[i + 1] !== '*' && (i === 0 || line[i - 1] !== '*')) {
      const end = findUnescaped(line, '*', i + 1);
      if (end !== -1 && end > i + 1) {
        flushBuf();
        out.push(
          <em key={`${keyPrefix}-i-${key++}`}>{line.slice(i + 1, end)}</em>,
        );
        i = end + 1;
        continue;
      }
    }
    // Link: [text](url)
    if (line[i] === '[') {
      const labelEnd = line.indexOf(']', i + 1);
      if (labelEnd !== -1 && line[labelEnd + 1] === '(') {
        const urlEnd = line.indexOf(')', labelEnd + 2);
        if (urlEnd !== -1) {
          const label = line.slice(i + 1, labelEnd);
          const url = line.slice(labelEnd + 2, urlEnd);
          flushBuf();
          out.push(
            <a
              key={`${keyPrefix}-a-${key++}`}
              href={url}
              target="_blank"
              rel="noreferrer"
              className={styles.mdLink}
            >
              {label}
            </a>,
          );
          i = urlEnd + 1;
          continue;
        }
      }
    }
    buf += line[i];
    i += 1;
  }
  flushBuf();
  return out;
}

function findUnescaped(s: string, ch: string, start: number): number {
  for (let i = start; i < s.length; i += 1) {
    if (s[i] === ch) return i;
  }
  return -1;
}

interface Block {
  kind: 'p' | 'h' | 'ul' | 'ol' | 'code' | 'blockquote';
  level?: number;
  lang?: string;
  text: string;
}

/**
 * Tokenize the input into a sequence of blocks. The tokenizer is
 * intentionally minimal — it recognises the most common shapes a model
 * emits in chat and folds everything else into a paragraph.
 */
function tokenize(input: string): Block[] {
  const lines = input.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block: ```lang\n...\n```
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const lang = fence[1] || undefined;
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1; // skip closing fence
      blocks.push({ kind: 'code', lang, text: body.join('\n') });
      continue;
    }

    // Heading: # ... ######
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      blocks.push({ kind: 'h', level: heading[1].length, text: heading[2] });
      i += 1;
      continue;
    }

    // Unordered list
    if (/^[\s]*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[\s]*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*[-*]\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'ul', text: items.join('\n') });
      continue;
    }

    // Ordered list
    if (/^[\s]*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[\s]*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*\d+\.\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'ol', text: items.join('\n') });
      continue;
    }

    // Blockquote: > ...
    if (/^>\s+/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^>\s+/.test(lines[i])) {
        body.push(lines[i].replace(/^>\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'blockquote', text: body.join('\n') });
      continue;
    }

    // Blank line: skip
    if (line.trim() === '') {
      i += 1;
      continue;
    }

    // Paragraph: collect consecutive non-blank, non-special lines.
    const para: string[] = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^```/.test(lines[i]) &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^[\s]*[-*]\s+/.test(lines[i]) &&
      !/^[\s]*\d+\.\s+/.test(lines[i]) &&
      !/^>\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ kind: 'p', text: para.join('\n') });
  }

  return blocks;
}

export function Markdown({ content }: { content: string }) {
  const blocks = React.useMemo(() => tokenize(content), [content]);

  return (
    <div className={styles.markdown}>
      {blocks.map((block, idx) => {
        const key = `b-${idx}`;
        switch (block.kind) {
          case 'h': {
            const level = Math.min(Math.max(block.level ?? 1, 1), 6);
            const Tag = `h${level}` as keyof React.JSX.IntrinsicElements;
            return (
              <Tag key={key} className={styles.mdHeading} data-level={level}>
                {renderInline(block.text, `${key}-i`)}
              </Tag>
            );
          }
          case 'p':
            return (
              <p key={key} className={styles.mdParagraph}>
                {renderInline(block.text, `${key}-i`)}
              </p>
            );
          case 'code':
            return (
              <pre key={key} className={styles.mdCodeBlock} data-lang={block.lang}>
                <code>{block.text}</code>
              </pre>
            );
          case 'ul':
            return (
              <ul key={key} className={styles.mdList}>
                {block.text.split('\n').map((item, i) => (
                  <li key={`${key}-li-${i}`}>{renderInline(item, `${key}-i-${i}`)}</li>
                ))}
              </ul>
            );
          case 'ol':
            return (
              <ol key={key} className={styles.mdList}>
                {block.text.split('\n').map((item, i) => (
                  <li key={`${key}-li-${i}`}>{renderInline(item, `${key}-i-${i}`)}</li>
                ))}
              </ol>
            );
          case 'blockquote':
            return (
              <blockquote key={key} className={styles.mdBlockquote}>
                {renderInline(block.text, `${key}-i`)}
              </blockquote>
            );
        }
        return null;
      })}
    </div>
  );
}

// Suppress unused warning for escapeHtml (kept available for future raw-html
// support without re-importing it).
void escapeHtml;