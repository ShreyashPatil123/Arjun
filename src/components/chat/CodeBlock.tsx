import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

/**
 * A fenced code block: language label, copy button, coloured source.
 *
 * ## Why the highlighter is written here rather than installed
 *
 * A real highlighter ships a grammar per language and, with it, a few hundred
 * kilobytes. This application is built and verified offline and its bundle is
 * already at the size Vite warns about, so the trade was made the other way:
 * one lexer over the shapes every C-family and script language shares —
 * comments, strings, numbers, keywords, and a name followed by `(`.
 *
 * That is a real limitation and worth stating plainly. It colours Rust,
 * TypeScript, Python, Go, JSON and SQL usefully; it does not know any
 * language's actual grammar, so a keyword used as an identifier is still
 * coloured as a keyword. It never *changes* the text — every character of the
 * source is emitted exactly once, in order, which is the only property that
 * matters for code somebody is about to copy.
 */

/** Palette from the reference design (VS Code dark). */
const COLOR = {
  keyword: '#569CD6',
  string: '#CE9178',
  comment: '#6A9955',
  fn: '#DCDCAA',
  number: '#B5CEA8',
} as const;

/**
 * Keywords shared across the languages this application actually shows.
 *
 * One set rather than one per language: the block's `lang` is whatever the
 * model wrote after the fence, which is frequently absent or wrong, and a
 * lookup that missed would leave the block uncoloured for no gain.
 */
const KEYWORDS = new Set(
  (
    'abstract as async await break case catch class const constructor continue crate ' +
    'declare default defer del delete do elif else enum except export extends false fi ' +
    'final finally fn for from func function go if impl implements import in instanceof ' +
    'interface let loop match mod move mut namespace new nil none not null or pass priv ' +
    'private pub public raise readonly ref return select self static struct super switch ' +
    'this throw trait true try type typeof union unsafe use var void where while with yield ' +
    'and bool boolean char def double float i8 i16 i32 i64 int int8 int16 int32 int64 ' +
    'lambda long str string u8 u16 u32 u64 usize isize f32 f64 vec option result ' +
    'insert update group order limit join on values set'
  ).split(' '),
);

interface Piece {
  text: string;
  colour?: string;
}

/**
 * Splits source into coloured pieces.
 *
 * Single pass, longest-match-first, and every branch advances the cursor — so
 * the concatenation of the output is the input, and the loop terminates on any
 * string at all, including a truncated one from a stream still arriving.
 */
export function highlight(source: string): Piece[] {
  const out: Piece[] = [];
  let plain = '';
  const flush = () => {
    if (plain) {
      out.push({ text: plain });
      plain = '';
    }
  };
  const push = (text: string, colour: string) => {
    flush();
    out.push({ text, colour });
  };

  let i = 0;
  while (i < source.length) {
    const rest = source.slice(i);

    // Comments: // ... , # ... , -- ... , and /* ... */
    const line = rest.match(/^(\/\/|#|--)[^\n]*/);
    if (line) {
      push(line[0], COLOR.comment);
      i += line[0].length;
      continue;
    }
    if (rest.startsWith('/*')) {
      const end = source.indexOf('*/', i + 2);
      // An unterminated block comment runs to the end. That is what a
      // compiler would do with it, and a stream mid-flight produces it often.
      const stop = end === -1 ? source.length : end + 2;
      push(source.slice(i, stop), COLOR.comment);
      i = stop;
      continue;
    }

    // Strings, with escapes. An unterminated one stops at the newline so a
    // single stray quote cannot colour the rest of the file.
    const quote = rest[0];
    if (quote === '"' || quote === "'" || quote === '`') {
      let j = 1;
      while (j < rest.length) {
        if (rest[j] === '\\') {
          j += 2;
          continue;
        }
        if (rest[j] === quote) {
          j += 1;
          break;
        }
        if (rest[j] === '\n' && quote !== '`') break;
        j += 1;
      }
      push(rest.slice(0, j), COLOR.string);
      i += j;
      continue;
    }

    // Numbers, including hex and floats.
    const num = rest.match(/^(0[xX][0-9a-fA-F_]+|\d[\d_]*\.?[\d_]*([eE][+-]?\d+)?)/);
    if (num) {
      push(num[0], COLOR.number);
      i += num[0].length;
      continue;
    }

    // Words: a keyword, a call, or an ordinary identifier.
    const word = rest.match(/^[A-Za-z_$][\w$]*/);
    if (word) {
      const w = word[0];
      if (KEYWORDS.has(w.toLowerCase())) {
        push(w, COLOR.keyword);
      } else if (/^\s*\(/.test(rest.slice(w.length))) {
        push(w, COLOR.fn);
      } else {
        plain += w;
      }
      i += w.length;
      continue;
    }

    plain += source[i];
    i += 1;
  }
  flush();
  return out;
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false);
  const pieces = React.useMemo(() => highlight(code), [code]);

  const copy = () => {
    void navigator.clipboard
      .writeText(code)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      })
      // A clipboard the webview refuses is not worth an error banner: the
      // code is still on screen and still selectable.
      .catch(() => setCopied(false));
  };

  return (
    <figure className="my-4 overflow-hidden rounded-xl border border-line bg-surface-sunken">
      <figcaption
        className="flex items-center gap-2 border-b border-line px-3 py-[6px]
                   text-[11px] text-ink-faint"
      >
        <span className="font-mono uppercase tracking-wider">{lang || 'text'}</span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={copy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          className="flex items-center gap-1 rounded px-[6px] py-[2px] text-ink-faint
                     transition-colors hover:bg-surface-hover hover:text-ink
                     focus-visible:outline focus-visible:outline-1
                     focus-visible:outline-line"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </figcaption>

      <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-relaxed">
        <code>
          {pieces.map((piece, i) =>
            piece.colour ? (
              <span key={i} style={{ color: piece.colour }}>
                {piece.text}
              </span>
            ) : (
              <span key={i} className="text-ink">
                {piece.text}
              </span>
            ),
          )}
        </code>
      </pre>
    </figure>
  );
}
