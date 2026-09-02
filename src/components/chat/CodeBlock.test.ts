import { describe, expect, it } from 'vitest';
import { highlight } from './CodeBlock';

/**
 * The one property that has to hold.
 *
 * A highlighter that mis-colours a token is a cosmetic problem. A highlighter
 * that drops, duplicates or reorders a character hands somebody code that
 * does not compile, from a block with a Copy button on it. Every test here is
 * ultimately that assertion under a different input.
 */
function rebuilt(source: string): string {
  return highlight(source)
    .map(p => p.text)
    .join('');
}

const CASES: [string, string][] = [
  ['empty', ''],
  ['plain prose', 'just some words'],
  ['rust', 'pub fn main() {\n    let x = 42; // the answer\n}'],
  ['typescript', 'export const a = `tpl ${b}`;\n/* block */\nfoo(1, 0xFF, 1.5e3);'],
  ['python', 'def f(n):\n    return "s" if n else \'t\'  # comment'],
  ['sql', "select * from t where name = 'x' -- trailing"],
  ['unterminated string', 'const s = "never closed\nnext();'],
  ['unterminated block comment', 'code();\n/* runs off the end'],
  ['lone backslash', 'a = "\\\\"'],
  ['unicode', 'let s = "réponse — 日本語";'],
  ['only punctuation', '{}[]();,.<>'],
];

describe('highlight', () => {
  for (const [name, source] of CASES) {
    it(`emits ${name} back exactly, character for character`, () => {
      expect(rebuilt(source)).toBe(source);
    });
  }

  it('emits every source it is given back unchanged, truncated anywhere', () => {
    // A streamed block arrives cut off at arbitrary offsets. Each prefix has
    // to survive on its own, because each one gets rendered.
    const source = CASES.map(c => c[1]).join('\n');
    for (let cut = 0; cut <= source.length; cut += 1) {
      const prefix = source.slice(0, cut);
      expect(rebuilt(prefix)).toBe(prefix);
    }
  });

  it('colours a keyword, a string and a comment differently', () => {
    const pieces = highlight('const s = "hi"; // note');
    const colourOf = (text: string) => pieces.find(p => p.text === text)?.colour;
    expect(colourOf('const')).toBe('#569CD6');
    expect(colourOf('"hi"')).toBe('#CE9178');
    expect(colourOf('// note')).toBe('#6A9955');
  });

  it('colours a name followed by a bracket as a call, not as plain text', () => {
    const pieces = highlight('render(x)');
    expect(pieces.find(p => p.text === 'render')?.colour).toBe('#DCDCAA');
  });

  it('leaves an ordinary identifier uncoloured', () => {
    const pieces = highlight('total = 1');
    expect(pieces.find(p => p.text.includes('total'))?.colour).toBeUndefined();
  });

  it('does not treat a hash inside a string as the start of a comment', () => {
    // Deliberately not a URL: the egress gate reads string literals across
    // the whole tree, and a host in a fixture is indistinguishable from a
    // host in shipping code.
    const pieces = highlight('label = "count#42 / total"');
    expect(pieces.some(p => p.colour === '#6A9955')).toBe(false);
  });
});
