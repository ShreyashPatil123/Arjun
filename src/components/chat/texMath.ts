/**
 * LaTeX math, as models write it in chat, turned into something a person can
 * read.
 *
 * Models answer engineering questions in LaTeX — `\[ 150 \text{ psi} \times
 * 0.06894757 \frac{\text{bar}}{\text{psi}} \]` — and the chat printed that
 * source verbatim. This reads the subset models actually use (text, fractions,
 * powers and indices, roots, Greek letters, operators, relations, spacing) and
 * leaves anything else readable rather than dropping it: an unknown command
 * keeps its name, so no part of an answer disappears.
 *
 * Hand-rolled for the same reason as `Markdown.tsx`: no rendering library, and
 * nothing fetched — a math font from a CDN would be an outbound call. The
 * output is plain data rendered as React text, so nothing a model writes here
 * can become markup.
 */

export type MathNode =
  | { kind: 'text'; text: string }
  | { kind: 'frac'; num: MathNode[]; den: MathNode[] }
  | { kind: 'sup'; body: MathNode[] }
  | { kind: 'sub'; body: MathNode[] }
  | { kind: 'sqrt'; body: MathNode[] };

const SYMBOLS: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ',
  eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν',
  xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ',
  chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ',
  Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
  infty: '∞', partial: '∂', nabla: '∇', degree: '°', textdegree: '°', circ: '∘', prime: '′',
  ldots: '…', dots: '…', cdots: '⋯', vdots: '⋮', ddots: '⋱', angle: '∠', perp: '⊥',
  parallel: '∥', therefore: '∴', because: '∵', forall: '∀', exists: '∃', emptyset: '∅',
  hbar: 'ℏ', ell: 'ℓ', sum: '∑', prod: '∏', int: '∫', oint: '∮', lceil: '⌈', rceil: '⌉',
  lfloor: '⌊', rfloor: '⌋', langle: '⟨', rangle: '⟩', vert: '|', percent: '%',
};

/** Binary operators and relations: spaced, as TeX spaces them. */
const SPACED: Record<string, string> = {
  times: '×', cdot: '·', div: '÷', pm: '±', mp: '∓', ast: '∗', oplus: '⊕', otimes: '⊗',
  cap: '∩', cup: '∪', setminus: '∖', wedge: '∧', vee: '∨', land: '∧', lor: '∨',
  approx: '≈', simeq: '≃', sim: '∼', cong: '≅', equiv: '≡', neq: '≠', ne: '≠', leq: '≤',
  le: '≤', geq: '≥', ge: '≥', ll: '≪', gg: '≫', lt: '<', gt: '>', propto: '∝', in: '∈',
  notin: '∉', subset: '⊂', subseteq: '⊆', supset: '⊃', supseteq: '⊇', mid: '∣',
  to: '→', rightarrow: '→', leftarrow: '←', gets: '←', Rightarrow: '⇒', Leftarrow: '⇐',
  leftrightarrow: '↔', Leftrightarrow: '⇔', implies: '⇒', iff: '⇔', mapsto: '↦',
  longrightarrow: '⟶', Longrightarrow: '⟹', uparrow: '↑', downarrow: '↓',
};

/** Commands whose argument is text, spaces and all: `\text{ psi}`. */
const TEXT_MODE = new Set([
  'text', 'textrm', 'textit', 'textbf', 'textsf', 'texttt', 'textup', 'textnormal',
  'mbox', 'hbox', 'operatorname',
]);

/** Commands that only change the font of math: their argument is still math. */
const MATH_FONT = new Set([
  'mathrm', 'mathit', 'mathbf', 'mathsf', 'mathtt', 'mathnormal', 'boldsymbol', 'bm',
  'mathcal', 'mathbb', 'mathfrak', 'mathscr',
]);

/** Named functions, set upright: `\sin x`, `\log_{10}`. */
const FUNCTIONS = new Set([
  'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
  'tanh', 'log', 'ln', 'lg', 'exp', 'lim', 'max', 'min', 'sup', 'inf', 'det', 'deg', 'gcd',
  'arg', 'dim', 'ker', 'Pr',
]);

const SPACES: Record<string, string> = {
  ',': ' ', ':': ' ', ';': ' ', ' ': ' ', '!': '',
  quad: ' ', qquad: '  ', enspace: ' ', thinspace: ' ',
};

/** Sizing and layout commands that carry no content of their own. */
const IGNORED = new Set([
  'left', 'right', 'middle', 'big', 'Big', 'bigg', 'Bigg', 'bigl', 'bigr', 'Bigl', 'Bigr',
  'biggl', 'biggr', 'Biggl', 'Biggr', 'limits', 'nolimits', 'displaystyle', 'textstyle',
  'scriptstyle', 'nonumber', 'notag', 'hline', 'centering',
]);

/** Accents, drawn with a combining mark when they sit on a single character. */
const ACCENTS: Record<string, string> = {
  bar: '̅', overline: '̅', hat: '̂', widehat: '̂', vec: '⃗',
  dot: '̇', ddot: '̈', tilde: '̃', widetilde: '̃', underline: '̲',
};

const FRACTIONS = new Set(['frac', 'dfrac', 'tfrac', 'cfrac']);

/** Own keys only: `\constructor` must not find `Object.prototype.constructor`. */
function lookup(table: Record<string, string>, key: string): string | undefined {
  return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
}

class TexParser {
  private i = 0;

  constructor(private readonly s: string) {}

  parse(): MathNode[] {
    return this.sequence(false);
  }

  /** Nodes up to the end, or up to the `}` that closes the current group. */
  private sequence(inGroup: boolean): MathNode[] {
    const nodes: MathNode[] = [];
    while (this.i < this.s.length) {
      const c = this.s[this.i];
      if (c === '}') {
        this.i += 1;
        if (inGroup) return nodes;
        continue; // a stray close brace: nothing to show
      }
      if (c === '{') {
        this.i += 1;
        nodes.push(...this.sequence(true));
        continue;
      }
      if (c === '^' || c === '_') {
        this.i += 1;
        const body = this.argument();
        // `^\circ` is how TeX writes a degree sign.
        if (c === '^' && plainOf(body).trim() === '∘') nodes.push(text('°'));
        else nodes.push({ kind: c === '^' ? 'sup' : 'sub', body });
        continue;
      }
      if (c === '\\') {
        this.command(nodes);
        continue;
      }
      this.i += 1;
      if (/\s/.test(c)) nodes.push(text(' '));
      else if (c === '&') nodes.push(text(' ')); // an alignment point
      else if (c === '~') nodes.push(text(' '));
      else if (c === '-') nodes.push(text('−'));
      else if (c === "'") nodes.push(text('′'));
      else if (c === '=' || c === '+' || c === '<' || c === '>') nodes.push(text(` ${c} `));
      else nodes.push(text(c));
    }
    return nodes;
  }

  /** One argument: a braced group, one command, or one character. */
  private argument(): MathNode[] {
    this.skipSpaces();
    if (this.i >= this.s.length) return [];
    const c = this.s[this.i];
    if (c === '{') {
      this.i += 1;
      return this.sequence(true);
    }
    if (c === '\\') {
      const nodes: MathNode[] = [];
      this.command(nodes);
      return nodes;
    }
    this.i += 1;
    return [text(c === '-' ? '−' : c)];
  }

  /** The raw text of a braced group (or one character), for text-mode commands. */
  private rawGroup(): string {
    this.skipSpaces();
    if (this.s[this.i] !== '{') {
      const c = this.s[this.i] ?? '';
      this.i += c ? 1 : 0;
      return c;
    }
    let depth = 0;
    const start = this.i + 1;
    for (; this.i < this.s.length; this.i += 1) {
      const c = this.s[this.i];
      if (c === '\\') {
        this.i += 1; // an escaped brace is not a brace
        continue;
      }
      if (c === '{') depth += 1;
      if (c === '}') {
        depth -= 1;
        if (depth === 0) {
          this.i += 1;
          return this.s.slice(start, this.i - 1);
        }
      }
    }
    return this.s.slice(start); // unclosed: take the rest
  }

  private skipSpaces() {
    while (this.i < this.s.length && /\s/.test(this.s[this.i])) this.i += 1;
  }

  private command(nodes: MathNode[]) {
    this.i += 1; // the backslash
    if (this.i >= this.s.length) return;
    const first = this.s[this.i];
    if (!/[A-Za-z]/.test(first)) {
      this.i += 1;
      const space = lookup(SPACES, first);
      if (first === '\\') nodes.push(text('\n'));
      else if (space !== undefined) nodes.push(text(space));
      else if (first === '|') nodes.push(text('‖'));
      else nodes.push(text(first)); // \{ \} \% \_ \& \# \$
      return;
    }
    let end = this.i;
    while (end < this.s.length && /[A-Za-z]/.test(this.s[end])) end += 1;
    const name = this.s.slice(this.i, end);
    this.i = end;

    if (FRACTIONS.has(name)) {
      const num = this.argument();
      const den = this.argument();
      nodes.push({ kind: 'frac', num, den });
      return;
    }
    if (name === 'sqrt') {
      this.skipSpaces();
      if (this.s[this.i] === '[') {
        const close = this.s.indexOf(']', this.i);
        this.i = close === -1 ? this.s.length : close + 1; // the index of a root is left out
      }
      nodes.push({ kind: 'sqrt', body: this.argument() });
      return;
    }
    if (TEXT_MODE.has(name)) {
      nodes.push(text(textMode(this.rawGroup())));
      return;
    }
    if (MATH_FONT.has(name)) {
      nodes.push(...this.argument());
      return;
    }
    if (name === 'begin' || name === 'end') {
      const env = this.rawGroup();
      // An array's column spec is a second group that is not content.
      if (name === 'begin' && /^(array|tabular)\*?$/.test(env)) this.rawGroup();
      if (name === 'end') nodes.push(text('\n'));
      return;
    }
    if (name === 'tag') {
      nodes.push(text(` (${textMode(this.rawGroup())})`));
      return;
    }
    const accent = lookup(ACCENTS, name);
    if (accent !== undefined) {
      const body = this.argument();
      const plain = plainOf(body);
      if ([...plain].length === 1) nodes.push(text(plain + accent));
      else nodes.push(...body);
      return;
    }
    if (IGNORED.has(name)) {
      // `\left.` and `\right.` are invisible delimiters.
      if ((name === 'left' || name === 'right') && this.s[this.i] === '.') this.i += 1;
      return;
    }
    const space = lookup(SPACES, name);
    if (space !== undefined) {
      nodes.push(text(space));
      return;
    }
    // TeX swallows the spaces after a named command: `\pi r` is `πr`.
    const spaced = lookup(SPACED, name);
    if (spaced !== undefined) {
      nodes.push(text(` ${spaced} `));
      this.skipSpaces();
      return;
    }
    const symbol = lookup(SYMBOLS, name);
    if (symbol !== undefined) {
      nodes.push(text(symbol));
      this.skipSpaces();
      return;
    }
    if (FUNCTIONS.has(name)) {
      this.skipSpaces();
      // `\sin x` is "sin x"; `\log_{10}` and `\sin(x)` take no space.
      const next = this.s[this.i] ?? '';
      nodes.push(text(next === '' || '_^([{'.includes(next) ? name : `${name} `));
      return;
    }
    // Unknown: keep the name, so nothing the model wrote is lost.
    nodes.push(text(name));
  }
}

function text(t: string): MathNode {
  return { kind: 'text', text: t };
}

/** The content of `\text{…}`: literal, apart from escapes and the odd symbol. */
function textMode(raw: string): string {
  return raw
    .replace(/\\(textdegree|degree)\b\s*/g, '°')
    .replace(/\\[,:;!]/g, ' ')
    .replace(/\\([%_&#${}\s])/g, '$1')
    .replace(/\$/g, '');
}

/** Merges adjacent text and settles spacing, the way TeX would set it. */
function tidy(nodes: MathNode[]): MathNode[] {
  const out: MathNode[] = [];
  for (const node of nodes) {
    const n: MathNode =
      node.kind === 'text'
        ? { kind: 'text', text: node.text }
        : node.kind === 'frac'
          ? { kind: 'frac', num: trimEnds(tidy(node.num)), den: trimEnds(tidy(node.den)) }
          : { kind: node.kind, body: trimEnds(tidy(node.body)) };
    const last = out[out.length - 1];
    if (n.kind === 'text' && last?.kind === 'text') last.text += n.text;
    else out.push(n);
  }
  // The spaces the model wrote are kept (they are how it meant the formula to
  // read), collapsed, and dropped just inside brackets: `\left( x \right)`.
  for (const n of out) {
    if (n.kind === 'text') {
      n.text = n.text
        .replace(/ {2,}/g, ' ')
        .replace(/ *\n */g, '\n')
        .replace(/([([{⟨]) +/g, '$1')
        .replace(/ +([)\]}⟩])/g, '$1');
    }
  }
  return out;
}

function trimEnds(nodes: MathNode[]): MathNode[] {
  const first = nodes[0];
  if (first?.kind === 'text') first.text = first.text.replace(/^[ \n]+/, '');
  const last = nodes[nodes.length - 1];
  if (last?.kind === 'text') last.text = last.text.replace(/[ \n]+$/, '');
  return nodes.filter(n => n.kind !== 'text' || n.text !== '');
}

/** Parses the inside of a math span: what sits between `\[` and `\]`. */
export function parseTex(source: string): MathNode[] {
  return trimEnds(tidy(new TexParser(source).parse()));
}

const SUPERSCRIPT: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸',
  '9': '⁹', '+': '⁺', '−': '⁻', '=': '⁼', '(': '⁽', ')': '⁾', n: 'ⁿ', i: 'ⁱ',
};
const SUBSCRIPT: Record<string, string> = {
  '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈',
  '9': '₉', '+': '₊', '−': '₋', '=': '₌', '(': '₍', ')': '₎', a: 'ₐ', e: 'ₑ', o: 'ₒ',
  x: 'ₓ', i: 'ᵢ', r: 'ᵣ', u: 'ᵤ', v: 'ᵥ',
};

function scripted(plain: string, table: Record<string, string>, marker: string): string {
  const chars = [...plain];
  if (chars.length > 0 && chars.every(c => lookup(table, c) !== undefined)) {
    return chars.map(c => table[c]).join('');
  }
  return chars.length === 1 ? `${marker}${plain}` : `${marker}(${plain})`;
}

function grouped(plain: string): string {
  return /[ +−=×·/]/.test(plain) ? `(${plain})` : plain;
}

function plainOf(nodes: MathNode[]): string {
  return nodes
    .map(n => {
      switch (n.kind) {
        case 'text':
          return n.text;
        case 'frac':
          return `${grouped(plainOf(n.num))}/${grouped(plainOf(n.den))}`;
        case 'sup':
          return scripted(plainOf(n.body), SUPERSCRIPT, '^');
        case 'sub':
          return scripted(plainOf(n.body), SUBSCRIPT, '_');
        case 'sqrt':
          return `√${grouped(plainOf(n.body))}`;
      }
      return '';
    })
    .join('');
}

/**
 * The same math as one line of plain text: `150 psi × 0.06894757 bar/psi`.
 * What a screen reader is given, and what the tests read.
 */
export function mathToPlain(nodes: MathNode[]): string {
  return plainOf(nodes)
    .replace(/[ -   ]/g, ' ')
    .replace(/ {2,}/g, ' ')
    .trim();
}

/** A math span found in a line of prose. */
export interface InlineMath {
  tex: string;
  display: boolean;
  /** Index just past the closing delimiter. */
  end: number;
}

/**
 * The math span that starts at `line[i]`, if one does.
 *
 * `\( … \)` and `\[ … \]` are unambiguous. A single `$` follows Pandoc's rule —
 * no space just inside either dollar, and no digit right after the closing
 * one — so prices such as "$5 and $6" stay prose.
 */
export function inlineMathAt(line: string, i: number): InlineMath | null {
  if (line[i] === '\\' && (line[i + 1] === '(' || line[i + 1] === '[')) {
    const close = line[i + 1] === '(' ? '\\)' : '\\]';
    const end = line.indexOf(close, i + 2);
    if (end === -1) return null;
    const tex = line.slice(i + 2, end);
    if (tex.trim() === '') return null;
    return { tex, display: line[i + 1] === '[', end: end + 2 };
  }
  if (line[i] !== '$' || (i > 0 && line[i - 1] === '\\')) return null;
  if (line[i + 1] === '$') {
    const end = line.indexOf('$$', i + 2);
    if (end === -1 || line.slice(i + 2, end).trim() === '') return null;
    return { tex: line.slice(i + 2, end), display: true, end: end + 2 };
  }
  if (!line[i + 1] || /\s/.test(line[i + 1])) return null;
  for (let j = i + 1; j < line.length; j += 1) {
    if (line[j] !== '$' || line[j - 1] === '\\') continue;
    if (/\s/.test(line[j - 1]) || /\d/.test(line[j + 1] ?? '')) return null;
    return { tex: line.slice(i + 1, j), display: false, end: j + 1 };
  }
  return null;
}

const DISPLAY_OPEN = /^\s*(\\\[|\$\$)/;

/**
 * A display formula starting at `lines[i]`: `\[` or `$$` opening the line, and
 * the matching close ending a line (this one or a later one). A formula with
 * prose after its close on the same line is left to the inline reader, which
 * keeps the prose. An unclosed one — still streaming — is shown as written
 * until its close arrives.
 */
export function displayMathAt(lines: string[], i: number): { tex: string; next: number } | null {
  const open = lines[i].match(DISPLAY_OPEN);
  if (!open) return null;
  const close = open[1] === '$$' ? '$$' : '\\]';
  const body: string[] = [];
  let rest = lines[i].slice(open[0].length);
  for (let j = i; j < lines.length; j += 1) {
    if (j > i) rest = lines[j];
    const end = rest.indexOf(close);
    if (end !== -1) {
      if (rest.slice(end + close.length).trim() !== '') return null;
      body.push(rest.slice(0, end));
      const tex = body.join('\n').trim();
      return tex ? { tex, next: j + 1 } : null;
    }
    body.push(rest);
  }
  return null;
}
