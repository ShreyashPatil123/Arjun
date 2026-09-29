import { describe, expect, it } from 'vitest';

import { displayMathAt, inlineMathAt, mathToPlain, parseTex } from './texMath';

const plain = (tex: string) => mathToPlain(parseTex(tex));

describe('parseTex: the LaTeX models write in engineering answers', () => {
  /**
   * The defect, verbatim from a conversion answer: the chat printed
   * `\[ 1 \text{ psi} = 0.06894757 \text{ bar} \]` as source.
   */
  it('reads the psi-to-bar working that was printed as source', () => {
    expect(plain(String.raw`1 \text{ psi} = 0.06894757 \text{ bar}`)).toBe('1 psi = 0.06894757 bar');
    expect(
      plain(String.raw`150 \text{ psi} \times 0.06894757 \frac{\text{bar}}{\text{psi}} = 10.3421355 \text{ bar}`),
    ).toBe('150 psi × 0.06894757 bar/psi = 10.3421355 bar');
    expect(plain(String.raw`150 \text{ psi} \approx 10.34 \text{ bar}`)).toBe('150 psi ≈ 10.34 bar');
  });

  it('keeps a fraction as a fraction, not as flattened text', () => {
    const [frac] = parseTex(String.raw`\frac{\text{bar}}{\text{psi}}`);
    expect(frac).toMatchObject({
      kind: 'frac',
      num: [{ kind: 'text', text: 'bar' }],
      den: [{ kind: 'text', text: 'psi' }],
    });
    expect(plain(String.raw`\frac{a+b}{2}`)).toBe('(a + b)/2');
  });

  it('sets minus signs, margins and units as they read', () => {
    expect(plain(String.raw`A = 4.7 - 5.0 = -0.3\,\text{mm}`)).toBe('A = 4.7 − 5.0 = −0.3 mm');
    expect(plain(String.raw`25^\circ\text{C}`)).toBe('25°C');
    expect(plain(String.raw`25^{\circ} C`)).toBe('25° C');
  });

  it('writes powers, indices and roots', () => {
    expect(plain('x^2 + y_1')).toBe('x² + y₁');
    expect(plain(String.raw`\text{m}^{-1}`)).toBe('m⁻¹');
    expect(plain(String.raw`\sqrt{2}`)).toBe('√2');
    expect(plain(String.raw`\sqrt{a+b}`)).toBe('√(a + b)');
    expect(plain('t_{wall}')).toBe('t_(wall)');
  });

  it('reads Greek letters, operators and relations', () => {
    expect(plain(String.raw`\Delta p \leq 0.5\,\text{bar}`)).toBe('Δp ≤ 0.5 bar');
    expect(plain(String.raw`\sigma = \frac{F}{A}`)).toBe('σ = F/A');
    expect(plain(String.raw`a \cdot b \neq c \pm d`)).toBe('a · b ≠ c ± d');
    expect(plain(String.raw`\log_{10} x`)).toBe('log₁₀ x');
    expect(plain(String.raw`\sin \theta`)).toBe('sin θ');
  });

  it('drops sizing commands and the spaces just inside brackets', () => {
    expect(plain(String.raw`\left( \frac{1}{2} \right)`)).toBe('(1/2)');
    expect(plain(String.raw`\bigl[ x \bigr]`)).toBe('[x]');
  });

  /** Nothing a model wrote disappears: an unknown command keeps its name. */
  it('keeps the name of a command it does not know', () => {
    expect(plain(String.raw`\widget{x}`)).toBe('widgetx');
  });

  it('never throws on half-written math, which is what streaming produces', () => {
    for (const partial of [String.raw`\frac{1`, String.raw`x^`, String.raw`\text{ps`, '\\', '{{{', '}}}']) {
      expect(() => parseTex(partial)).not.toThrow();
    }
  });

  /** A prototype key is not a command. */
  it('does not read Object.prototype as a symbol table', () => {
    expect(plain(String.raw`\constructor \toString`)).toBe('constructor toString');
  });
});

describe('inlineMathAt: finding math inside a line of prose', () => {
  it('finds the unambiguous delimiters', () => {
    const line = String.raw`Speed is \(v = 5\) m/s`;
    expect(inlineMathAt(line, line.indexOf('\\'))).toEqual({ tex: 'v = 5', display: false, end: line.indexOf(' m/s') });
    const display = String.raw`so \[ a = b \] holds`;
    expect(inlineMathAt(display, display.indexOf('\\'))).toMatchObject({ tex: ' a = b ', display: true });
  });

  it('reads $…$ and $$…$$', () => {
    expect(inlineMathAt('where $x^2$ grows', 6)).toMatchObject({ tex: 'x^2', display: false });
    expect(inlineMathAt('$$E = mc^2$$', 0)).toMatchObject({ tex: 'E = mc^2', display: true });
  });

  /** Pandoc's rule keeps prices as prose. */
  it('leaves amounts of money alone', () => {
    const line = 'It costs $5 and $6 to ship.';
    expect(inlineMathAt(line, line.indexOf('$'))).toBeNull();
    expect(inlineMathAt('between $5-$6 each', 8)).toBeNull();
    expect(inlineMathAt(String.raw`an escaped \$5$`, 12)).toBeNull();
  });

  it('ignores an unclosed delimiter', () => {
    expect(inlineMathAt(String.raw`\( x = 1`, 0)).toBeNull();
  });
});

describe('displayMathAt: a formula on lines of its own', () => {
  it('reads a one-line formula', () => {
    expect(displayMathAt([String.raw`\[ 1 \text{ psi} = 0.06894757 \text{ bar} \]`], 0)).toEqual({
      tex: String.raw`1 \text{ psi} = 0.06894757 \text{ bar}`,
      next: 1,
    });
  });

  it('reads a formula spread over several lines', () => {
    expect(displayMathAt(['$$', 'a = b', '+ c', '$$', 'after'], 0)).toEqual({ tex: 'a = b\n+ c', next: 4 });
    expect(displayMathAt(['\\[', 'x', '\\]'], 0)).toEqual({ tex: 'x', next: 3 });
  });

  it('leaves prose after the close, an unclosed formula, and ordinary lines to the paragraph', () => {
    expect(displayMathAt([String.raw`\[ x \] where x is the reading`], 0)).toBeNull();
    expect(displayMathAt(['\\[', 'x = 1'], 0)).toBeNull();
    expect(displayMathAt(['Result: 10.34 bar'], 0)).toBeNull();
  });
});
