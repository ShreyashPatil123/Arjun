import React from 'react';

interface SarathiLogoProps {
  size?: number;
  /** Stroke weight for the rings. Lower reads lighter at large sizes. */
  strokeWidth?: number;
  className?: string;
  /** Accessible label. Pass null for a purely decorative mark. */
  title?: string | null;
}

const C = 50;
const SPOKES = 12;

/** Polar to cartesian, 0° at twelve o'clock. */
const pt = (r: number, deg: number): readonly [number, number] => {
  const rad = ((deg - 90) * Math.PI) / 180;
  return [C + r * Math.cos(rad), C + r * Math.sin(rad)];
};

const f = (n: number) => n.toFixed(2);

/** Evenly spaced dots on a ring. */
const ringDots = (radius: number, count: number, dotR: number, keyPrefix: string) =>
  Array.from({ length: count }, (_, i) => {
    const [x, y] = pt(radius, (360 / count) * i);
    return <circle key={`${keyPrefix}-${i}`} cx={x} cy={y} r={dotR} fill="currentColor" />;
  });

/** A leaf/petal: base at `rIn`, tip at `rOut`, bulging by `spread` degrees. */
const petal = (deg: number, rIn: number, rOut: number, spread: number) => {
  const [bx, by] = pt(rIn, deg);
  const [tx, ty] = pt(rOut, deg);
  const mid = (rIn + rOut) / 2;
  const [lx, ly] = pt(mid, deg - spread);
  const [rx, ry] = pt(mid, deg + spread);
  return `M ${f(bx)} ${f(by)} Q ${f(lx)} ${f(ly)} ${f(tx)} ${f(ty)} Q ${f(rx)} ${f(ry)} ${f(bx)} ${f(by)} Z`;
};

/**
 * The Sarathi mark — a dharma wheel (chakra).
 *
 * Drawn as geometry rather than shipped as an image: it inherits `currentColor`
 * so it follows the theme, stays sharp at any size, and cannot fail to load.
 *
 * This is a faithful rendering of the traditional form, not a pixel copy. The
 * details that make the wheel readable are all here — the serrated crown, the
 * beaded rim band, spokes beaded along their length with finer dotted rays
 * between them, and a petal rosette at the hub.
 */
export const SarathiLogo: React.FC<SarathiLogoProps> = ({
  size = 64,
  strokeWidth = 1.4,
  className,
  title = 'Sarathi',
}) => {
  // Unique per instance: two logos on one page must not share a mask id.
  const maskId = `sarathi-beads-${React.useId().replace(/:/g, '')}`;
  // Serrated crown: broad triangular points around the rim.
  const crown = Array.from({ length: SPOKES }, (_, i) => {
    const a = (360 / SPOKES) * i;
    const half = 360 / SPOKES / 2.2;
    const [tx, ty] = pt(49.5, a);
    const [lx, ly] = pt(42.5, a - half);
    const [rx, ry] = pt(42.5, a + half);
    return `M ${f(lx)} ${f(ly)} L ${f(tx)} ${f(ty)} L ${f(rx)} ${f(ry)} Z`;
  }).join(' ');

  // Main spokes: narrow at the hub, swelling to a rounded head near the rim.
  const spokeBodies = Array.from({ length: SPOKES }, (_, i) => {
    const a = (360 / SPOKES) * i;
    const [n1x, n1y] = pt(17, a - 2.6);
    const [n2x, n2y] = pt(17, a + 2.6);
    const [w1x, w1y] = pt(27, a - 5.2);
    const [w2x, w2y] = pt(27, a + 5.2);
    const [h1x, h1y] = pt(30.5, a - 4.2);
    const [h2x, h2y] = pt(30.5, a + 4.2);
    return (
      `M ${f(n1x)} ${f(n1y)} ` +
      `L ${f(w1x)} ${f(w1y)} ` +
      `Q ${f(pt(31.8, a - 2.2)[0])} ${f(pt(31.8, a - 2.2)[1])} ${f(h1x)} ${f(h1y)} ` +
      `L ${f(h2x)} ${f(h2y)} ` +
      `Q ${f(pt(31.8, a + 2.2)[0])} ${f(pt(31.8, a + 2.2)[1])} ${f(w2x)} ${f(w2y)} ` +
      `L ${f(n2x)} ${f(n2y)} Z`
    );
  }).join(' ');

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className={className}
      role={title ? 'img' : undefined}
      aria-label={title ?? undefined}
      aria-hidden={title ? undefined : true}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {title ? <title>{title}</title> : null}

      <path d={crown} fill="currentColor" />

      {/* Beaded rim band: ring, dots, ring. */}
      <circle cx={C} cy={C} r="42" stroke="currentColor" strokeWidth={strokeWidth} />
      {ringDots(38.4, 44, 1.05, 'rim')}
      <circle cx={C} cy={C} r="34.8" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx={C} cy={C} r="32.6" stroke="currentColor" strokeWidth={strokeWidth} />

      {/* Beads are punched through the spokes as real holes via a mask, so the
        * page background shows through. Painting them a background colour would
        * break the moment the logo sat on any other surface. */}
      <mask id={maskId}>
        <rect x="0" y="0" width="100" height="100" fill="white" />
        {Array.from({ length: SPOKES }, (_, i) => {
          const a = (360 / SPOKES) * i;
          return [20.5, 23.5, 26.5, 29.3].map((r, j) => {
            const [x, y] = pt(r, a);
            return <circle key={`bead-${i}-${j}`} cx={x} cy={y} r="1.15" fill="black" />;
          });
        })}
      </mask>

      <path d={spokeBodies} fill="currentColor" mask={`url(#${maskId})`} />

      {/* Finer dotted rays in the gaps between spokes. */}
      {Array.from({ length: SPOKES }, (_, i) => {
        const a = (360 / SPOKES) * i + 360 / SPOKES / 2;
        return [19.5, 22.5, 25.5, 28.5, 31].map((r, j) => {
          const [x, y] = pt(r, a);
          return <circle key={`ray-${i}-${j}`} cx={x} cy={y} r="1.05" fill="currentColor" />;
        });
      })}

      {/* Hub */}
      <circle cx={C} cy={C} r="16.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx={C} cy={C} r="14.6" stroke="currentColor" strokeWidth={strokeWidth} />

      {/* Petal rosette. */}
      <path
        d={Array.from({ length: SPOKES }, (_, i) => petal((360 / SPOKES) * i, 4.6, 13.4, 13)).join(' ')}
        fill="currentColor"
      />

      {/* Centre ring — stroked, so the core stays genuinely transparent. */}
      <circle cx={C} cy={C} r="4.1" stroke="currentColor" strokeWidth="2.6" />
    </svg>
  );
};
