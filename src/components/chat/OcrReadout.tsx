import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, ScanText } from 'lucide-react';
import type { OcrPageRead } from '../../services/ocr.service';
import styles from './ChatSurface.module.css';

/**
 * What the OCR model is reading, as it reads it.
 *
 * Before this, an attached document produced one line — "Understanding
 * document…" — and then an answer. That is a claim that a model looked at the
 * page, not a demonstration of it, and there was no way to tell a clean read
 * from a garbled one until the answer was already wrong.
 *
 * This shows the model's own output arriving: each region as it is committed,
 * labelled with the block type the model assigned it (`title`, `text`,
 * `table`, `figure`, `footer`), and the transcription filling in underneath.
 * Nothing here is synthesised — every character came from the model, and a
 * page that produced nothing shows as producing nothing.
 */
export interface OcrReadoutProps {
  pages: OcrPageRead[];
  /** True while the run is still in flight. */
  live?: boolean;
}

/** Region labels the model emits, in the order it tends to emit them. */
const LABEL_TITLES: Record<string, string> = {
  title: 'Title',
  text: 'Text',
  table: 'Table',
  figure: 'Figure',
  footer: 'Footer',
};

function pageSummary(page: OcrPageRead): string {
  const where =
    page.pages && page.pages > 1 ? `Page ${page.page} of ${page.pages}` : 'Page 1';
  if (!page.done) return `${where} — reading…`;
  const seconds = page.elapsedMs != null ? page.elapsedMs / 1000 : null;
  const bits = [`${page.characters ?? 0} characters`];
  if (seconds != null) bits.push(`${seconds.toFixed(1)} s`);
  // Only a measured rate. A read that reported no elapsed time gets no rate
  // rather than one divided by a guess.
  if (seconds != null && seconds > 0 && page.characters != null) {
    bits.push(`${Math.round(page.characters / seconds)} char/s`);
  }
  // A read that ran out of budget did not finish, and the difference is not
  // cosmetic: it is the signature of the model looping on a page, where the
  // text below is a repeated fragment rather than the document.
  if (page.hitDecodeCap) bits.push('stopped at the token limit');
  return `${where} — ${bits.join(' · ')}`;
}

export function OcrReadout({ pages, live }: OcrReadoutProps) {
  const [open, setOpen] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow the read down the page. Only while it is live: yanking the scroll
  // on a finished panel the person is re-reading would be hostile.
  const written = pages.reduce(
    (n, p) => n + p.loose.length + p.regions.reduce((m, r) => m + r.text.length, 0),
    0,
  );
  useEffect(() => {
    if (!live || !open) return;
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [live, open, written]);

  if (pages.length === 0) return null;

  const model = pages.find(p => p.modelId)?.modelId ?? null;
  const detent = pages.find(p => p.detent)?.detent ?? null;
  const name = pages[0].name;
  const finished = pages.every(p => p.done);

  return (
    <section
      className={styles.ocrReadout}
      data-live={live && !finished ? '' : undefined}
      aria-label={`How the OCR model is reading ${name}`}
    >
      <button
        type="button"
        className={styles.ocrReadoutHead}
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
      >
        <ScanText size={14} className={styles.ocrReadoutIcon} aria-hidden="true" />
        <span className={styles.ocrReadoutTitle}>
          {finished ? 'Read by the OCR model' : 'The OCR model is reading'}
          <span className={styles.ocrReadoutFile}>{name}</span>
        </span>
        {/* The model is named because the routing explanation names it too,
          * and two places disagreeing about which model ran is exactly the
          * confusion this panel exists to remove. */}
        {model && (
          <span className={styles.ocrReadoutModel}>
            {model}
            {detent ? ` · ${detent}` : ''}
          </span>
        )}
        <ChevronDown
          size={14}
          className={styles.ocrReadoutChevron}
          data-open={open || undefined}
          aria-hidden="true"
        />
      </button>

      {open && (
        <div className={styles.ocrReadoutBody} ref={bodyRef}>
          {pages.map(page => (
            <div key={`${page.name}-${page.page}`} className={styles.ocrPage}>
              <p className={styles.ocrPageHead}>{pageSummary(page)}</p>

              {page.hitDecodeCap && (
                <p className={styles.ocrPageWarning} role="status">
                  This page filled its whole token budget without the model
                  stopping, which usually means it repeated itself rather than
                  finishing. Treat the text below as incomplete.
                </p>
              )}

              {page.regions.length === 0 && page.loose.trim() === '' && (
                <p className={styles.ocrPageEmpty}>
                  {page.done
                    ? 'Nothing was read on this page.'
                    : 'Waiting for the first region…'}
                </p>
              )}

              {page.regions.map(region => (
                <div key={region.index} className={styles.ocrRegion}>
                  <span
                    className={styles.ocrRegionLabel}
                    data-label={region.label}
                  >
                    {LABEL_TITLES[region.label] ?? region.label}
                  </span>
                  <pre className={styles.ocrRegionText}>{region.text}</pre>
                </div>
              ))}

              {page.loose.trim() !== '' && (
                <div className={styles.ocrRegion}>
                  <span className={styles.ocrRegionLabel} data-label="loose">
                    Ungrounded
                  </span>
                  <pre className={styles.ocrRegionText}>{page.loose}</pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
