import { describe, expect, it } from 'vitest';
import {
  applyAttachmentOcrEvent,
  type AttachmentOcrEvent,
  type OcrPageRead,
} from './ocr.service';

/**
 * Folding a live OCR read into something renderable.
 *
 * The properties that matter are the ones that fail silently: a delta landing
 * in the wrong region reads as a plausible page with the words shuffled, and
 * a delta for a region that never opened either invents a region or is lost.
 * Both would produce a readout that disagrees with the text the answer was
 * actually built from, which is the one thing this panel exists to rule out.
 */
const FILE = 'work-order.png';

function region(index: number, label: string, page = 1): AttachmentOcrEvent {
  return { event: 'region', name: FILE, page, index, label };
}

function text(
  index: number | null,
  delta: string,
  page = 1,
): AttachmentOcrEvent {
  return { event: 'text', name: FILE, page, index, delta };
}

function fold(events: AttachmentOcrEvent[]): OcrPageRead[] {
  return events.reduce<OcrPageRead[]>(
    (acc, e) => applyAttachmentOcrEvent(acc, e),
    [],
  );
}

describe('applyAttachmentOcrEvent', () => {
  it('opens a region before any of its text', () => {
    const [page] = fold([region(0, 'title')]);
    expect(page.regions).toEqual([{ index: 0, label: 'title', text: '' }]);
    expect(page.done).toBe(false);
  });

  it('lands a delta in the region that opened it, not the newest one', () => {
    const [page] = fold([
      region(0, 'title'),
      region(1, 'text'),
      text(0, 'QUARTERLY FIELD REPORT'),
      text(1, 'Substation 14'),
      text(0, ' (rev B)'),
    ]);
    expect(page.regions).toEqual([
      { index: 0, label: 'title', text: 'QUARTERLY FIELD REPORT (rev B)' },
      { index: 1, label: 'text', text: 'Substation 14' },
    ]);
  });

  it('keeps ungrounded text rather than dropping or inventing a region', () => {
    const [page] = fold([region(0, 'title'), text(null, 'stray header\n')]);
    expect(page.regions).toHaveLength(1);
    expect(page.loose).toBe('stray header\n');
  });

  it('does not conjure a region for a delta whose region never opened', () => {
    const [page] = fold([text(7, 'orphaned')]);
    expect(page.regions).toEqual([]);
    expect(page.loose).toBe('orphaned');
  });

  it('keeps pages of one document separate and in arrival order', () => {
    const pages = fold([
      region(0, 'title', 1),
      text(0, 'Page one', 1),
      region(0, 'title', 2),
      text(0, 'Page two', 2),
    ]);
    expect(pages.map(p => p.page)).toEqual([1, 2]);
    expect(pages[0].regions[0].text).toBe('Page one');
    expect(pages[1].regions[0].text).toBe('Page two');
  });

  it('only reports measured figures, and only once the page completes', () => {
    const before = fold([region(0, 'title'), text(0, 'REPORT')])[0];
    expect(before.done).toBe(false);
    expect(before.characters).toBeNull();
    expect(before.elapsedMs).toBeNull();
    expect(before.modelId).toBeNull();

    const after = fold([
      region(0, 'title'),
      text(0, 'REPORT'),
      {
        event: 'page',
        name: FILE,
        page: 1,
        pages: 1,
        modelId: 'unlimited-ocr-q6-k',
        detent: 'detailed',
        characters: 6,
        elapsedMs: 1200,
        hitDecodeCap: false,
        looped: false,
      },
    ])[0];
    expect(after.done).toBe(true);
    expect(after.characters).toBe(6);
    expect(after.elapsedMs).toBe(1200);
    expect(after.modelId).toBe('unlimited-ocr-q6-k');
    expect(after.detent).toBe('detailed');
    // The text survives the completion event rather than being replaced by
    // the summary of it.
    expect(after.regions[0].text).toBe('REPORT');
  });

  // A page read twice — a retry, or a duplicate subscription — must show
  // one set of regions, not two interleaved sets with the text split
  // between them.
  it('replaces a region when its index is seen again on the same page', () => {
    const page = fold([
      region(0, 'title'),
      text(0, 'FIRST ATTEMPT'),
      region(1, 'text'),
      text(1, 'body'),
      region(0, 'title'),
      text(0, 'SECOND ATTEMPT'),
    ])[0];
    expect(page.regions).toEqual([
      { index: 0, label: 'title', text: 'SECOND ATTEMPT' },
    ]);
  });

  // The failure this field exists for: a page that looped filled its budget
  // and, without the flag, arrived looking like any other completed read.
  it('records a read that ran out of budget rather than finishing', () => {
    const page = fold([
      region(0, 'text'),
      text(0, '*\n*\n*\n'),
      {
        event: 'page',
        name: FILE,
        page: 1,
        pages: 1,
        modelId: 'unlimited-ocr-q6-k',
        detent: 'maximum',
        characters: 16384,
        elapsedMs: 42000,
        hitDecodeCap: true,
        // Not the repetition guard: that cuts a page long before the decode
        // budget runs out, so a page that reaches the cap has `looped` false.
        // The two flags name different failures and this fixture is the cap.
        looped: false,
      },
    ])[0];
    expect(page.done).toBe(true);
    expect(page.hitDecodeCap).toBe(true);
    expect(page.looped).toBe(false);
  });

  // The other way a page stops short, and the commoner one. The backend cuts
  // a repeating read as soon as it recognises the repetition, so the decode
  // cap is never reached and `hitDecodeCap` stays false — `looped` is the only
  // thing standing between a truncated page and a readout that presents it as
  // a complete one.
  it('records a read that was cut because the model repeated itself', () => {
    const page = fold([
      region(0, 'text'),
      text(0, 'the pump the pump the pump'),
      {
        event: 'page',
        name: FILE,
        page: 1,
        pages: 1,
        modelId: 'unlimited-ocr-q6-k',
        detent: 'maximum',
        characters: 26,
        elapsedMs: 8000,
        hitDecodeCap: false,
        looped: true,
      },
    ])[0];
    expect(page.done).toBe(true);
    expect(page.looped).toBe(true);
    expect(page.hitDecodeCap).toBe(false);
  });

  it('defaults to not having hit the cap before the page completes', () => {
    expect(fold([region(0, 'title')])[0].hitDecodeCap).toBe(false);
    expect(fold([region(0, 'title')])[0].looped).toBe(false);
  });

  it('never mutates the array it was given', () => {
    const first = fold([region(0, 'title')]);
    const snapshot = JSON.parse(JSON.stringify(first));
    applyAttachmentOcrEvent(first, text(0, 'REPORT'));
    expect(first).toEqual(snapshot);
  });
});
