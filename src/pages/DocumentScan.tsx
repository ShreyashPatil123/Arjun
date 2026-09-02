import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPageImage, type PageImage } from '../services/ocr.service';
import { ScanView } from '../components/ocr/ScanView';
import styles from './DocumentScan.module.css';

/**
 * Reading a document, one page at a time.
 *
 * The page to read is addressed in the URL — `?doc=<sha256>&page=<n>` — for
 * the same reason documents are stored by hash: a scan is reproducible, and a
 * link to "the thing I was looking at" survives a reload and means the same
 * bytes on another operator's machine. There is no document picker here;
 * documents arrive through ingestion and are addressed by what they are, not
 * by where they sit in a list.
 *
 * Pages are read one at a time rather than as a batch. That is not a
 * simplification — it is what the runtime supports well, and it is what makes
 * the scan watchable.
 */
export const DocumentScan = () => {
  const [params] = useSearchParams();
  const doc = params.get('doc');
  const rawPage = params.get('page');
  const page = Number(rawPage ?? '1');
  const [image, setImage] = useState<PageImage | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!doc || !Number.isInteger(page) || page < 1) return;
    let live = true;
    setImage(null);
    setLoadError(null);
    getPageImage(doc, page)
      .then((p) => {
        if (live) setImage(p);
      })
      .catch((e) => {
        if (live) setLoadError(String(e));
      });
    return () => {
      live = false;
    };
  }, [doc, page]);

  if (!doc) {
    return (
      <div className={styles.page}>
        <div className={styles.empty}>
          <h2 className={styles.title}>No document selected</h2>
          <p className={styles.body}>
            Open a document from Knowledge, or address one directly with{' '}
            <code className={styles.code}>?doc=&lt;sha256&gt;&amp;page=1</code>.
            Documents are addressed by hash, so the same link always opens the
            same bytes.
          </p>
        </div>
      </div>
    );
  }

  if (!Number.isInteger(page) || page < 1) {
    return (
      <div className={styles.page}>
        <div className={styles.empty}>
          <h2 className={styles.title}>That page number is not valid</h2>
          <p className={styles.body}>
            Pages are numbered from 1. Got{' '}
            <code className={styles.code}>{rawPage}</code>.
          </p>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className={styles.page}>
        <div className={styles.empty}>
          <h2 className={styles.title}>That page could not be loaded</h2>
          <p className={styles.body}>{loadError}</p>
        </div>
      </div>
    );
  }

  if (!image) {
    return (
      <div className={styles.page}>
        <div className={styles.empty}>
          <p className={styles.body}>Loading page {page}…</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <ScanView
        documentSha256={doc}
        page={page}
        pageImageSrc={image.dataUrl}
        /* The page's own dimensions, read from the file by the backend. The
         * overlay's coordinate space is the image's, so these must be what
         * the rasteriser produced — not a guess, and not the screen size. */
        pageWidth={image.width}
        pageHeight={image.height}
      />
    </div>
  );
};
