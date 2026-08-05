import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Search,
  AlertTriangle,
  RefreshCw,
  Download,
  Heart,
  Clock,
  Layers,
  ChevronDown,
  ChevronRight,
  X,
} from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import {
  browseModelCards,
  findModelAdapters,
  formatSize,
  type AdapterPage,
  type CatalogPage,
  type CategoryCount,
} from '../services/catalog.service';
import {
  downloadAdapter,
  getAdapterDetails,
  type AdapterDetails,
} from '../services/adapters.service';
import { startModelDownload } from '../services/download.service';
import { useDownloads } from '../hooks/useDownloads';
import { useToast } from '../hooks/useToast';
import { KIND_LABELS, type ModelCard, type ModelCategory } from '../types/ai';
import styles from './Browse.module.css';

/** A downloadable size, as the card reports it. */
type Quant = ModelCard['quantizations'][number];

/**
 * Model browser: category sidebar, search, and cards.
 *
 * Filtering happens on already-fetched cards, but *searching* goes back to
 * HuggingFace — the loaded page is only the popular sweep, so filtering it for
 * a specific fine-tune would wrongly report that nothing matches.
 *
 * Details open in a drawer rather than inline. Cards sit in a grid, and grid
 * rows share a height, so expanding one card stretched its neighbours into tall
 * empty boxes; the drawer also gives the size and adapter tables room to be
 * read, which a third of a card's width did not.
 */
export const Browse: React.FC = () => {
  const [page, setPage] = useState<CatalogPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ModelCategory | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  /** The term the current results reflect, so the header can say so. */
  const [activeSearch, setActiveSearch] = useState('');
  /** The card whose details are open in the drawer. */
  const [detailCard, setDetailCard] = useState<ModelCard | null>(null);

  const load = useCallback(async (query?: string) => {
    setLoading(true);
    setError(null);
    try {
      setPage(await browseModelCards(query));
      setActiveSearch(query ?? '');
    } catch (err) {
      // These messages are written for people — rate limiting says to add a
      // token — so they are shown as-is rather than replaced.
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(() => {
    if (!page) return [];
    if (!selected) return page.cards;
    return page.cards.filter((c) => c.categories.includes(selected));
  }, [page, selected]);

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    load(searchTerm.trim() || undefined);
    setSelected(null);
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Models</h1>
          <p className={styles.subtitle}>
            {activeSearch
              ? `Results for “${activeSearch}”`
              : 'Popular models from HuggingFace, sized for your hardware.'}
          </p>
        </div>

        <form className={styles.search} onSubmit={submitSearch}>
          <Search size={15} className={styles.searchIcon} />
          <input
            className={styles.searchInput}
            placeholder="Search all of HuggingFace…"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search models"
          />
          <Button type="submit" size="sm" disabled={loading}>
            Search
          </Button>
        </form>
      </header>

      {page?.notice && (
        <div className={styles.notice} role="status">
          <AlertTriangle size={15} />
          <span>{page.notice}</span>
        </div>
      )}

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={15} />
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={() => load(activeSearch || undefined)}>
            <RefreshCw size={13} /> Try again
          </Button>
        </div>
      )}

      <div className={styles.body}>
        <nav className={styles.sidebar} aria-label="Filter by category">
          <button
            className={`${styles.catBtn} ${selected === null ? styles.catActive : ''}`}
            onClick={() => setSelected(null)}
          >
            <span>All</span>
            <span className={styles.count}>{page?.cards.length ?? 0}</span>
          </button>

          {(page?.categories ?? []).map((c: CategoryCount) => (
            <button
              key={c.category}
              className={`${styles.catBtn} ${selected === c.category ? styles.catActive : ''}`}
              onClick={() => setSelected(c.category)}
            >
              <span>{c.label}</span>
              <span className={styles.count}>{c.count}</span>
            </button>
          ))}
        </nav>

        <section className={styles.results}>
          {loading && (
            <div className={styles.centered}>
              <Spinner />
              <p>Reading the model library…</p>
            </div>
          )}

          {!loading && visible.length === 0 && !error && (
            <div className={styles.centered}>
              <p>
                {selected
                  ? 'No models in this category.'
                  : 'No models found. Try a different search.'}
              </p>
            </div>
          )}

          {!loading &&
            visible.map((card) => (
              <Card
                key={card.repoId}
                card={card}
                open={detailCard?.repoId === card.repoId}
                onOpen={() => setDetailCard(card)}
              />
            ))}
        </section>
      </div>

      {detailCard && <DetailDrawer card={detailCard} onClose={() => setDetailCard(null)} />}
    </div>
  );
};

/** Largest size that fits, or the smallest overall when nothing does. */
function pickOffer(card: ModelCard): { offer: Quant | null; fitsHere: boolean } {
  const fitting = card.quantizations.filter((q) => q.fits);

  // Quantization trades quality for size, so the biggest that runs is the most
  // capable that runs.
  const best = fitting.reduce<Quant | null>(
    (acc, q) => (acc === null || q.sizeBytes > acc.sizeBytes ? q : acc),
    null
  );
  if (best) return { offer: best, fitsHere: true };

  const smallest = card.quantizations.reduce<Quant | null>(
    (acc, q) => (acc === null || q.sizeBytes < acc.sizeBytes ? q : acc),
    null
  );
  return { offer: smallest, fitsHere: false };
}

/**
 * Starts a download, confirming first when the size cannot run here.
 *
 * Shared by the card and the drawer so both warn identically.
 */
async function beginDownload(
  card: ModelCard,
  quant: Quant,
  addToast: (kind: 'success' | 'error', msg: string) => void
): Promise<void> {
  if (!quant.fits) {
    const ok = window.confirm(
      `${quant.label} needs more memory than this computer has free.\n\n` +
        `It would download ${formatSize(quant.sizeBytes)} and then most likely fail to ` +
        `load, or run very slowly. Download it anyway?`
    );
    if (!ok) return;
  }

  try {
    await startModelDownload({
      modelId: card.repoId,
      modelName: card.name,
      providerId: 'huggingface',
      quantization: quant.label,
      format: 'GGUF',
      backend: 'llama.cpp (GGUF)',
    });
    addToast('success', `Downloading ${card.name} (${quant.label}) — see progress in Storage`);
  } catch (err) {
    addToast('error', `Could not start the download: ${String(err)}`);
  }
}

interface CardProps {
  card: ModelCard;
  open: boolean;
  onOpen: () => void;
}

function Card({ card, open, onOpen }: CardProps) {
  const [starting, setStarting] = useState(false);
  const { addToast } = useToast();
  const { isDownloading } = useDownloads();
  const fitting = card.quantizations.filter((q) => q.fits);
  const { offer, fitsHere } = pickOffer(card);

  const download = async () => {
    if (!offer) return;
    setStarting(true);
    try {
      await beginDownload(card, offer, addToast);
    } finally {
      setStarting(false);
    }
  };

  return (
    <article className={`${styles.card} ${open ? styles.cardOpen : ''}`}>
      <div className={styles.cardTop}>
        <span className={styles.publisher}>{card.publisher}</span>
        <div className={styles.badges}>
          {/* What this *is* comes first: whether it can run on its own is the
            * thing a non-specialist most needs to know before downloading. */}
          <span
            className={card.kind === 'lora-adapter' ? styles.badgeStrong : styles.badgeKind}
            title={card.kindExplanation}
          >
            {KIND_LABELS[card.kind]}
          </span>
          {card.license && <span className={styles.badge}>{card.license}</span>}
        </div>
      </div>

      <h2 className={styles.cardName}>{card.name}</h2>
      <p className={styles.summary}>{card.summary}</p>

      {/* Spelled out, because "LoRA adapter" alone does not tell someone that
        * downloading it on its own will not give them a working model. */}
      <p className={styles.kindNote}>{card.kindExplanation}</p>

      {card.datasets && card.datasets.length > 0 && (
        <p className={styles.datasets}>Trained on {card.datasets.join(', ')}</p>
      )}

      <div className={styles.cats}>
        {card.categories.map((c) => (
          <span key={c} className={styles.cat}>
            {c.replace(/-/g, ' ')}
          </span>
        ))}
      </div>

      <div className={styles.stats}>
        <span title="Downloads">
          <Download size={12} /> {card.downloadsLabel}
        </span>
        <span title="Likes">
          <Heart size={12} /> {card.likes}
        </span>
        {card.ageLabel && (
          <span title="Last updated">
            <Clock size={12} /> {card.ageLabel}
          </span>
        )}
        <span title="Quantizations available">
          <Layers size={12} /> {card.quantizations.length}
        </span>
      </div>

      {card.baseModel && (
        <p className={styles.base}>
          Based on <code>{card.baseModel}</code>
        </p>
      )}

      <div className={styles.cardFoot}>
        <button className={styles.expand} onClick={onOpen} aria-expanded={open}>
          {`Sizes & LoRA — ${fitting.length} of ${card.quantizations.length} fit`}
        </button>

        {/* An adapter is not a runnable model, so it gets no download button —
          * it is installed from the base model's card instead. */}
        {card.kind !== 'lora-adapter' && offer && (
          <button
            className={fitsHere ? styles.downloadBtn : styles.downloadBtnWarn}
            disabled={starting || isDownloading(card.repoId)}
            onClick={() => void download()}
            title={
              fitsHere
                ? `Download the largest size that fits: ${offer.label}, ${formatSize(
                    offer.sizeBytes
                  )}`
                : `${offer.label} is the smallest build at ${formatSize(
                    offer.sizeBytes
                  )}, but it still needs more memory than this computer has free`
            }
          >
            <Download size={13} />
            {isDownloading(card.repoId)
              ? 'Downloading…'
              : starting
              ? 'Starting…'
              : // "anyway" is the honest word: the card already says nothing
                // fits, so the button should not pretend otherwise.
                `Download ${offer.label}${fitsHere ? '' : ' anyway'}`}
          </button>
        )}
      </div>
    </article>
  );
}

interface DrawerProps {
  card: ModelCard;
  onClose: () => void;
}

/**
 * Full detail for one model, docked beside the list.
 *
 * Nothing here changes the grid's layout, so opening details cannot disturb
 * any other card.
 */
function DetailDrawer({ card, onClose }: DrawerProps) {
  const [adapters, setAdapters] = useState<AdapterPage | null>(null);
  const [loadingAdapters, setLoadingAdapters] = useState(true);
  const [installing, setInstalling] = useState<string | null>(null);
  const [installed, setInstalled] = useState<Set<string>>(new Set());
  const [installError, setInstallError] = useState<string | null>(null);
  const [starting, setStarting] = useState<string | null>(null);
  const { addToast } = useToast();
  const { isDownloading } = useDownloads();

  // Escape closes it, as with any dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Looked up per model, not with the listing: one request per card would mean
  // a hundred extra calls per sweep and would hit the rate limit immediately.
  useEffect(() => {
    let cancelled = false;
    setAdapters(null);
    setLoadingAdapters(true);

    findModelAdapters(card.baseModel || card.repoId)
      .then((found) => {
        if (!cancelled) setAdapters(found);
      })
      .catch((err) => {
        if (!cancelled) setAdapters({ adapters: [], readyCount: 0, notice: String(err) });
      })
      .finally(() => {
        if (!cancelled) setLoadingAdapters(false);
      });

    return () => {
      cancelled = true;
    };
  }, [card.repoId, card.baseModel]);

  const install = async (adapterRepoId: string) => {
    setInstalling(adapterRepoId);
    setInstallError(null);
    try {
      // Adapters install beside the base model, so the base model's id is what
      // identifies where they belong.
      await downloadAdapter('huggingface', card.baseModel || card.repoId, adapterRepoId);
      setInstalled((prev) => new Set(prev).add(adapterRepoId));
    } catch (err) {
      // The backend explains refusals in plain language — pass it through.
      setInstallError(String(err));
    } finally {
      setInstalling(null);
    }
  };

  const download = async (q: Quant) => {
    setStarting(q.label);
    try {
      await beginDownload(card, q, addToast);
    } finally {
      setStarting(null);
    }
  };

  return (
    <>
      <div className={styles.scrim} onClick={onClose} aria-hidden="true" />

      <aside
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-label={`${card.name} details`}
      >
        <header className={styles.drawerHead}>
          <div>
            <span className={styles.publisher}>{card.publisher}</span>
            <h2 className={styles.drawerTitle}>{card.name}</h2>
          </div>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close details">
            <X size={16} />
          </button>
        </header>

        <div className={styles.drawerBody}>
          <p className={styles.summary}>{card.summary}</p>

          <section>
            <h3 className={styles.drawerSection}>Sizes</h3>
            {/* Headed, because "5.2 GB" on its own does not say whether that is
              * the download, the memory needed, or both. */}
            <table className={styles.quants}>
              <thead>
                <tr className={styles.quantHead}>
                  <th scope="col">Version</th>
                  <th scope="col">Download</th>
                  <th scope="col">Quality</th>
                  <th scope="col" className={styles.qFit}>
                    Runs here
                  </th>
                  <th scope="col" aria-label="Get" />
                </tr>
              </thead>
              <tbody>
                {card.quantizations.map((q) => (
                  <tr key={q.label} className={q.fits ? styles.fits : styles.tooBig}>
                    <td className={styles.qLabel}>{q.label}</td>
                    <td>{formatSize(q.sizeBytes)}</td>
                    <td className={styles.qNote}>{q.qualityNote}</td>
                    <td className={styles.qFit}>{q.fits ? 'fits' : 'too large'}</td>
                    <td className={styles.qAction}>
                      {card.kind !== 'lora-adapter' && (
                        <button
                          className={styles.rowBtn}
                          disabled={
                            starting !== null || isDownloading(card.repoId, q.label)
                          }
                          onClick={() => void download(q)}
                          aria-label={`Download ${card.name} ${q.label}`}
                        >
                          {isDownloading(card.repoId, q.label) ? '…' : <Download size={12} />}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <p className={styles.sizeNote}>
              “Download” is the file size — roughly what the weights take in memory too.
              Running also needs spare memory for the conversation, which “Runs here”
              already allows for.
            </p>
          </section>

          <section>
            <h3 className={styles.drawerSection}>LoRA adapters</h3>

            {loadingAdapters && <p className={styles.adapterNotice}>Looking for adapters…</p>}
            {adapters?.notice && <p className={styles.adapterNotice}>{adapters.notice}</p>}
            {installError && <p className={styles.adapterError}>{installError}</p>}

            {adapters && adapters.adapters.length > 0 && (
              <>
                <table className={styles.adapterTable}>
                  <thead>
                    <tr className={styles.quantHead}>
                      <th scope="col">Adapter</th>
                      <th scope="col">By</th>
                      <th scope="col" className={styles.qFit}>
                        Usable
                      </th>
                      <th scope="col" aria-label="Get" />
                    </tr>
                  </thead>
                  <tbody>
                    {adapters.adapters.map((a) => (
                      <AdapterRow
                        key={a.repoId}
                        adapter={a}
                        installing={installing === a.repoId}
                        installed={installed.has(a.repoId)}
                        onInstall={() => void install(a.repoId)}
                      />
                    ))}
                  </tbody>
                </table>
                <p className={styles.sizeNote}>
                  Select an adapter's name to see what it changes about the model.
                </p>
              </>
            )}
          </section>
        </div>
      </aside>
    </>
  );
}

interface AdapterRowProps {
  adapter: { repoId: string; name: string; author: string; focus: string; ggufReady: boolean };
  installing: boolean;
  installed: boolean;
  onInstall: () => void;
}

/**
 * One adapter, with its name acting as a disclosure for what it changes.
 *
 * The detail is fetched on selection rather than with the list: one lookup per
 * adapter would hit HuggingFace's rate limit immediately.
 */
function AdapterRow({ adapter: a, installing, installed, onInstall }: AdapterRowProps) {
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState<AdapterDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || details || loading) return;

    setLoading(true);
    setFailed(null);
    try {
      setDetails(await getAdapterDetails(a.repoId));
    } catch (err) {
      setFailed(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <tr className={a.ggufReady ? styles.fits : styles.tooBig}>
        <td>
          <button
            className={styles.adapterNameBtn}
            onClick={() => void toggle()}
            aria-expanded={open}
            title="What does this change?"
          >
            {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
            <span className={styles.adapterFocus}>{a.focus}</span>
          </button>
        </td>
        <td className={styles.adapterAuthor}>{a.author}</td>
        <td className={styles.qFit}>
          {a.ggufReady ? (
            'ready'
          ) : (
            <span
              className={styles.needsConversion}
              title="PEFT safetensors. llama.cpp needs GGUF, and Sarathi cannot convert yet."
            >
              needs conversion
            </span>
          )}
        </td>
        <td className={styles.qAction}>
          {/* No button when it cannot load: a download here would fetch a file
            * llama.cpp cannot use, which is worse than saying why not. */}
          {a.ggufReady && (
            <button
              className={styles.getBtn}
              disabled={installing || installed}
              onClick={onInstall}
            >
              {installed ? 'installed' : installing ? 'getting…' : 'Get'}
            </button>
          )}
        </td>
      </tr>

      {open && (
        <tr>
          <td colSpan={4} className={styles.adapterDetailCell}>
            {loading && <p className={styles.adapterNotice}>Reading its page…</p>}
            {failed && <p className={styles.adapterError}>{failed}</p>}

            {details && (
              <div className={styles.adapterDetail}>
                {details.effects.length > 0 && (
                  <ul className={styles.effects}>
                    {details.effects.map((e) => (
                      <li key={e.skill} className={styles.effect}>
                        <span className={styles.effectSkill}>{e.skill}</span>
                        {/* A guess read off the repository name must never be
                          * presented with the same weight as the author's own
                          * tags, so each line says which it is. */}
                        <span
                          className={
                            e.confidence === 'stated' ? styles.stated : styles.suggested
                          }
                          title={
                            e.confidence === 'stated'
                              ? "Declared by the adapter's author"
                              : 'Guessed from the name — the author did not say'
                          }
                        >
                          {e.confidence === 'stated' ? 'author says' : 'from its name'}
                        </span>
                        <span className={styles.effectText}>{e.description}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {details.datasets.length > 0 && (
                  <p className={styles.detailLine}>
                    <strong>Trained on:</strong> {details.datasets.join(', ')}
                  </p>
                )}

                <p className={styles.detailLine}>
                  {details.sizeBytes > 0 && <>{formatSize(details.sizeBytes)} · </>}
                  {details.downloads.toLocaleString()} downloads
                  {details.license && <> · {details.license}</>}
                </p>

                {details.notice && <p className={styles.adapterNotice}>{details.notice}</p>}
                {details.blockedReason && (
                  <p className={styles.adapterNotice}>{details.blockedReason}</p>
                )}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
