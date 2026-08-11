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
  Sparkles,
  ArrowDownWideNarrow,
} from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import {
  browseModelCards,
  findModelAdapters,
  formatSize,
  type AdapterPage,
  type CatalogPage,
} from '../services/catalog.service';
import {
  downloadAdapter,
  getAdapterDetails,
  type AdapterDetails,
} from '../services/adapters.service';
import { startModelDownload } from '../services/download.service';
import { useDownloads } from '../hooks/useDownloads';
import { useToast } from '../hooks/useToast';
import {
  CATEGORY_LABELS,
  KIND_LABELS,
  MODEL_CATEGORIES,
  type ModelCard,
  type ModelCategory,
  type ModelKind,
} from '../types/ai';
import styles from './Browse.module.css';

/** A downloadable size, as the card reports it. */
type Quant = ModelCard['quantizations'][number];

/**
 * The one filter in effect. Exactly one, never a combination.
 *
 * The sidebar reads as tabs, and tabs are alternatives rather than a stack of
 * constraints. Letting a category and a type both apply meant "Coding" showed
 * *coding models of the type already selected* — one card, under a heading that
 * had said 26 a moment earlier — and which models a tab held depended on the
 * order the user had clicked things. Each tab now answers the same question of
 * the whole catalog, so it shows the same models however it was reached.
 */
type Selection =
  | { tab: 'all' }
  | { tab: 'recommended' }
  | { tab: 'category'; value: ModelCategory }
  | { tab: 'type'; value: ModelKind };

/** How the grid is ordered. */
type SortKey = 'best' | 'worst' | 'smallest' | 'largest' | 'popular' | 'newest';

const SORT_LABELS: Record<SortKey, string> = {
  best: 'Best first',
  worst: 'Worst first',
  smallest: 'Smallest download',
  largest: 'Largest download',
  popular: 'Most downloaded',
  newest: 'Recently updated',
};

/**
 * How good a bet this model is, as a single number.
 *
 * Ordering by downloads alone puts whatever went viral on top, including
 * 1-bit experiments and models too large to run here. This ranks on what makes
 * a model useful *to this user*: Sarathi vouches for it, it runs on their
 * hardware, it will not spray reasoning tokens into their editor — and only
 * then how many people have used it, as a tie-break between otherwise equal
 * candidates.
 *
 * Popularity is compressed with a log so a model with ten times the downloads
 * counts as somewhat better rather than ten times better; raw counts span six
 * orders of magnitude and would otherwise swamp every other term.
 */
function quality(card: ModelCard): number {
  const runsHere = card.quantizations.some((q) => q.fits);
  return (
    (card.recommended ? 1000 : 0) +
    (runsHere ? 500 : 0) +
    (card.emitsReasoning ? -250 : 0) +
    Math.log10(card.downloads + 1) * 20 +
    Math.log10(card.likes + 1) * 5
  );
}

/** The size actually downloaded if the user presses the button on this card. */
function offerSize(card: ModelCard): number {
  return pickOffer(card).offer?.sizeBytes ?? Number.MAX_SAFE_INTEGER;
}

function sortCards(cards: ModelCard[], key: SortKey): ModelCard[] {
  const sorted = [...cards];
  switch (key) {
    case 'best':
      return sorted.sort((a, b) => quality(b) - quality(a));
    case 'worst':
      return sorted.sort((a, b) => quality(a) - quality(b));
    case 'smallest':
      return sorted.sort((a, b) => offerSize(a) - offerSize(b));
    case 'largest':
      return sorted.sort((a, b) => offerSize(b) - offerSize(a));
    case 'popular':
      return sorted.sort((a, b) => b.downloads - a.downloads);
    case 'newest':
      // `lastModified` is ISO 8601, which sorts correctly as text.
      return sorted.sort((a, b) => (b.lastModified ?? '').localeCompare(a.lastModified ?? ''));
  }
}

/** Whether a card belongs in the given tab. */
function inSelection(card: ModelCard, selection: Selection): boolean {
  switch (selection.tab) {
    case 'all':
      return true;
    case 'recommended':
      return card.recommended;
    case 'category':
      return card.categories.includes(selection.value);
    case 'type':
      return card.kind === selection.value;
  }
}

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
  const [selection, setSelection] = useState<Selection>({ tab: 'all' });
  const [sort, setSort] = useState<SortKey>('best');
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

  /**
   * The full catalog for this search, and the single source every tab and
   * every count is derived from.
   *
   * Nothing narrows this. Selecting a tab changes which cards are *shown*, not
   * what the next tab is computed from — otherwise opening one tab silently
   * shrinks the others, and a tab's contents depend on the route taken to it.
   */
  const all = useMemo(() => page?.cards ?? [], [page]);

  const visible = useMemo(
    () => sortCards(all.filter((c) => inSelection(c, selection)), sort),
    [all, selection, sort]
  );

  /**
   * Counts, all taken from the whole catalog.
   *
   * Because exactly one tab applies at a time, each of these is by construction
   * the length of what clicking it shows — the count and the grid cannot drift
   * apart.
   */
  const categoryCounts = useMemo(() => {
    const counts = new Map<ModelCategory, number>();
    for (const card of all) {
      for (const c of card.categories) counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return counts;
  }, [all]);

  const kindCounts = useMemo(() => {
    const counts = new Map<ModelKind, number>();
    for (const card of all) counts.set(card.kind, (counts.get(card.kind) ?? 0) + 1);
    return counts;
  }, [all]);

  const recommendedCount = useMemo(() => all.filter((c) => c.recommended).length, [all]);

  /**
   * Recommended cards shown as their own section, ahead of everything else.
   *
   * Only on the "All" tab: on a narrower tab, splitting the grid in two
   * obscures how many results that tab actually holds.
   */
  const showRecommendedSection =
    selection.tab === 'all' && !activeSearch && recommendedCount > 0;

  const [featured, remainder] = useMemo(() => {
    if (!showRecommendedSection) return [[], visible] as const;
    return [
      visible.filter((c) => c.recommended),
      visible.filter((c) => !c.recommended),
    ] as const;
  }, [showRecommendedSection, visible]);

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    load(searchTerm.trim() || undefined);
    setSelection({ tab: 'all' });
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

        <label className={styles.sort}>
          <ArrowDownWideNarrow size={14} aria-hidden />
          <span className={styles.sortLabel}>Sort</span>
          <select
            className={styles.sortSelect}
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            aria-label="Sort models"
          >
            {(Object.keys(SORT_LABELS) as SortKey[]).map((k) => (
              <option key={k} value={k}>
                {SORT_LABELS[k]}
              </option>
            ))}
          </select>
        </label>
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
        <nav className={styles.sidebar} aria-label="Browse models by">
          {/* Every entry is an alternative view of the same catalog, so each one
              replaces the selection rather than adding to it, and each count is
              taken from the whole catalog. A tab therefore holds the same models
              no matter which tab preceded it. */}
          {recommendedCount > 0 && (
            <button
              className={`${styles.catBtn} ${styles.recommendBtn} ${
                selection.tab === 'recommended' ? styles.catActive : ''
              }`}
              onClick={() => setSelection({ tab: 'recommended' })}
              aria-current={selection.tab === 'recommended'}
              title="Models Sarathi vouches for: dependable publishers, proven conversions, no reasoning tokens, and they run on this machine."
            >
              <span>
                <Sparkles size={13} className={styles.recommendIcon} /> Sarathi Recommended
              </span>
              <span className={styles.count}>{recommendedCount}</span>
            </button>
          )}

          <button
            className={`${styles.catBtn} ${selection.tab === 'all' ? styles.catActive : ''}`}
            onClick={() => setSelection({ tab: 'all' })}
            aria-current={selection.tab === 'all'}
          >
            <span>All</span>
            <span className={styles.count}>{all.length}</span>
          </button>

          {MODEL_CATEGORIES.filter((c) => (categoryCounts.get(c) ?? 0) > 0).map((c) => (
            <button
              key={c}
              className={`${styles.catBtn} ${
                selection.tab === 'category' && selection.value === c ? styles.catActive : ''
              }`}
              onClick={() => setSelection({ tab: 'category', value: c })}
              aria-current={selection.tab === 'category' && selection.value === c}
            >
              <span>{CATEGORY_LABELS[c]}</span>
              <span className={styles.count}>{categoryCounts.get(c)}</span>
            </button>
          ))}

          {/* Type asks a different question from category — "what is this entry"
              rather than "what is it good at" — but it is still one of the
              alternatives, not a second constraint layered on top. */}
          {kindCounts.size > 1 && (
            <>
              <h2 className={styles.sidebarHeading}>Type</h2>
              {(Object.keys(KIND_LABELS) as ModelKind[])
                .filter((k) => (kindCounts.get(k) ?? 0) > 0)
                .map((k) => (
                  <button
                    key={k}
                    className={`${styles.catBtn} ${
                      selection.tab === 'type' && selection.value === k ? styles.catActive : ''
                    }`}
                    onClick={() => setSelection({ tab: 'type', value: k })}
                    aria-current={selection.tab === 'type' && selection.value === k}
                  >
                    <span>{KIND_LABELS[k]}</span>
                    <span className={styles.count}>{kindCounts.get(k)}</span>
                  </button>
                ))}
            </>
          )}
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
                {selection.tab === 'all'
                  ? 'No models found. Try a different search.'
                  : 'Nothing in this category yet.'}
              </p>
            </div>
          )}

          {!loading && featured.length > 0 && (
            <>
              <div className={styles.sectionBar}>
                <h2 className={styles.sectionTitle}>
                  <Sparkles size={14} className={styles.recommendIcon} /> Sarathi Recommended
                </h2>
                <p className={styles.sectionNote}>
                  Dependable publishers, conversions that have been proven in use, no reasoning
                  tokens in the output — and they run on this machine.
                </p>
              </div>
              {featured.map((card) => (
                <Card
                  key={card.repoId}
                  card={card}
                  open={detailCard?.repoId === card.repoId}
                  onOpen={() => setDetailCard(card)}
                />
              ))}
            </>
          )}

          {!loading && featured.length > 0 && remainder.length > 0 && (
            <div className={styles.sectionBar}>
              <h2 className={styles.sectionTitle}>Everything else</h2>
              <p className={styles.sectionNote}>
                The rest of the library, newest and most downloaded first. Worth reading the size
                and quality notes before picking one.
              </p>
            </div>
          )}

          {!loading &&
            remainder.map((card) => (
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
                    <td className={styles.qNote}>
                      {q.lowQuality && (
                        <AlertTriangle
                          size={11}
                          className={styles.qWarnIcon}
                          aria-hidden="true"
                        />
                      )}
                      {q.qualityNote}
                    </td>
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
              title="PEFT safetensors. Sarathi converts it to GGUF after downloading, which needs the base model installed and a supported model family."
            >
              converts on install
            </span>
          )}
        </td>
        <td className={styles.qAction}>
          {/* Both kinds are installable now. A PEFT adapter is downloaded and
            * converted in one step; if the conversion cannot be done the whole
            * install is rolled back, so the button never leaves a file behind
            * that llama.cpp would refuse. */}
          <button
            className={styles.getBtn}
            disabled={installing || installed}
            onClick={onInstall}
            title={
              a.ggufReady
                ? 'Download this adapter'
                : 'Download and convert this adapter to GGUF'
            }
          >
            {installed
              ? 'installed'
              : installing
                ? a.ggufReady
                  ? 'getting…'
                  : 'converting…'
                : 'Get'}
          </button>
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
                          * tags, so each line says which it is.
                          *
                          * The label says "author tagged", not "author says":
                          * the author supplied a tag, and the sentence beside it
                          * is Sarathi explaining what that tag means. Wording it
                          * as speech put our words in their mouth, which also
                          * made every adapter in a category look identical. */}
                        <span
                          className={
                            e.confidence === 'stated' ? styles.stated : styles.suggested
                          }
                          title={
                            e.confidence === 'stated'
                              ? 'The author tagged this adapter with this skill. The description is Sarathi explaining what that tag means.'
                              : 'Guessed from the repository name — the author did not tag it'
                          }
                        >
                          {e.confidence === 'stated' ? 'author tagged' : 'from its name'}
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
