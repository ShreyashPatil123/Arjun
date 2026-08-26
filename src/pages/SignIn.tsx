import React, { useCallback, useEffect, useState } from 'react';
import { KeyRound, ShieldCheck } from 'lucide-react';
import {
  governanceService,
  ROLE_LABELS,
  type AuthenticationStatus,
  type Session,
  type User,
} from '../services/governance.service';
import styles from './SignIn.module.css';

interface SignInProps {
  onSignedIn: (session: Session) => void;
}

/** Mirrors the policy the backend enforces, so the form can say so before submitting. */
const MIN_PASSWORD_LENGTH = 12;

/**
 * Sign-in, and the first-run setup that precedes it.
 *
 * On a fresh deployment nobody has a password, so the first thing that can
 * happen is an administrator choosing one — PS step 7. Only after that does the
 * ordinary sign-in form appear.
 *
 * The account list is shown openly. On a machine inside a plant, who has an
 * account is not the secret; the passwords are. Hiding the list would only make
 * people mistype their own username.
 */
export const SignIn = ({ onSignedIn }: SignInProps) => {
  const [status, setStatus] = useState<AuthenticationStatus | null>(null);
  const [accounts, setAccounts] = useState<User[]>([]);
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [nextStatus, list] = await Promise.all([
        governanceService.authenticationStatus(),
        governanceService.listAccounts(),
      ]);
      setStatus(nextStatus);
      setAccounts(list);
      setUserId(prev => {
        if (prev) return prev;
        // On a fresh deployment only an administrator can go first, so preselect
        // one rather than letting someone pick an account that will be refused.
        const candidates =
          nextStatus === 'awaitingFirstAdministrator'
            ? list.filter(u => u.roles.includes('administrator'))
            : list;
        return candidates[0]?.id ?? '';
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const isSetup = status === 'awaitingFirstAdministrator';
  const eligible = isSetup
    ? accounts.filter(u => u.roles.includes('administrator'))
    : accounts;

  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH;
  const mismatched = isSetup && confirmation.length > 0 && password !== confirmation;
  const canSubmit =
    !busy &&
    userId !== '' &&
    password.length >= MIN_PASSWORD_LENGTH &&
    (!isSetup || password === confirmation);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;

    setBusy(true);
    setError(null);
    try {
      if (isSetup) {
        await governanceService.setInitialAdministratorPassword(userId, password);
        // Straight into a real session, so setup does not end on a dead end.
        onSignedIn(await governanceService.signIn(userId, password));
      } else {
        onSignedIn(await governanceService.signIn(userId, password));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPassword('');
      setConfirmation('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={styles.page}>
      <form className={styles.card} onSubmit={submit}>
        <div className={styles.brand}>ARJUN</div>

        <h1 className={styles.heading}>
          {isSetup ? 'Set the administrator password' : 'Sign in'}
        </h1>

        <p className={styles.lede}>
          {isSetup
            ? 'Nobody has a password on this machine yet. Choose one for an administrator account; everyone else is set up from Settings afterwards.'
            : 'ARJUN records who did what, so it needs to know who you are.'}
        </p>

        <label className={styles.field}>
          <span className={styles.label}>Account</span>
          <select
            className={styles.select}
            value={userId}
            onChange={e => setUserId(e.target.value)}
          >
            {eligible.map(account => (
              <option key={account.id} value={account.id}>
                {account.displayName} — {account.roles.map(r => ROLE_LABELS[r] ?? r).join(', ')}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Password</span>
          <input
            className={styles.input}
            type="password"
            value={password}
            autoComplete={isSetup ? 'new-password' : 'current-password'}
            onChange={e => setPassword(e.target.value)}
          />
          {isSetup && (
            <span className={tooShort ? styles.hintWarn : styles.hint}>
              At least {MIN_PASSWORD_LENGTH} characters. A phrase of several words is easier
              to type and harder to guess than a short password with symbols in it.
            </span>
          )}
        </label>

        {isSetup && (
          <label className={styles.field}>
            <span className={styles.label}>Confirm password</span>
            <input
              className={styles.input}
              type="password"
              value={confirmation}
              autoComplete="new-password"
              onChange={e => setConfirmation(e.target.value)}
            />
            {mismatched && <span className={styles.hintWarn}>The two do not match.</span>}
          </label>
        )}

        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}

        <button className={styles.submit} type="submit" disabled={!canSubmit}>
          <KeyRound size={16} />
          {busy ? 'Working…' : isSetup ? 'Set password and sign in' : 'Sign in'}
        </button>

        {isSetup && (
          <p className={styles.recovery}>
            There is no password recovery. An air-gapped machine has nowhere to send a reset
            to, and a recovery question is only a weaker second password. Record this one
            wherever your site keeps its break-glass credentials.
          </p>
        )}

        <p className={styles.footnote}>
          <ShieldCheck size={13} />
          <span>Checked on this machine. Nothing about this sign-in leaves it.</span>
        </p>
      </form>
    </div>
  );
};
