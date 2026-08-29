import React, { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppStateProvider } from './contexts/AppStateContext';
import { ConfigProvider } from './contexts/ConfigContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './contexts/ToastContext';
import { AppShell } from './components/layout';
import { Workbench } from './pages/Workbench';
import { Tasks } from './pages/Tasks';
import { Conversations } from './pages/Conversations';
import { Placeholder } from './pages/Placeholder';
import { AuditNetwork } from './pages/AuditNetwork';
import { SignIn } from './pages/SignIn';
import { governanceService, type Session } from './services/governance.service';
import { Settings } from './pages/Settings';
import { SystemInfo } from './pages/SystemInfo';
import { Browse } from './pages/Browse';
import { Storage } from './pages/Storage';
import { Health } from './pages/Health';
import { ModelHealth } from './pages/ModelHealth';
import { Approvals } from './pages/Approvals';
import { Demo } from './pages/Demo';
import { SIHDashboard } from './pages/SIHDashboard';

/**
 * Everything below the shell assumes a signed-in user: the policy gateway
 * decides from their roles, and the audit log records their id. So sign-in is a
 * gate around the whole application rather than a page inside it — there is no
 * meaningful state to show somebody ARJUN cannot yet name.
 */
function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [checked, setChecked] = useState(false);

  const restore = useCallback(async () => {
    try {
      setSession(await governanceService.currentSession());
    } catch {
      // Backend unavailable (a browser-only dev run). Falls through to the
      // sign-in screen, which will surface the real error when it tries.
    } finally {
      setChecked(true);
    }
  }, []);

  useEffect(() => {
    void restore();
  }, [restore]);

  // Nothing is drawn until we know, so the shell never flashes before the gate.
  if (!checked) return null;

  if (!session) {
    return (
      <ThemeProvider>
        <ToastProvider>
          <SignIn onSignedIn={setSession} />
        </ToastProvider>
      </ThemeProvider>
    );
  }

  return (
    <AppStateProvider>
      <ConfigProvider>
        <ThemeProvider>
          <ToastProvider>
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<AppShell />}>
                  <Route index element={<Workbench />} />

                  <Route path="tasks" element={<Tasks />} />
                <Route path="conversations" element={<Conversations />} />

                  {/* Surfaces PS 26117 requires that do not exist yet. Routed and
                    * named now so the shell is complete and the gaps are visible. */}
                  <Route
                    path="knowledge"
                    element={
                      <Placeholder
                        title="Knowledge"
                        purpose="Connected collections of manuals, SOPs, inspection reports and past correspondence — indexed on this machine, from a local folder or an internal network share."
                        phase="Phase 4 — knowledge service and connectors"
                      />
                    }
                  />
                  <Route path="approvals" element={<Approvals />} />
                  <Route path="audit" element={<AuditNetwork />} />
                  <Route path="health" element={<Health />} />
                  <Route path="model-health" element={<ModelHealth />} />
                  <Route path="demo" element={<Demo />} />
                  <Route path="sih" element={<SIHDashboard />} />

                  {/* Carried over from Sarathi. */}
                  <Route path="models" element={<Storage />} />
                  <Route path="browse" element={<Browse />} />
                  <Route path="system" element={<SystemInfo />} />
                  <Route path="settings" element={<Settings />} />
                </Route>
              </Routes>
            </BrowserRouter>
          </ToastProvider>
        </ThemeProvider>
      </ConfigProvider>
    </AppStateProvider>
  );
}

export default App;
