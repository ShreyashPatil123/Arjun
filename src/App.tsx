import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppStateProvider } from './contexts/AppStateContext';
import { ConfigProvider } from './contexts/ConfigContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './contexts/ToastContext';
import { AppShell } from './components/layout';
import { Welcome } from './pages/Welcome';
import { Settings } from './pages/Settings';
import { SystemInfo } from './pages/SystemInfo';
import { Models } from './pages/Models';
import { Chat } from './pages/Chat';
import { LoRA } from './pages/LoRA';

function App() {
  return (
    <AppStateProvider>
      <ConfigProvider>
        <ThemeProvider>
          <ToastProvider>
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<AppShell />}>
                  <Route index element={<Welcome />} />
                  <Route path="settings" element={<Settings />} />
                  <Route path="system" element={<SystemInfo />} />
                  <Route path="models" element={<Models />} />
                  <Route path="chat" element={<Chat />} />
                  <Route path="lora" element={<LoRA />} />
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