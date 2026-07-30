import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button, Card } from './';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Unhandled React Rendering Error:', error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '32px', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <Card>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: '16px', padding: '24px', maxWidth: '480px' }}>
              <AlertTriangle size={48} color="var(--warning)" />
              <h2 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>Component Rendering Exception</h2>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {this.state.error?.message || 'An unexpected rendering error occurred.'}
              </p>
              <Button variant="primary" icon={<RefreshCw size={16} />} onClick={this.handleReset}>
                Reload Interface
              </Button>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
