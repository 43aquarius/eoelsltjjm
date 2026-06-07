'use client';

import { type ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-[#fef6f1]">
      <TopBar />
      <Sidebar />
      <main
        className="min-h-screen"
        style={{ marginLeft: 'var(--sidebar-width)', paddingTop: 'var(--topbar-height)' }}
      >
        <div className="p-6 lg:p-8 max-w-7xl mx-auto">{children}</div>
      </main>
    </div>
  );
}
