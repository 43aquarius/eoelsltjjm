import type { Metadata } from 'next';
import './globals.css';
import { AppShell } from '@/components/layout/AppShell';
import { CommandMenu } from '@/components/features/CommandMenu';
import { LanguageProvider } from '@/lib/LanguageContext';

export const metadata: Metadata = {
  title: 'LocalContract - 需求合约匹配',
  description: '将本地服务从推荐转变为承诺',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh">
      <body className="antialiased">
        <LanguageProvider>
          <AppShell>
            {children}
            <CommandMenu />
          </AppShell>
        </LanguageProvider>
      </body>
    </html>
  );
}
