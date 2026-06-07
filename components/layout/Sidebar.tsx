'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  FileText,
  Users,
  FileCheck,
  Store,
  BarChart3,
  Zap,
} from 'lucide-react';
import { useTranslation } from '@/lib/LanguageContext';

const navItems = [
  { href: '/', labelKey: 'home' as const, icon: Home },
  { href: '/create', labelKey: 'createDemand' as const, icon: FileText },
  { href: '/matches', labelKey: 'matchResults' as const, icon: Users },
  { href: '/contract/demo', labelKey: 'contract' as const, icon: FileCheck },
  { href: '/merchant', labelKey: 'merchant' as const, icon: Store },
  { href: '/dashboard', labelKey: 'dashboard' as const, icon: BarChart3 },
];

export function Sidebar() {
  const pathname = usePathname();
  const t = useTranslation();

  return (
    <aside
      className="fixed left-0 bottom-0 bg-white border-r border-[rgba(123,61,44,0.08)] flex flex-col shadow-sm"
      style={{ top: 'var(--topbar-height)', width: 'var(--sidebar-width)' }}
    >
      {/* Logo */}
      <div className="p-5 border-b border-[rgba(123,61,44,0.06)]">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#7b3d2c] flex items-center justify-center shadow-md">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-[#7b3d2c]">LocalContract</h1>
            <p className="text-[11px] text-[#b8a89c] mt-0.5">需求合约匹配系统</p>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== '/' && pathname.startsWith(item.href));
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`
                  flex items-center gap-3 px-4 py-3 rounded-xl
                  text-sm font-medium transition-all duration-250
                  ${
                    isActive
                      ? 'bg-[rgba(123,61,44,0.08)] text-[#7b3d2c]'
                      : 'text-[#9a7e72] hover:bg-[rgba(123,61,44,0.04)] hover:text-[#7b3d2c]'
                  }
                `}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                <span className="truncate">{t.nav[item.labelKey]}</span>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-[rgba(123,61,44,0.06)]">
        <div className="flex items-center gap-3 p-3 rounded-xl bg-[rgba(123,61,44,0.03)]">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#7b3d2c] to-[#5c2d20] flex items-center justify-center flex-shrink-0">
            <span className="text-sm font-semibold text-white">D</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-[#7b3d2c] truncate">演示用户</p>
            <p className="text-[11px] text-[#b8a89c] truncate">user@demo.com</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
