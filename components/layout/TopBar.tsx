'use client';

import { useEffect, useState } from 'react';
import { Command, MapPin, Clock, HelpCircle } from 'lucide-react';
import { useStore } from '@/lib/store';
import { useTranslation } from '@/lib/LanguageContext';
import { LanguageSwitch } from '@/components/ui/LanguageSwitch';

export function TopBar() {
  const { isDemoMode, simulationTime, advanceSimulation, setCommandMenuOpen } = useStore();
  const t = useTranslation();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  return (
    <header
      className="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-md border-b border-[rgba(123,61,44,0.08)] z-50"
      style={{ height: 'var(--topbar-height)' }}
    >
      <div className="h-full px-6 flex items-center justify-between">
        {/* Left */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setCommandMenuOpen(true)}
            className="flex items-center gap-3 px-4 py-2 rounded-xl bg-[rgba(123,61,44,0.04)] hover:bg-[rgba(123,61,44,0.08)] transition-colors"
          >
            <Command className="w-5 h-5 text-[#b8a89c]" />
            <span className="text-sm text-[#9a7e72] hidden sm:inline">{t.footer.searchPlaceholder}</span>
            <kbd className="hidden sm:inline px-2 py-0.5 text-[11px] text-[#d4c8be] bg-[rgba(123,61,44,0.05)] rounded-lg border border-[rgba(123,61,44,0.08)]">
              ⌘K
            </kbd>
          </button>
        </div>

        {/* Right */}
        <div className="flex items-center gap-3">
          {isDemoMode && (
            <>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[rgba(231,135,69,0.08)] border border-[rgba(231,135,69,0.15)]">
                <div className="w-2 h-2 rounded-full bg-[#e78745] animate-pulse" />
                <span className="text-xs font-medium text-[#e78745] hidden sm:inline">
                  {t.footer.demoMode}
                </span>
              </div>

              <button
                onClick={advanceSimulation}
                className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-[rgba(123,61,44,0.05)] transition-colors"
                title="前进15分钟"
              >
                <Clock className="w-4 h-4 text-[#9a7e72]" />
                <span className="text-xs text-[#7b3d2c] font-mono hidden md:inline">
                  {formatTime(simulationTime)}
                </span>
              </button>
            </>
          )}

          <div className="hidden md:flex items-center gap-1.5 text-[#9a7e72]">
            <MapPin className="w-4 h-4" />
            <span className="text-xs">{t.footer.location}</span>
          </div>

          <LanguageSwitch />

          <button className="p-2 rounded-xl hover:bg-[rgba(123,61,44,0.05)] transition-colors">
            <HelpCircle className="w-5 h-5 text-[#9a7e72]" />
          </button>
        </div>
      </div>
    </header>
  );
}
