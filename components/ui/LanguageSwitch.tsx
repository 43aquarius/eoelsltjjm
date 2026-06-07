'use client';

import { Globe } from 'lucide-react';
import { useLanguage } from '@/lib/LanguageContext';

export function LanguageSwitch() {
  const { language, setLanguage } = useLanguage();

  return (
    <button
      onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
      className="flex items-center gap-1.5 px-2 py-1 rounded-lg hover:bg-[rgba(123,61,44,0.06)] transition-colors"
      title={language === 'zh' ? 'Switch to English' : '切换到中文'}
    >
      <Globe className="w-3.5 h-3.5 text-[#a89588]" />
      <span className="text-[11px] text-[#8b6e62] font-medium">
        {language === 'zh' ? '中文' : 'EN'}
      </span>
    </button>
  );
}
