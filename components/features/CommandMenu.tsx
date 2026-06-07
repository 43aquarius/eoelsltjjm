'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Home,
  FileText,
  Users,
  FileCheck,
  Store,
  BarChart3,
  Search,
  ArrowRight,
  Zap,
  Play,
} from 'lucide-react';
import { useStore } from '@/lib/store';

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  shortcut?: string;
  action: () => void;
  category: string;
}

const navigationItems = [
  { href: '/', label: 'Home', description: 'Go to homepage', icon: Home },
  { href: '/create', label: 'Create Demand', description: 'Create a new demand', icon: FileText },
  { href: '/matches', label: 'Match Results', description: 'View matching merchants', icon: Users },
  { href: '/contract/demo', label: 'Contract', description: 'View contract details', icon: FileCheck },
  { href: '/merchant', label: 'Merchant', description: 'Merchant dashboard', icon: Store },
  { href: '/dashboard', label: 'Platform Dashboard', description: 'View platform metrics', icon: BarChart3 },
];

const demoActions = [
  {
    id: 'run-demo',
    label: 'Run Demo Flow',
    description: 'Execute complete demo scenario',
    icon: Play,
  },
  {
    id: 'quick-match',
    label: 'Quick Match',
    description: 'Run matching algorithm with sample data',
    icon: Zap,
  },
];

export function CommandMenu() {
  const router = useRouter();
  const { isCommandMenuOpen, setCommandMenuOpen } = useStore();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const commands = useMemo<CommandItem[]>(() => {
    const navCommands: CommandItem[] = navigationItems.map((item) => ({
      id: `nav-${item.href}`,
      label: item.label,
      description: item.description,
      icon: <item.icon className="w-3.5 h-3.5" />,
      action: () => router.push(item.href),
      category: 'Navigation',
    }));

    const demoCommands: CommandItem[] = demoActions.map((item) => ({
      id: item.id,
      label: item.label,
      description: item.description,
      icon: <item.icon className="w-3.5 h-3.5" />,
      action: () => {
        if (item.id === 'run-demo') {
          router.push('/create');
        } else if (item.id === 'quick-match') {
          router.push('/matches');
        }
      },
      category: 'Demo Actions',
    }));

    return [...navCommands, ...demoCommands];
  }, [router]);

  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands;

    const lowerQuery = query.toLowerCase();
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(lowerQuery) ||
        cmd.description?.toLowerCase().includes(lowerQuery) ||
        cmd.category.toLowerCase().includes(lowerQuery)
    );
  }, [commands, query]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands]);

  useEffect(() => {
    if (isCommandMenuOpen) {
      inputRef.current?.focus();
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isCommandMenuOpen, setQuery]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandMenuOpen(!isCommandMenuOpen);
        return;
      }

      if (e.key === 'Escape' && isCommandMenuOpen) {
        setCommandMenuOpen(false);
        return;
      }

      if (!isCommandMenuOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        );
        scrollToSelected(selectedIndex + 1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        );
        scrollToSelected(selectedIndex - 1);
      } else if (e.key === 'Enter' && filteredCommands[selectedIndex]) {
        e.preventDefault();
        executeCommand(filteredCommands[selectedIndex]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isCommandMenuOpen, filteredCommands, selectedIndex, setCommandMenuOpen]);

  const scrollToSelected = (index: number) => {
    if (!listRef.current) return;

    const items = listRef.current.querySelectorAll('[data-command-item]');
    const item = items[index] as HTMLElement;
    if (item) {
      item.scrollIntoView({ block: 'nearest' });
    }
  };

  const executeCommand = (command: CommandItem) => {
    command.action();
    setCommandMenuOpen(false);
  };

  if (!isCommandMenuOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]"
      onClick={() => setCommandMenuOpen(false)}
    >
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />

      <div
        className="relative w-full max-w-lg bg-white border border-[rgba(123,61,44,0.12)] rounded-xl shadow-2xl shadow-[rgba(123,61,44,0.15)] overflow-hidden animate-slideUp"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 px-3 py-2.5 border-b border-[rgba(123,61,44,0.08)]">
          <Search className="w-4 h-4 text-[#c4b5a9]" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent text-[#7b3d2c] text-sm placeholder-[#c4b5a9] focus:outline-none"
          />
          <kbd className="px-1.5 py-0.5 text-[10px] text-[#c4b5a9] bg-[rgba(123,61,44,0.04)] rounded border border-[rgba(123,61,44,0.08)]">
            ESC
          </kbd>
        </div>

        <div ref={listRef} className="max-h-[360px] overflow-y-auto p-1.5">
          {filteredCommands.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-[#a89588]">
              <Search className="w-6 h-6 mb-2 opacity-50" />
              <p className="text-xs">No results found</p>
              <p className="text-[10px] text-[#c4b5a9]">Try a different search term</p>
            </div>
          ) : (
            <>
              {filteredCommands.some((c) => c.category === 'Navigation') && (
                <div className="mb-1">
                  <div className="px-2 py-1 text-[10px] font-medium text-[#c4b5a9] uppercase tracking-wider">
                    Navigation
                  </div>
                  {filteredCommands
                    .filter((c) => c.category === 'Navigation')
                    .map((command, index) => (
                      <button
                        key={command.id}
                        data-command-item
                        onClick={() => executeCommand(command)}
                        onMouseEnter={() =>
                          setSelectedIndex(
                            filteredCommands.findIndex((c) => c.id === command.id)
                          )
                        }
                        className={`
                          w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg
                          text-left transition-colors
                          ${
                            selectedIndex === filteredCommands.findIndex((c) => c.id === command.id)
                              ? 'bg-[rgba(123,61,44,0.08)] text-[#7b3d2c]'
                              : 'text-[#8b6e62] hover:bg-[rgba(123,61,44,0.04)]'
                          }
                        `}
                      >
                        <div className="p-1 rounded-md bg-[rgba(123,61,44,0.06)]">
                          {command.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium">{command.label}</div>
                          {command.description && (
                            <div className="text-[10px] text-[#c4b5a9] truncate">
                              {command.description}
                            </div>
                          )}
                        </div>
                        <ArrowRight className="w-3.5 h-3.5 text-[#c4b5a9]" />
                      </button>
                    ))}
                </div>
              )}

              {filteredCommands.some((c) => c.category === 'Demo Actions') && (
                <div>
                  <div className="px-2 py-1 text-[10px] font-medium text-[#c4b5a9] uppercase tracking-wider">
                    Demo Actions
                  </div>
                  {filteredCommands
                    .filter((c) => c.category === 'Demo Actions')
                    .map((command) => (
                      <button
                        key={command.id}
                        data-command-item
                        onClick={() => executeCommand(command)}
                        onMouseEnter={() =>
                          setSelectedIndex(
                            filteredCommands.findIndex((c) => c.id === command.id)
                          )
                        }
                        className={`
                          w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg
                          text-left transition-colors
                          ${
                            selectedIndex === filteredCommands.findIndex((c) => c.id === command.id)
                              ? 'bg-[rgba(123,61,44,0.08)] text-[#7b3d2c]'
                              : 'text-[#8b6e62] hover:bg-[rgba(123,61,44,0.04)]'
                          }
                        `}
                      >
                        <div className="p-1 rounded-md bg-[rgba(231,135,69,0.1)] text-[#e78745]">
                          {command.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium">{command.label}</div>
                          {command.description && (
                            <div className="text-[10px] text-[#c4b5a9] truncate">
                              {command.description}
                            </div>
                          )}
                        </div>
                        <ArrowRight className="w-3.5 h-3.5 text-[#c4b5a9]" />
                      </button>
                    ))}
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center justify-between px-3 py-2 border-t border-[rgba(123,61,44,0.08)] bg-[#faf5f2]">
          <div className="flex items-center gap-2.5 text-[10px] text-[#c4b5a9]">
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-[rgba(123,61,44,0.04)] rounded border border-[rgba(123,61,44,0.08)]">↑↓</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-[rgba(123,61,44,0.04)] rounded border border-[rgba(123,61,44,0.08)]">↵</kbd>
              Select
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-[rgba(123,61,44,0.04)] rounded border border-[rgba(123,61,44,0.08)]">esc</kbd>
              Close
            </span>
          </div>
          <div className="text-[10px] text-[#c4b5a9]">
            {filteredCommands.length} result{filteredCommands.length !== 1 && 's'}
          </div>
        </div>
      </div>
    </div>
  );
}
