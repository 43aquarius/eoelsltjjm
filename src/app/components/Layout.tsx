import { Outlet, NavLink } from "react-router";
import { Home, PlusCircle, CreditCard, PieChart } from "lucide-react";
import { motion } from "motion/react";
import { useStore } from "../store";

export const Layout = () => {
  const { theme, setTheme } = useStore();
  const navItems = [
    { to: "/", icon: Home, label: "Records" },
    { to: "/add", icon: PlusCircle, label: "Add" },
    { to: "/card", icon: CreditCard, label: "Card" },
    { to: "/admin", icon: PieChart, label: "Admin" },
  ];

  return (
    <div className={`theme-${theme} flex flex-col h-screen bg-[var(--bg)] text-[var(--text)] max-w-md mx-auto relative shadow-2xl overflow-hidden transition-colors duration-500`} style={{ fontFamily: 'var(--font)' }}>
      <style>{`
        .theme-linear {
          --bg: #000000;
          --surface: rgba(10, 10, 10, 0.8);
          --surface-elevated: #111111;
          --border: rgba(255,255,255,0.1);
          --border-hover: rgba(255,255,255,0.25);
          --text: #ffffff;
          --text-muted: #a3a3a3;
          --primary: #ffffff;
          --primary-text: #000000;
          --radius: 0.5rem;
          --shadow: 0 4px 20px rgba(0,0,0,0.5);
          --font: 'Inter', sans-serif;
        }
        .theme-apple {
          --bg: #f2f2f7;
          --surface: rgba(255, 255, 255, 0.72);
          --surface-elevated: #ffffff;
          --border: rgba(0,0,0,0.06);
          --border-hover: rgba(0,0,0,0.12);
          --text: #1c1c1e;
          --text-muted: #8e8e93;
          --primary: #007aff;
          --primary-text: #ffffff;
          --radius: 1.5rem;
          --shadow: 0 2px 16px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.02);
          --font: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", sans-serif';
        }
        /* Apple layout overrides */
        .theme-apple header { padding-top: 3.5rem !important; padding-bottom: 1rem !important; }
        .theme-apple header h1 { font-size: 1.75rem !important; letter-spacing: -0.03em !important; font-weight: 700 !important; }
        .theme-apple header p { font-size: 0.875rem !important; margin-top: 0.25rem !important; }
        .theme-apple .space-y-4 > :not(:first-child) { margin-top: 1.25rem !important; }
        .theme-apple nav { height: 5rem !important; padding-bottom: 1.25rem !important; backdrop-filter: blur(40px) saturate(180%) !important; -webkit-backdrop-filter: blur(40px) saturate(180%) !important; }
        .theme-apple nav a span { font-size: 0.625rem !important; letter-spacing: 0.02em !important; }
        .theme-apple nav a svg { width: 1.375rem !important; height: 1.375rem !important; }
        .theme-apple .p-6 { padding: 1.75rem !important; }
        .theme-apple .p-5 { padding: 1.5rem !important; }
        .theme-apple .p-4 { padding: 1.25rem !important; }
        .theme-apple .gap-4 { gap: 1.25rem !important; }
        .theme-apple .gap-3 { gap: 1rem !important; }
        .theme-apple .rounded-\[var\(--radius\)\] { border-radius: 1.5rem !important; }
        .theme-apple .rounded-\[calc\(var\(--radius\)\*0\.8\)\] { border-radius: 1.25rem !important; }
        .theme-apple .rounded-\[calc\(var\(--radius\)\*0\.5\)\] { border-radius: 0.75rem !important; }
        .theme-apple .rounded-full { border-radius: 9999px !important; }
        .theme-apple .rounded-xl { border-radius: 1.25rem !important; }
        .theme-apple .rounded-2xl { border-radius: 1.5rem !important; }
        .theme-apple .rounded-3xl { border-radius: 2rem !important; }
        .theme-apple .text-2xl { font-size: 1.625rem !important; letter-spacing: -0.025em !important; }
        .theme-apple .text-xl { font-size: 1.375rem !important; letter-spacing: -0.02em !important; }
        .theme-apple .text-lg { font-size: 1.125rem !important; letter-spacing: -0.01em !important; }
        .theme-apple .text-sm { font-size: 0.875rem !important; }
        .theme-apple .text-xs { font-size: 0.75rem !important; }
        .theme-apple .text-\[13px\] { font-size: 0.8125rem !important; }
        .theme-apple .text-\[11px\] { font-size: 0.6875rem !important; }
        .theme-apple .text-\[10px\] { font-size: 0.625rem !important; }
        .theme-apple .text-\[9px\] { font-size: 0.5625rem !important; }
        .theme-apple .font-medium { font-weight: 500 !important; }
        .theme-apple .font-semibold { font-weight: 600 !important; }
        .theme-apple .font-bold { font-weight: 700 !important; }
        .theme-apple .tracking-tight { letter-spacing: -0.02em !important; }
        .theme-apple .tracking-wide { letter-spacing: 0.01em !important; }
        .theme-apple .tracking-wider { letter-spacing: 0.03em !important; }
        .theme-apple .tracking-widest { letter-spacing: 0.06em !important; }
        .theme-apple .border { border-width: 0.5px !important; }
        .theme-apple .border-t { border-top-width: 0.5px !important; }
        .theme-apple .border-b { border-bottom-width: 0.5px !important; }
        .theme-apple .backdrop-blur-xl { backdrop-filter: blur(40px) saturate(180%) !important; -webkit-backdrop-filter: blur(40px) saturate(180%) !important; }
        .theme-apple .backdrop-blur-md { backdrop-filter: blur(20px) saturate(180%) !important; -webkit-backdrop-filter: blur(20px) saturate(180%) !important; }
        .theme-apple .aspect-\[4\/3\] { border-radius: 1.5rem !important; }
        .theme-apple .h-32 { height: 6.5rem !important; }
        .theme-apple .h-24 { height: 5.5rem !important; }
        .theme-apple .h-40 { height: 9rem !important; }
        .theme-apple .h-48 { height: 10rem !important; }
        .theme-apple .py-8 { padding-top: 2rem !important; padding-bottom: 2rem !important; }
        .theme-apple .py-6 { padding-top: 1.75rem !important; padding-bottom: 1.75rem !important; }
        .theme-apple .py-5 { padding-top: 1.25rem !important; padding-bottom: 1.25rem !important; }
        .theme-apple .py-4 { padding-top: 1rem !important; padding-bottom: 1rem !important; }
        .theme-apple .py-3 { padding-top: 0.75rem !important; padding-bottom: 0.75rem !important; }
        .theme-apple .py-2\.5 { padding-top: 0.625rem !important; padding-bottom: 0.625rem !important; }
        .theme-apple .py-1\.5 { padding-top: 0.375rem !important; padding-bottom: 0.375rem !important; }
        .theme-apple .px-6 { padding-left: 1.75rem !important; padding-right: 1.75rem !important; }
        .theme-apple .px-5 { padding-left: 1.5rem !important; padding-right: 1.5rem !important; }
        .theme-apple .px-4 { padding-left: 1.25rem !important; padding-right: 1.25rem !important; }
        .theme-apple .mb-8 { margin-bottom: 2.5rem !important; }
        .theme-apple .mb-6 { margin-bottom: 1.75rem !important; }
        .theme-apple .mb-4 { margin-bottom: 1.25rem !important; }
        .theme-apple .mt-8 { margin-top: 2.5rem !important; }
        .theme-apple .mt-4 { margin-top: 1.25rem !important; }
        .theme-apple .space-y-6 > :not(:first-child) { margin-top: 1.75rem !important; }
        .theme-apple .space-y-5 > :not(:first-child) { margin-top: 1.5rem !important; }
        .theme-apple .space-y-3 > :not(:first-child) { margin-top: 1rem !important; }
        .theme-apple .space-y-2 > :not(:first-child) { margin-top: 0.75rem !important; }
        .theme-apple .space-x-4 > :not(:first-child) { margin-left: 1.25rem !important; }
        .theme-apple .space-x-3 > :not(:first-child) { margin-left: 1rem !important; }
        .theme-apple .space-x-2 > :not(:first-child) { margin-left: 0.75rem !important; }
        .theme-md3 {
          --bg: #fffbff;
          --surface: #f7f2fa;
          --surface-elevated: #ffffff;
          --surface-container-low: #f7f2fa;
          --surface-container: #f3edf7;
          --surface-container-high: #ece6f0;
          --surface-container-highest: #e6e0e9;
          --border: #cac4d0;
          --border-hover: #79747e;
          --text: #1c1b1f;
          --text-muted: #49454f;
          --primary: #6750a4;
          --primary-text: #ffffff;
          --primary-container: #eaddff;
          --on-primary-container: #21005d;
          --secondary: #625b71;
          --secondary-container: #e8def8;
          --tertiary: #7d5260;
          --tertiary-container: #ffd8e4;
          --error: #b3261e;
          --radius: 0.75rem;
          --radius-card: 1rem;
  --radius-chip: 0.5rem;
          --shadow: 0 1px 2px 0 rgba(0,0,0,0.12), 0 1px 3px 1px rgba(0,0,0,0.08);
          --shadow-elevated: 0 2px 6px 2px rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.08);
          --font: 'Roboto, "Noto Sans SC", sans-serif;
          --state-hover: rgba(28, 27, 31, 0.08);
          --state-pressed: rgba(28, 27, 31, 0.12);
          --state-focus: rgba(28, 27, 31, 0.12);
        }
        /* M3 layout overrides */
        .theme-md3 header { padding-top: 3.5rem !important; padding-bottom: 0.75rem !important; }
        .theme-md3 header h1 { font-size: 1.375rem !important; font-weight: 400 !important; letter-spacing: 0 !important; line-height: 1.75rem !important; }
        .theme-md3 header p { font-size: 0.875rem !important; margin-top: 0.125rem !important; font-weight: 400 !important; }
        .theme-md3 nav { height: 5rem !important; padding-bottom: 0.75rem !important; backdrop-filter: none !important; -webkit-backdrop-filter: none !important; background: var(--surface-container-low) !important; border-top: none !important; }
        .theme-md3 nav a { border-radius: 1rem !important; padding: 0.25rem 0 !important; }
        .theme-md3 nav a svg { width: 1.5rem !important; height: 1.5rem !important; }
        .theme-md3 nav a span { font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.01em !important; }
        .theme-md3 nav a.opacity-70 { opacity: 1 !important; color: var(--text-muted) !important; }
        .theme-md3 .p-6 { padding: 1rem !important; }
        .theme-md3 .p-5 { padding: 1rem !important; }
        .theme-md3 .p-4 { padding: 1rem !important; }
        .theme-md3 .gap-4 { gap: 1rem !important; }
        .theme-md3 .gap-3 { gap: 0.75rem !important; }
        .theme-md3 .rounded-\[var\(--radius\)\] { border-radius: var(--radius-card) !important; }
        .theme-md3 .rounded-\[calc\(var\(--radius\)\*0\.8\)\] { border-radius: var(--radius-card) !important; }
        .theme-md3 .rounded-\[calc\(var\(--radius\)\*0\.5\)\] { border-radius: var(--radius-chip) !important; }
        .theme-md3 .rounded-full { border-radius: 9999px !important; }
        .theme-md3 .text-2xl { font-size: 1.375rem !important; letter-spacing: 0 !important; font-weight: 400 !important; }
        .theme-md3 .text-xl { font-size: 1.125rem !important; letter-spacing: 0 !important; font-weight: 400 !important; }
        .theme-md3 .text-lg { font-size: 1rem !important; letter-spacing: 0.01em !important; }
        .theme-md3 .text-sm { font-size: 0.875rem !important; letter-spacing: 0.01em !important; }
        .theme-md3 .text-xs { font-size: 0.75rem !important; letter-spacing: 0.03em !important; }
        .theme-md3 .text-\[13px\] { font-size: 0.8125rem !important; letter-spacing: 0.009em !important; }
        .theme-md3 .text-\[11px\] { font-size: 0.6875rem !important; letter-spacing: 0.03em !important; font-weight: 500 !important; text-transform: uppercase !important; }
        .theme-md3 .text-\[10px\] { font-size: 0.625rem !important; letter-spacing: 0.04em !important; font-weight: 500 !important; }
        .theme-md3 .text-\[9px\] { font-size: 0.5625rem !important; letter-spacing: 0.05em !important; font-weight: 500 !important; }
        .theme-md3 .font-medium { font-weight: 500 !important; }
        .theme-md3 .font-semibold { font-weight: 500 !important; }
        .theme-md3 .font-bold { font-weight: 500 !important; }
        .theme-md3 .tracking-tight { letter-spacing: 0 !important; }
        .theme-md3 .tracking-wide { letter-spacing: 0.01em !important; }
        .theme-md3 .tracking-wider { letter-spacing: 0.03em !important; }
        .theme-md3 .tracking-widest { letter-spacing: 0.05em !important; }
        .theme-md3 .border { border-width: 1px !important; border-color: var(--border) !important; }
        .theme-md3 .border-t { border-top-width: 1px !important; border-color: var(--border) !important; }
        .theme-md3 .border-b { border-bottom-width: 1px !important; border-color: var(--border) !important; }
        .theme-md3 .backdrop-blur-xl { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
        .theme-md3 .backdrop-blur-md { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
        .theme-md3 .aspect-\[4\/3\] { border-radius: var(--radius-card) !important; }
        .theme-md3 .h-32 { height: 7.5rem !important; }
        .theme-md3 .h-24 { height: 5rem !important; }
        .theme-md3 .h-40 { height: 8.5rem !important; }
        .theme-md3 .h-48 { height: 9rem !important; }
        .theme-md3 .py-8 { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; }
        .theme-md3 .py-6 { padding-top: 1rem !important; padding-bottom: 1rem !important; }
        .theme-md3 .py-5 { padding-top: 1rem !important; padding-bottom: 1rem !important; }
        .theme-md3 .py-4 { padding-top: 0.75rem !important; padding-bottom: 0.75rem !important; }
        .theme-md3 .py-3 { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .theme-md3 .py-2\.5 { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .theme-md3 .py-1\.5 { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        .theme-md3 .px-6 { padding-left: 1rem !important; padding-right: 1rem !important; }
        .theme-md3 .px-5 { padding-left: 1rem !important; padding-right: 1rem !important; }
        .theme-md3 .px-4 { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        .theme-md3 .mb-8 { margin-bottom: 1.5rem !important; }
        .theme-md3 .mb-6 { margin-bottom: 1rem !important; }
        .theme-md3 .mb-4 { margin-bottom: 0.75rem !important; }
        .theme-md3 .mt-8 { margin-top: 1.5rem !important; }
        .theme-md3 .mt-4 { margin-top: 0.75rem !important; }
        .theme-md3 .space-y-6 > :not(:first-child) { margin-top: 1rem !important; }
        .theme-md3 .space-y-5 > :not(:first-child) { margin-top: 0.75rem !important; }
        .theme-md3 .space-y-4 > :not(:first-child) { margin-top: 0.75rem !important; }
        .theme-md3 .space-y-3 > :not(:first-child) { margin-top: 0.5rem !important; }
        .theme-md3 .space-y-2 > :not(:first-child) { margin-top: 0.375rem !important; }
        .theme-md3 .space-x-4 > :not(:first-child) { margin-left: 0.75rem !important; }
        .theme-md3 .space-x-3 > :not(:first-child) { margin-left: 0.5rem !important; }
        .theme-md3 .space-x-2 > :not(:first-child) { margin-left: 0.375rem !important; }
        .theme-md3 .shadow-sm { box-shadow: var(--shadow) !important; }
        .theme-md3 .shadow-md { box-shadow: var(--shadow-elevated) !important; }
        .theme-md3 .shadow-lg { box-shadow: var(--shadow-elevated) !important; }
        .theme-md3 .shadow-xl { box-shadow: var(--shadow-elevated) !important; }
        .theme-md3 .shadow-2xl { box-shadow: var(--shadow-elevated) !important; }
        /* M3: nav active indicator pill */
        .theme-md3 nav a { position: relative !important; }
        .theme-md3 nav a.text-\[var\(--text\)\]::before {
          content: '' !important;
          position: absolute !important;
          top: 50% !important;
          left: 50% !important;
          transform: translate(-50%, -50%) !important;
          width: 4rem !important;
          height: 2rem !important;
          background: var(--secondary-container) !important;
          border-radius: 1rem !important;
          z-index: -1 !important;
          animation: m3-indicator-in 250ms cubic-bezier(0.2, 0, 0, 1) both !important;
        }
        @keyframes m3-indicator-in { from { opacity: 0; transform: translate(-50%, -50%) scaleX(0.6); } to { opacity: 1; transform: translate(-50%, -50%) scaleX(1); } }
        /* M3: card hover state layer */
        .theme-md3 .group:hover .group-hover\:opacity-100 { opacity: 1 !important; }
        .theme-md3 .group { transition: background-color 200ms cubic-bezier(0.2, 0, 0, 1) !important; }
        .theme-md3 .group:hover { background-color: color-mix(in srgb, var(--surface-elevated) 92%, var(--text)) !important; }
        /* M3: tonal button for tags */
        .theme-md3 header button.bg-\[var\(--primary\)\] { background: var(--primary) !important; font-weight: 500 !important; letter-spacing: 0.01em !important; }
        .theme-md3 header button:not(.bg-\[var\(--primary\)\]) { background: var(--surface-container-high) !important; }
        /* M3: input fields use surface-container-highest */
        .theme-md3 input[type="text"],
        .theme-md3 input[type="date"],
        .theme-md3 select,
        .theme-md3 textarea { background: var(--surface-container-high) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-chip) !important; }
        .theme-md3 input[type="text"]:focus,
        .theme-md3 input[type="date"]:focus,
        .theme-md3 select:focus,
        .theme-md3 textarea:focus { border-color: var(--primary) !important; border-width: 2px !important; }
        /* M3: export button as tonal */
        .theme-md3 button.bg-\[var\(--surface-elevated\)\].border { background: var(--secondary-container) !important; border: none !important; color: var(--secondary) !important; border-radius: var(--radius-chip) !important; font-weight: 500 !important; }
        /* M3: AI info banner */
        .theme-md3 .bg-\[var\(--primary\)\]\/10 { background: var(--primary-container) !important; color: var(--on-primary-container) !important; border: none !important; border-radius: var(--radius-card) !important; }
        .theme-notion {
          --bg: #ffffff;
          --surface: #ffffff;
          --surface-elevated: #f7f6f3;
          --surface-hover: rgba(55,53,47,0.035);
          --border: rgba(55,53,47,0.09);
          --border-hover: rgba(55,53,47,0.16);
          --border-light: rgba(55,53,47,0.06);
          --text: #37352f;
          --text-muted: #91918e;
          --text-faint: #b9b8b5;
          --primary: #2383e2;
          --primary-text: #ffffff;
          --primary-hover: #0068d5;
          --accent-bg: rgba(35,131,226,0.08);
          --accent-text: #2383e2;
          --radius: 0.25rem;
          --radius-md: 0.375rem;
          --shadow: none;
          --shadow-hover: 0 1px 3px rgba(0,0,0,0.04);
          --font: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif;
          --font-serif: 'Georgia, "Lyon Display", "Noto Serif", ui-serif, serif;
          --font-mono: "SFMono-Regular", "Menlo", "Consolas", "PT Mono", monospace;
          --state-hover: rgba(55,53,47,0.055);
          --state-active: rgba(55,53,47,0.09);
        }
        /* Notion layout overrides */
        .theme-notion header { padding-top: 3rem !important; padding-bottom: 0.5rem !important; background: rgba(255,255,255,0.85) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        .theme-notion header h1 { font-family: var(--font-serif) !important; font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.01em !important; line-height: 1.3 !important; color: var(--text) !important; }
        .theme-notion header p { font-size: 0.8125rem !important; margin-top: 0.125rem !important; color: var(--text-muted) !important; font-weight: 400 !important; }
        .theme-notion nav { height: 2.75rem !important; padding-bottom: 0 !important; background: rgba(255,255,255,0.9) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; border-top: 1px solid var(--border-light) !important; }
        .theme-notion nav a { border-radius: 0.25rem !important; padding: 0.25rem 0.5rem !important; gap: 0.25rem !important; }
        .theme-notion nav a svg { width: 1.125rem !important; height: 1.125rem !important; strokeWidth: 1.5 !important; }
        .theme-notion nav a span { font-size: 0.6875rem !important; font-weight: 400 !important; letter-spacing: 0 !important; }
        .theme-notion nav a.opacity-70 { opacity: 1 !important; color: var(--text-muted) !important; }
        .theme-notion nav a.text-\[var\(--text\)\] { color: var(--text) !important; font-weight: 500 !important; }
        .theme-notion nav a.text-\[var\(--text\)\]::before { display: none !important; }
        /* Notion: no dot indicator, active = bold + color */
        .theme-notion nav a .relative .absolute.-bottom-2\.5 { display: none !important; }
        .theme-notion .p-6 { padding: 0.25rem 0 !important; }
        .theme-notion .p-5 { padding: 0.75rem !important; }
        .theme-notion .p-4 { padding: 0.5rem !important; }
        .theme-notion .gap-4 { gap: 0.75rem !important; }
        .theme-notion .gap-3 { gap: 0.5rem !important; }
        .theme-notion .rounded-\[var\(--radius\)\] { border-radius: var(--radius-md) !important; }
        .theme-notion .rounded-\[calc\(var\(--radius\)\*0\.8\)\] { border-radius: var(--radius-md) !important; }
        .theme-notion .rounded-\[calc\(var\(--radius\)\*0\.5\)\] { border-radius: var(--radius) !important; }
        .theme-notion .rounded-full { border-radius: 0.25rem !important; }
        .theme-notion .rounded-xl { border-radius: var(--radius-md) !important; }
        .theme-notion .rounded-2xl { border-radius: var(--radius-md) !important; }
        .theme-notion .rounded-3xl { border-radius: 0.5rem !important; }
        .theme-notion .text-2xl { font-family: var(--font-serif) !important; font-size: 1.5rem !important; letter-spacing: -0.01em !important; font-weight: 700 !important; }
        .theme-notion .text-xl { font-size: 1.125rem !important; letter-spacing: -0.005em !important; font-weight: 600 !important; }
        .theme-notion .text-lg { font-size: 1rem !important; font-weight: 600 !important; }
        .theme-notion .text-sm { font-size: 0.8125rem !important; line-height: 1.4 !important; }
        .theme-notion .text-xs { font-size: 0.6875rem !important; color: var(--text-muted) !important; }
        .theme-notion .text-\[13px\] { font-size: 0.8125rem !important; }
        .theme-notion .text-\[11px\] { font-size: 0.6875rem !important; letter-spacing: 0 !important; font-weight: 500 !important; text-transform: none !important; color: var(--text-muted) !important; }
        .theme-notion .text-\[10px\] { font-size: 0.625rem !important; letter-spacing: 0 !important; color: var(--text-faint) !important; }
        .theme-notion .text-\[9px\] { font-size: 0.5625rem !important; letter-spacing: 0 !important; color: var(--text-faint) !important; }
        .theme-notion .font-medium { font-weight: 500 !important; }
        .theme-notion .font-semibold { font-weight: 600 !important; }
        .theme-notion .font-bold { font-weight: 600 !important; }
        .theme-notion .tracking-tight { letter-spacing: -0.01em !important; }
        .theme-notion .tracking-wide { letter-spacing: 0 !important; }
        .theme-notion .tracking-wider { letter-spacing: 0.01em !important; }
        .theme-notion .tracking-widest { letter-spacing: 0.02em !important; }
        .theme-notion .border { border: none !important; }
        .theme-notion .border-t { border-top: 1px solid var(--border-light) !important; }
        .theme-notion .border-b { border-bottom: 1px solid var(--border-light) !important; }
        .theme-notion .border-gray-100,
        .theme-notion .border-gray-200,
        .theme-notion .border-gray-50 { border-color: var(--border-light) !important; }
        .theme-notion .backdrop-blur-xl { backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        .theme-notion .backdrop-blur-md { backdrop-filter: blur(8px) !important; -webkit-backdrop-filter: blur(8px) !important; }
        .theme-notion .aspect-\[4\/3\] { border-radius: var(--radius-md) !important; border-style: solid !important; }
        .theme-notion .h-32 { height: auto !important; min-height: 5.5rem !important; }
        .theme-notion .h-24 { height: auto !important; min-height: 4rem !important; }
        .theme-notion .h-40 { height: 7rem !important; }
        .theme-notion .h-48 { height: 7rem !important; }
        .theme-notion .py-8 { padding-top: 1rem !important; padding-bottom: 1rem !important; }
        .theme-notion .py-6 { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        .theme-notion .py-5 { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .theme-notion .py-4 { padding-top: 0.375rem !important; padding-bottom: 0.375rem !important; }
        .theme-notion .py-3 { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        .theme-notion .py-2\.5 { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        .theme-notion .py-1\.5 { padding-top: 0.125rem !important; padding-bottom: 0.125rem !important; }
        .theme-notion .px-6 { padding-left: 0 !important; padding-right: 0 !important; }
        .theme-notion .px-5 { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        .theme-notion .px-4 { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        .theme-notion .mb-8 { margin-bottom: 1rem !important; }
        .theme-notion .mb-6 { margin-bottom: 0.75rem !important; }
        .theme-notion .mb-4 { margin-bottom: 0.5rem !important; }
        .theme-notion .mt-8 { margin-top: 1rem !important; }
        .theme-notion .mt-4 { margin-top: 0.5rem !important; }
        .theme-notion .space-y-6 > :not(:first-child) { margin-top: 0.5rem !important; }
        .theme-notion .space-y-5 > :not(:first-child) { margin-top: 0.5rem !important; }
        .theme-notion .space-y-4 > :not(:first-child) { margin-top: 0.5rem !important; }
        .theme-notion .space-y-3 > :not(:first-child) { margin-top: 0.375rem !important; }
        .theme-notion .space-y-2 > :not(:first-child) { margin-top: 0.25rem !important; }
        .theme-notion .space-x-4 > :not(:first-child) { margin-left: 0.5rem !important; }
        .theme-notion .space-x-3 > :not(:first-child) { margin-left: 0.375rem !important; }
        .theme-notion .space-x-2 > :not(:first-child) { margin-left: 0.25rem !important; }
        /* Notion: kill all shadows */
        .theme-notion .shadow-sm,
        .theme-notion .shadow-md,
        .theme-notion .shadow-lg,
        .theme-notion .shadow-xl,
        .theme-notion .shadow-2xl,
        .theme-notion .shadow-\[var\(--shadow\)\] { box-shadow: none !important; }
        /* Notion: cards = no border, hover bg */
        .theme-notion .group { border: none !important; background: transparent !important; transition: background-color 150ms ease !important; border-radius: var(--radius-md) !important; }
        .theme-notion .group:hover { background-color: var(--state-hover) !important; }
        .theme-notion .group .bg-\[var\(--surface\)\] { background: transparent !important; }
        .theme-notion .group .bg-\[var\(--surface-elevated\)\] { background: transparent !important; }
        /* Notion: card image section */
        .theme-notion .group .w-1\/3 { border-radius: var(--radius-md) 0 0 var(--radius-md) !important; overflow: hidden !important; }
        .theme-notion .group img { border-radius: var(--radius-md) 0 0 var(--radius-md) !important; }
        /* Notion: delete button = text link style */
        .theme-notion .group button { background: transparent !important; border: none !important; box-shadow: none !important; color: var(--text-muted) !important; padding: 0.25rem !important; border-radius: var(--radius) !important; }
        .theme-notion .group button:hover { color: #eb5757 !important; background: rgba(235,87,87,0.08) !important; }
        /* Notion: tag pills */
        .theme-notion header button.rounded-full { border-radius: 0.25rem !important; font-weight: 500 !important; font-size: 0.8125rem !important; padding: 0.25rem 0.625rem !important; border: 1px solid var(--border) !important; }
        .theme-notion header button.bg-\[var\(--primary\)\] { background: var(--text) !important; color: var(--bg) !important; border-color: var(--text) !important; }
        .theme-notion header button:not(.bg-\[var\(--primary\)\]) { background: transparent !important; color: var(--text-muted) !important; }
        .theme-notion header button:not(.bg-\[var\(--primary\)\]):hover { background: var(--state-hover) !important; color: var(--text) !important; }
        /* Notion: inputs */
        .theme-notion input[type="text"],
        .theme-notion input[type="date"],
        .theme-notion select,
        .theme-notion textarea { background: var(--surface-elevated) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; font-size: 0.8125rem !important; transition: border-color 150ms ease !important; }
        .theme-notion input[type="text"]:focus,
        .theme-notion input[type="date"]:focus,
        .theme-notion select:focus,
        .theme-notion textarea:focus { border-color: var(--primary) !important; box-shadow: 0 0 0 2px rgba(35,131,226,0.15) !important; }
        /* Notion: search bar */
        .theme-notion .bg-\[var\(--surface-elevated\)].border { background: var(--surface-elevated) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; }
        /* Notion: export button = text link */
        .theme-notion button.bg-\[var\(--surface-elevated\)\].border { background: transparent !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; color: var(--text) !important; font-weight: 500 !important; }
        .theme-notion button.bg-\[var\(--surface-elevated\)\].border:hover { background: var(--state-hover) !important; }
        /* Notion: AI info banner */
        .theme-notion .bg-\[var\(--primary\)\]\/10 { background: var(--accent-bg) !important; color: var(--accent-text) !important; border: 1px solid rgba(35,131,226,0.12) !important; border-radius: var(--radius-md) !important; }
        /* Notion: form section cards */
        .theme-notion .space-y-5.bg-\[var\(--surface\)\] { background: transparent !important; border: 1px solid var(--border-light) !important; border-radius: var(--radius-md) !important; padding: 0.75rem !important; }
        /* Notion: save button */
        .theme-notion button.bg-\[var\(--primary\)\].text-\[var\(--primary-text\)\] { background: var(--primary) !important; border-radius: var(--radius-md) !important; font-weight: 500 !important; font-size: 0.8125rem !important; transition: background-color 150ms ease !important; }
        .theme-notion button.bg-\[var\(--primary\)\].text-\[var\(--primary-text\)\]:hover { background: var(--primary-hover) !important; }
        /* Notion: upload zone hover */
        .theme-notion .border-dashed:hover { border-color: var(--primary) !important; background: var(--accent-bg) !important; }
        /* Notion: scan line */
        .theme-notion .bg-\[var\(--primary\)].shadow-\[0_0_15px_2px_var\(--primary\)\] { background: var(--primary) !important; box-shadow: 0 0 8px rgba(35,131,226,0.4) !important; }
        /* Notion: dashboard stat cards */
        .theme-notion .grid .bg-\[var(--surface-container\)\],
        .theme-notion .grid .bg-\[var\(--surface\)\] { background: transparent !important; border: 1px solid var(--border-light) !important; border-radius: var(--radius-md) !important; padding: 0.75rem !important; }
        .theme-notion .grid .text-2xl { font-family: var(--font-serif) !important; font-weight: 700 !important; font-size: 1.75rem !important; }
        /* Notion: chart cards */
        .theme-notion .bg-\[var(--surface-container\)].p-5,
        .theme-notion .bg-\[var\(--surface\)\].p-5 { background: transparent !important; border: 1px solid var(--border-light) !important; border-radius: var(--radius-md) !important; padding: 0.75rem !important; }
        /* Notion: recent entries */
        .theme-notion .border-b.border-\[var\(--border\)\] { border-bottom: 1px solid var(--border-light) !important; }
        /* Notion: filter button */
        .theme-notion button.text-\[var\(--text-muted\)\]:hover { color: var(--text) !important; background: var(--state-hover) !important; border-radius: var(--radius) !important; }
        /* Notion: memorial card */
        .theme-notion .max-w-sm.bg-\[var\(--surface\)\] { background: transparent !important; border: 1px solid var(--border-light) !important; }
        .theme-notion .max-w-sm .bg-\[var\(--surface-elevated\)\] { background: var(--surface-elevated) !important; }
        .theme-notion .max-w-sm .rounded-full { border-radius: 50% !important; }
        .theme-notion .max-w-sm .rounded-\[calc\(var\(--radius\)\*0\.3\)\] { border-radius: var(--radius) !important; }
        .theme-notion .max-w-sm button.bg-\[var\(--primary\)\] { background: var(--text) !important; border-radius: var(--radius-md) !important; }
        .theme-notion .max-w-sm button.bg-\[var\(--surface\)\] { background: transparent !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; }
        /* Notion: select dropdown in switcher */
        .theme-notion select { border-color: var(--border) !important; font-size: 0.6875rem !important; }
        /* Notion: custom scrollbar */
        .theme-notion .custom-scrollbar::-webkit-scrollbar { width: 6px !important; }
        .theme-notion .custom-scrollbar::-webkit-scrollbar-track { background: transparent !important; }
        .theme-notion .custom-scrollbar::-webkit-scrollbar-thumb { background: var(--border) !important; border-radius: 3px !important; }
        .theme-notion .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: var(--text-muted) !important; }
        /* Notion: selection color */
        .theme-notion ::selection { background: rgba(35,131,226,0.15) !important; }
        .theme-framer {
          --bg: #0a0a0b;
          --bg-gradient-1: radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,87,255,0.12) 0%, transparent 60%);
          --bg-gradient-2: radial-gradient(ellipse 60% 50% at 85% 75%, rgba(255,0,85,0.08) 0%, transparent 55%);
          --bg-gradient-3: radial-gradient(ellipse 70% 40% at 50% 100%, rgba(139,92,246,0.06) 0%, transparent 50%);
          --surface: rgba(255, 255, 255, 0.04);
          --surface-elevated: rgba(255, 255, 255, 0.07);
          --surface-glass: rgba(255, 255, 255, 0.06);
          --border: rgba(255,255,255,0.06);
          --border-hover: rgba(255,255,255,0.14);
          --border-glow: rgba(0,87,255,0.3);
          --text: #f5f5f7;
          --text-muted: #86868b;
          --text-faint: #6e6e73;
          --primary: #0057ff;
          --primary-text: #ffffff;
          --primary-glow: rgba(0,87,255,0.35);
          --accent-pink: #ff0055;
          --accent-purple: #8b5cf6;
          --accent-green: #00cc66;
          --radius: 1rem;
          --radius-lg: 1.5rem;
          --shadow: 0 4px 24px rgba(0,0,0,0.4);
          --shadow-glow: 0 0 40px rgba(0,87,255,0.15);
          --font: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif';
          --font-display: '"Inter", sans-serif';
          --state-hover: rgba(255,255,255,0.06);
        }
        /* Framer: gradient mesh background */
        .theme-framer {
          background:
            var(--bg-gradient-1),
            var(--bg-gradient-2),
            var(--bg-gradient-3),
            var(--bg) !important;
        }
        /* Framer: glassmorphism card base */
        .theme-framer .bg-\[var\(--surface\)\] {
          background: var(--surface-glass) !important;
          backdrop-filter: blur(40px) saturate(150%) !important;
          -webkit-backdrop-filter: blur(40px) saturate(150%) !important;
        }
        /* Framer: layout overrides */
        .theme-framer header { padding-top: 4rem !important; padding-bottom: 1rem !important; background: transparent !important; border-bottom-color: var(--border) !important; backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
        .theme-framer header h1 { font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.035em !important; line-height: 1.1 !important; background: linear-gradient(135deg, #f5f5f7 0%, #a1a1a6 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; }
        .theme-framer header p { font-size: 0.875rem !important; margin-top: 0.375rem !important; color: var(--text-muted) !important; font-weight: 400 !important; letter-spacing: -0.01em !important; }
        .theme-framer nav { height: 5rem !important; padding-bottom: 1rem !important; background: rgba(10,10,11,0.65) !important; backdrop-filter: blur(40px) saturate(180%) !important; -webkit-backdrop-filter: blur(40px) saturate(180%) !important; border-top: 1px solid var(--border) !important; }
        .theme-framer nav a { border-radius: 0.75rem !important; padding: 0.375rem 0.75rem !important; gap: 0.375rem !important; }
        .theme-framer nav a svg { width: 1.375rem !important; height: 1.375rem !important; strokeWidth: 1.5 !important; }
        .theme-framer nav a span { font-size: 0.625rem !important; font-weight: 500 !important; letter-spacing: 0.01em !important; }
        .theme-framer nav a.opacity-70 { opacity: 1 !important; color: var(--text-muted) !important; }
        .theme-framer nav a.text-\[var\(--text\)\] { color: var(--text) !important; }
        .theme-framer nav a .relative .absolute.-bottom-2\.5 { width: 1.5rem !important; height: 0.25rem !important; border-radius: 0.25rem !important; background: var(--primary) !important; box-shadow: 0 0 12px var(--primary-glow) !important; }
        /* Framer: cards = glass */
        .theme-framer .group { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important; transition: border-color 300ms ease, box-shadow 300ms ease, transform 300ms ease !important; }
        .theme-framer .group:hover { border-color: var(--border-hover) !important; box-shadow: var(--shadow-glow) !important; transform: translateY(-2px) !important; }
        .theme-framer .group .bg-\[var\(--surface\)\] { background: transparent !important; }
        .theme-framer .group .bg-\[var\(--surface-elevated\)\] { background: transparent !important; }
        .theme-framer .group .w-1\/3 { border-radius: var(--radius-lg) 0 0 var(--radius-lg) !important; }
        .theme-framer .group img { border-radius: var(--radius-lg) 0 0 var(--radius-lg) !important; }
        /* Framer: delete button */
        .theme-framer .group button { background: rgba(255,255,255,0.06) !important; border: 1px solid var(--border) !important; color: var(--text-muted) !important; border-radius: 0.5rem !important; padding: 0.375rem !important; transition: all 200ms ease !important; }
        .theme-framer .group button:hover { color: #ff3b30 !important; border-color: rgba(255,59,48,0.3) !important; background: rgba(255,59,48,0.08) !important; }
        /* Framer: tag pills */
        .theme-framer header button.rounded-full { border-radius: 0.5rem !important; font-weight: 600 !important; font-size: 0.8125rem !important; padding: 0.375rem 0.875rem !important; border: 1px solid var(--border) !important; letter-spacing: -0.01em !important; }
        .theme-framer header button.bg-\[var\(--primary\)\] { background: var(--primary) !important; color: #fff !important; border-color: var(--primary) !important; box-shadow: 0 0 20px var(--primary-glow) !important; }
        .theme-framer header button:not(.bg-\[var\(--primary\)\]) { background: var(--surface-glass) !important; color: var(--text-muted) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        .theme-framer header button:not(.bg-\[var\(--primary\)\]):hover { border-color: var(--border-hover) !important; color: var(--text) !important; background: var(--surface-elevated) !important; }
        /* Framer: inputs */
        .theme-framer input[type="text"],
        .theme-framer input[type="date"],
        .theme-framer select,
        .theme-framer textarea { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; font-size: 0.875rem !important; color: var(--text) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; transition: border-color 200ms ease, box-shadow 200ms ease !important; }
        .theme-framer input[type="text"]:focus,
        .theme-framer input[type="date"]:focus,
        .theme-framer select:focus,
        .theme-framer textarea:focus { border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--primary-glow) !important; }
        .theme-framer input::placeholder,
        .theme-framer textarea::placeholder { color: var(--text-faint) !important; }
        /* Framer: search bar */
        .theme-framer .bg-\[var\(--surface-elevated\)].border { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        .theme-framer .bg-\[var\(--surface-elevated\)].border:focus-within { border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--primary-glow) !important; }
        /* Framer: export button */
        .theme-framer button.bg-\[var\(--secondary-container\)\] { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: var(--radius) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; font-weight: 600 !important; }
        .theme-framer button.bg-\[var\(--secondary-container\)\]:hover { border-color: var(--border-hover) !important; box-shadow: var(--shadow-glow) !important; }
        /* Framer: AI info banner */
        .theme-framer .bg-\[var\(--primary\)\]\/10 { background: rgba(0,87,255,0.1) !important; color: var(--text) !important; border: 1px solid rgba(0,87,255,0.15) !important; border-radius: var(--radius-lg) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        /* Framer: form section cards */
        .theme-framer .space-y-5.bg-\[var\(--surface\)\] { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; padding: 1.25rem !important; backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important; }
        /* Framer: save button - gradient */
        .theme-framer button.bg-\[var\(--primary\)\].text-\[var\(--primary-text\)\] { background: linear-gradient(135deg, var(--primary) 0%, #2977ff 100%) !important; border-radius: var(--radius) !important; font-weight: 600 !important; font-size: 0.875rem !important; letter-spacing: -0.01em !important; box-shadow: 0 4px 20px var(--primary-glow) !important; transition: all 300ms ease !important; border: none !important; }
        .theme-framer button.bg-\[var\(--primary\)\].text-\[var\(--primary-text\)\]:hover { box-shadow: 0 6px 30px var(--primary-glow), 0 0 60px rgba(0,87,255,0.1) !important; transform: translateY(-1px) !important; }
        .theme-framer button.bg-\[var\(--primary\)\].text-\[var\(--primary-text\)\]:active { transform: scale(0.98) !important; }
        /* Framer: upload zone */
        .theme-framer .border-dashed { border-color: var(--border) !important; border-radius: var(--radius-lg) !important; background: var(--surface-glass) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; transition: all 300ms ease !important; }
        .theme-framer .border-dashed:hover { border-color: var(--primary) !important; background: rgba(0,87,255,0.06) !important; box-shadow: 0 0 30px rgba(0,87,255,0.1) !important; }
        .theme-framer .border-dashed .p-3 { background: var(--surface-elevated) !important; border-radius: var(--radius) !important; }
        /* Framer: scan line glow */
        .theme-framer .bg-\[var\(--primary\)].shadow-\[0_0_15px_2px_var\(--primary\)\] { background: linear-gradient(90deg, transparent, var(--primary), transparent) !important; box-shadow: 0 0 24px 4px var(--primary-glow) !important; }
        /* Framer: dashboard stat cards */
        .theme-framer .grid .bg-\[var(--surface-container\)\],
        .theme-framer .grid .bg-\[var\(--surface\)\] { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; padding: 1.25rem !important; backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important; transition: border-color 300ms ease, box-shadow 300ms ease !important; }
        .theme-framer .grid .bg-\[var(--surface-container\)\]:hover,
        .theme-framer .grid .bg-\[var\(--surface\)\]:hover { border-color: var(--border-hover) !important; box-shadow: var(--shadow-glow) !important; }
        .theme-framer .grid .text-2xl { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; background: linear-gradient(135deg, #f5f5f7 0%, #a1a1a6 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; }
        /* Framer: chart cards */
        .theme-framer .bg-\[var(--surface-container\)].p-5,
        .theme-framer .bg-\[var\(--surface\)\].p-5 { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; padding: 1.25rem !important; backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important; }
        /* Framer: recent entries divider */
        .theme-framer .border-b.border-\[var\(--border\)\] { border-bottom-color: var(--border) !important; }
        /* Framer: filter button */
        .theme-framer button.text-\[var\(--text-muted\)\] { border-radius: 0.5rem !important; }
        .theme-framer button.text-\[var\(--text-muted\)\]:hover { color: var(--text) !important; background: var(--state-hover) !important; }
        /* Framer: memorial card */
        .theme-framer .max-w-sm.bg-\[var\(--surface\)\] { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important; backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important; }
        .theme-framer .max-w-sm .bg-\[var\(--surface-elevated\)\] { background: var(--surface-elevated) !important; border-radius: var(--radius) !important; }
        .theme-framer .max-w-sm .rounded-full { border-radius: 50% !important; }
        .theme-framer .max-w-sm button.bg-\[var\(--primary\)\] { background: linear-gradient(135deg, var(--primary) 0%, #2977ff 100%) !important; border-radius: var(--radius) !important; box-shadow: 0 4px 20px var(--primary-glow) !important; border: none !important; }
        .theme-framer .max-w-sm button.bg-\[var\(--surface\)\] { background: var(--surface-glass) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important; }
        /* Framer: select dropdown */
        .theme-framer select { border-color: var(--border) !important; font-size: 0.6875rem !important; background: var(--surface-glass) !important; color: var(--text) !important; }
        /* Framer: custom scrollbar */
        .theme-framer .custom-scrollbar::-webkit-scrollbar { width: 6px !important; }
        .theme-framer .custom-scrollbar::-webkit-scrollbar-track { background: transparent !important; }
        .theme-framer .custom-scrollbar::-webkit-scrollbar-thumb { background: var(--border) !important; border-radius: 3px !important; }
        .theme-framer .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: var(--border-hover) !important; }
        /* Framer: selection */
        .theme-framer ::selection { background: rgba(0,87,255,0.25) !important; }
        /* Framer: shadow overrides for glow feel */
        .theme-framer .shadow-\[var\(--shadow\)\] { box-shadow: var(--shadow) !important; }
        .theme-framer .shadow-sm { box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important; }
        .theme-framer .shadow-md { box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important; }
        .theme-framer .shadow-lg { box-shadow: 0 8px 30px rgba(0,0,0,0.4) !important; }
        .theme-framer .shadow-xl { box-shadow: 0 12px 40px rgba(0,0,0,0.45) !important; }
        .theme-framer .shadow-2xl { box-shadow: 0 16px 50px rgba(0,0,0,0.5) !important; }
        /* Framer: typography */
        .theme-framer .text-2xl { font-size: 1.75rem !important; letter-spacing: -0.035em !important; font-weight: 700 !important; }
        .theme-framer .text-xl { font-size: 1.375rem !important; letter-spacing: -0.025em !important; font-weight: 700 !important; }
        .theme-framer .text-lg { font-size: 1.125rem !important; letter-spacing: -0.015em !important; font-weight: 600 !important; }
        .theme-framer .text-sm { font-size: 0.875rem !important; letter-spacing: -0.01em !important; }
        .theme-framer .text-xs { font-size: 0.75rem !important; color: var(--text-muted) !important; }
        .theme-framer .text-\[13px\] { font-size: 0.8125rem !important; letter-spacing: -0.005em !important; }
        .theme-framer .text-\[11px\] { font-size: 0.6875rem !important; letter-spacing: 0 !important; font-weight: 500 !important; color: var(--text-muted) !important; }
        .theme-framer .text-\[10px\] { font-size: 0.625rem !important; letter-spacing: 0 !important; color: var(--text-faint) !important; }
        .theme-framer .text-\[9px\] { font-size: 0.5625rem !important; letter-spacing: 0 !important; color: var(--text-faint) !important; }
        .theme-framer .font-medium { font-weight: 500 !important; }
        .theme-framer .font-semibold { font-weight: 600 !important; }
        .theme-framer .font-bold { font-weight: 700 !important; }
        .theme-framer .tracking-tight { letter-spacing: -0.025em !important; }
        .theme-framer .tracking-wide { letter-spacing: 0.01em !important; }
        .theme-framer .tracking-wider { letter-spacing: 0.03em !important; }
        .theme-framer .tracking-widest { letter-spacing: 0.05em !important; }
        /* Framer: spacing */
        .theme-framer .p-6 { padding: 1.25rem !important; }
        .theme-framer .p-5 { padding: 1.25rem !important; }
        .theme-framer .p-4 { padding: 1rem !important; }
        .theme-framer .gap-4 { gap: 1rem !important; }
        .theme-framer .gap-3 { gap: 0.75rem !important; }
        .theme-framer .py-8 { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; }
        .theme-framer .py-6 { padding-top: 1.25rem !important; padding-bottom: 1.25rem !important; }
        .theme-framer .py-5 { padding-top: 1rem !important; padding-bottom: 1rem !important; }
        .theme-framer .py-4 { padding-top: 0.75rem !important; padding-bottom: 0.75rem !important; }
        .theme-framer .py-3 { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .theme-framer .py-2\.5 { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .theme-framer .py-1\.5 { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        .theme-framer .px-6 { padding-left: 1.25rem !important; padding-right: 1.25rem !important; }
        .theme-framer .px-5 { padding-left: 1.25rem !important; padding-right: 1.25rem !important; }
        .theme-framer .px-4 { padding-left: 1rem !important; padding-right: 1rem !important; }
        .theme-framer .mb-8 { margin-bottom: 1.5rem !important; }
        .theme-framer .mb-6 { margin-bottom: 1.25rem !important; }
        .theme-framer .mb-4 { margin-bottom: 1rem !important; }
        .theme-framer .mt-8 { margin-top: 1.5rem !important; }
        .theme-framer .mt-4 { margin-top: 1rem !important; }
        .theme-framer .space-y-6 > :not(:first-child) { margin-top: 1.25rem !important; }
        .theme-framer .space-y-5 > :not(:first-child) { margin-top: 1.25rem !important; }
        .theme-framer .space-y-4 > :not(:first-child) { margin-top: 1rem !important; }
        .theme-framer .space-y-3 > :not(:first-child) { margin-top: 0.75rem !important; }
        .theme-framer .space-y-2 > :not(:first-child) { margin-top: 0.5rem !important; }
        .theme-framer .space-x-4 > :not(:first-child) { margin-left: 1rem !important; }
        .theme-framer .space-x-3 > :not(:first-child) { margin-left: 0.75rem !important; }
        .theme-framer .space-x-2 > :not(:first-child) { margin-left: 0.5rem !important; }
        .custom-scrollbar::-webkit-scrollbar { width: 0px; }
      `}</style>

      {/* Header Switcher */}
      <div className="absolute top-5 right-6 z-[60]">
        <div className="relative">
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value as any)}
            className="bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--text)] rounded-full pl-3 pr-6 py-1.5 text-[11px] font-medium outline-none focus:border-[var(--primary)] transition-colors shadow-[var(--shadow)] cursor-pointer appearance-none"
          >
            <option value="linear">风格1</option>
            <option value="apple">风格2</option>
            <option value="md3">风格3</option>
            <option value="notion">风格4</option>
            <option value="framer">风格5</option>
          </select>
          <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-[8px] text-[var(--text-muted)]">
            ▼
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pb-20 custom-scrollbar relative z-0">
        <Outlet />
      </div>

      <nav className="absolute bottom-0 w-full bg-[var(--surface)] backdrop-blur-xl border-t border-[var(--border)] flex justify-around items-center h-20 px-2 pb-4 z-50 transition-colors duration-500">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex flex-col items-center justify-center w-full h-full space-y-1.5 transition-all duration-300 ${isActive ? "text-[var(--text)]" : "text-[var(--text-muted)] hover:text-[var(--text)] opacity-70 hover:opacity-100"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <div className="relative">
                  <item.icon size={22} strokeWidth={isActive ? 2 : 1.5} />
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute -bottom-2.5 left-1/2 -translate-x-1/2 w-1 h-1 bg-[var(--primary)] rounded-full shadow-[0_0_10px_var(--primary)]"
                    />
                  )}
                </div>
                <span className="text-[10px] font-medium tracking-wide">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  );
};
