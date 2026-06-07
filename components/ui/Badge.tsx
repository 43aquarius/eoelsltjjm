'use client';

import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';

type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode;
  variant?: BadgeVariant;
  dot?: boolean;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-[rgba(123,61,44,0.08)] text-[#8b6e62]',
  primary: 'bg-[rgba(231,135,69,0.12)] text-[#e78745]',
  success: 'bg-[rgba(74,222,128,0.12)] text-[#16a34a]',
  warning: 'bg-[rgba(251,191,36,0.12)] text-[#d97706]',
  error: 'bg-[rgba(248,113,113,0.12)] text-[#dc2626]',
  info: 'bg-[rgba(96,165,250,0.12)] text-[#2563eb]',
};

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      children,
      variant = 'default',
      dot = false,
      className = '',
      ...props
    },
    ref
  ) => {
    const dotColorClass = {
      default: 'bg-[#a89588]',
      primary: 'bg-[#e78745]',
      success: 'bg-[#16a34a]',
      warning: 'bg-[#d97706]',
      error: 'bg-[#dc2626]',
      info: 'bg-[#2563eb]',
    }[variant];

    return (
      <span
        ref={ref}
        className={`
          inline-flex items-center gap-1.5
          px-2 py-0.5
          text-[11px] font-medium rounded-md
          ${variantStyles[variant]}
          ${className}
        `}
        {...props}
      >
        {dot && (
          <span className={`w-1.5 h-1.5 rounded-full ${dotColorClass}`} />
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';

const categoryColors: Record<string, string> = {
  Japanese: 'bg-[rgba(244,114,182,0.12)] text-[#db2777]',
  Chinese: 'bg-[rgba(251,146,60,0.12)] text-[#ea580c]',
  Korean: 'bg-[rgba(167,139,250,0.12)] text-[#7c3aed]',
  Thai: 'bg-[rgba(52,211,153,0.12)] text-[#059669]',
  Italian: 'bg-[rgba(248,113,113,0.12)] text-[#dc2626]',
  French: 'bg-[rgba(96,165,250,0.12)] text-[#2563eb]',
  Mediterranean: 'bg-[rgba(45,212,191,0.12)] text-[#0d9488]',
  Vegetarian: 'bg-[rgba(74,222,128,0.12)] text-[#16a34a]',
  International: 'bg-[rgba(251,191,36,0.12)] text-[#d97706]',
};

export function CategoryBadge({ category, className = '' }: { category: string; className?: string }) {
  const colorClass = categoryColors[category] || 'bg-[rgba(123,61,44,0.08)] text-[#8b6e62]';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-[11px] font-medium rounded-md ${colorClass} ${className}`}>
      {category}
    </span>
  );
}
