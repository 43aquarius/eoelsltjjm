'use client';

import { type ReactNode } from 'react';
import { Card, CardContent } from './Card';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    trend: 'up' | 'down' | 'neutral';
  };
  icon?: ReactNode;
  subtitle?: string;
  className?: string;
}

export function MetricCard({
  title,
  value,
  change,
  icon,
  subtitle,
  className = '',
}: MetricCardProps) {
  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      if (val >= 1000000) {
        return `${(val / 1000000).toFixed(1)}M`;
      }
      if (val >= 1000) {
        return `${(val / 1000).toFixed(1)}K`;
      }
      return val.toLocaleString();
    }
    return val;
  };

  return (
    <Card padding="sm" className={className}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-xs text-[#a89588] mb-1">{title}</p>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-semibold text-[#7b3d2c]">
                {formatValue(value)}
              </span>
              {change && (
                <span
                  className={`text-[10px] font-medium ${
                    change.trend === 'up'
                      ? 'text-[#16a34a]'
                      : change.trend === 'down'
                      ? 'text-[#dc2626]'
                      : 'text-[#8b6e62]'
                  }`}
                >
                  {change.trend === 'up' && '↑'}
                  {change.trend === 'down' && '↓'}
                  {change.trend === 'neutral' && '→'}
                  {Math.abs(change.value)}%
                </span>
              )}
            </div>
            {subtitle && (
              <p className="text-[10px] text-[#c4b5a9] mt-1">{subtitle}</p>
            )}
          </div>
          {icon && (
            <div className="p-2 rounded-lg bg-[rgba(231,135,69,0.1)] text-[#e78745]">
              {icon}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  label?: string;
}

export function ProgressRing({
  progress,
  size = 80,
  strokeWidth = 6,
  color = '#e78745',
  label,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(123,61,44,0.08)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-base font-semibold text-[#7b3d2c]">
          {progress.toFixed(0)}%
        </span>
        {label && (
          <span className="text-[10px] text-[#a89588]">{label}</span>
        )}
      </div>
    </div>
  );
}

interface BarChartProps {
  data: Array<{ label: string; value: number; color?: string }>;
  maxValue?: number;
  height?: number;
}

export function BarChart({
  data,
  maxValue,
  height = 120,
}: BarChartProps) {
  const max = maxValue || Math.max(...data.map((d) => d.value));

  return (
    <div
      className="flex items-end justify-between gap-2"
      style={{ height }}
    >
      {data.map((item, index) => {
        const barHeight = (item.value / max) * 100;
        return (
          <div key={index} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full rounded-t transition-all duration-300"
              style={{
                height: `${barHeight}%`,
                backgroundColor: item.color || '#e78745',
                minHeight: 4,
              }}
            />
            <span className="text-[10px] text-[#a89588] truncate w-full text-center">
              {item.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
