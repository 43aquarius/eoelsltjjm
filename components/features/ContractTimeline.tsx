'use client';

import { Check, X, AlertCircle, Clock, Circle } from 'lucide-react';
import type { ContractEvent, ContractStatus } from '@/types';

interface ContractTimelineProps {
  events: ContractEvent[];
  currentStatus: ContractStatus;
}

const statusConfig: Record<
  ContractStatus,
  { icon: React.ElementType; color: string; bgColor: string }
> = {
  draft: { icon: Circle, color: 'text-[#a89588]', bgColor: 'bg-[rgba(123,61,44,0.06)]' },
  confirmed: { icon: Check, color: 'text-[#7b3d2c]', bgColor: 'bg-[rgba(123,61,44,0.12)]' },
  merchant_accepted: { icon: Check, color: 'text-[#e78745]', bgColor: 'bg-[rgba(231,135,69,0.12)]' },
  user_arrived: { icon: Check, color: 'text-[#16a34a]', bgColor: 'bg-[rgba(74,222,128,0.12)]' },
  service_started: { icon: Clock, color: 'text-[#d97706]', bgColor: 'bg-[rgba(251,191,36,0.12)]' },
  completed: { icon: Check, color: 'text-[#16a34a]', bgColor: 'bg-[rgba(74,222,128,0.12)]' },
  breached: { icon: X, color: 'text-[#dc2626]', bgColor: 'bg-[rgba(248,113,113,0.12)]' },
  compensated: { icon: AlertCircle, color: 'text-[#fb923c]', bgColor: 'bg-[rgba(251,146,60,0.12)]' },
  cancelled: { icon: X, color: 'text-[#a89588]', bgColor: 'bg-[rgba(123,61,44,0.06)]' },
};

export function ContractTimeline({ events, currentStatus }: ContractTimelineProps) {
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  const groupedEvents = events.reduce((acc, event) => {
    const date = formatDate(event.timestamp);
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(event);
    return acc;
  }, {} as Record<string, ContractEvent[]>);

  return (
    <div className="relative">
      {Object.entries(groupedEvents).map(([date, dateEvents], dateIndex) => (
        <div key={date} className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="text-[10px] font-medium text-[#c4b5a9]">{date}</div>
            <div className="flex-1 h-px bg-[rgba(123,61,44,0.08)]" />
          </div>

          <div className="space-y-2">
            {dateEvents.map((event, eventIndex) => {
              const config = statusConfig[event.status];
              const Icon = config.icon;
              const isLast = dateIndex === Object.keys(groupedEvents).length - 1 &&
                eventIndex === dateEvents.length - 1;

              return (
                <div key={event.timestamp} className="relative flex gap-2.5">
                  {!isLast && (
                    <div className="absolute left-[9px] top-5 w-0.5 h-full bg-[rgba(123,61,44,0.08)]" />
                  )}

                  <div
                    className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center ${config.bgColor}`}
                  >
                    <Icon className={`w-2.5 h-2.5 ${config.color}`} />
                  </div>

                  <div className="flex-1 pb-2">
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-xs text-[#7b3d2c]">{event.description}</span>
                      <span className="text-[10px] text-[#c4b5a9]">{formatTime(event.timestamp)}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded ${config.bgColor} ${config.color}`}
                      >
                        {event.actor}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

interface ProgressStepsProps {
  currentStatus: ContractStatus;
  onAdvance?: () => void;
}

const steps: ContractStatus[] = [
  'draft',
  'confirmed',
  'merchant_accepted',
  'user_arrived',
  'service_started',
  'completed',
];

export function ProgressSteps({ currentStatus, onAdvance }: ProgressStepsProps) {
  const currentIndex = steps.indexOf(currentStatus);
  const progress = ((currentIndex + 1) / steps.length) * 100;

  const getStepStatus = (index: number) => {
    if (index < currentIndex) return 'completed';
    if (index === currentIndex) return 'current';
    return 'upcoming';
  };

  return (
    <div className="relative">
      <div className="absolute top-3.5 left-0 right-0 h-0.5 bg-[rgba(123,61,44,0.08)]">
        <div
          className="h-full bg-[#7b3d2c] transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="relative flex justify-between">
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          const config = statusConfig[step];
          const Icon = config.icon;

          return (
            <div key={step} className="flex flex-col items-center">
              <div
                className={`
                  w-7 h-7 rounded-full flex items-center justify-center mb-1.5
                  transition-all duration-300
                  ${
                    status === 'completed'
                      ? 'bg-[#16a34a]'
                      : status === 'current'
                      ? 'bg-[#7b3d2c] ring-4 ring-[rgba(123,61,44,0.15)]'
                      : 'bg-[rgba(123,61,44,0.06)]'
                  }
                `}
              >
                {status === 'completed' ? (
                  <Check className="w-3.5 h-3.5 text-white" />
                ) : (
                  <Icon
                    className={`w-3.5 h-3.5 ${
                      status === 'current' ? 'text-white' : 'text-[#c4b5a9]'
                    }`}
                  />
                )}
              </div>
              <span
                className={`text-[10px] capitalize ${
                  status === 'current' ? 'text-[#7b3d2c] font-medium' : 'text-[#c4b5a9]'
                }`}
              >
                {step.replace('_', ' ')}
              </span>
            </div>
          );
        })}
      </div>

      {onAdvance && currentStatus !== 'completed' && currentStatus !== 'cancelled' && (
        <button
          onClick={onAdvance}
          className="mt-4 w-full py-1.5 px-3 rounded-lg bg-[rgba(123,61,44,0.08)] text-[#7b3d2c] text-xs font-medium hover:bg-[rgba(123,61,44,0.12)] transition-colors"
        >
          Advance to Next Step →
        </button>
      )}
    </div>
  );
}
