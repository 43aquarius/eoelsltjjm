'use client';

interface ScoreBarProps {
  score: number;
  maxScore?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  colorThresholds?: {
    low: number;
    medium: number;
    high: number;
  };
  className?: string;
}

const sizeStyles = {
  sm: 'h-1',
  md: 'h-1.5',
  lg: 'h-2.5',
};

export function ScoreBar({
  score,
  maxScore = 100,
  size = 'md',
  showLabel = true,
  label,
  colorThresholds = { low: 40, medium: 70, high: 90 },
  className = '',
}: ScoreBarProps) {
  const percentage = Math.min((score / maxScore) * 100, 100);

  const getColor = (score: number) => {
    if (score >= colorThresholds.high) return '#16a34a';
    if (score >= colorThresholds.medium) return '#e78745';
    if (score >= colorThresholds.low) return '#d97706';
    return '#dc2626';
  };

  const color = getColor(score);

  return (
    <div className={`w-full ${className}`}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <span className="text-[11px] text-[#8b6e62]">{label}</span>
          )}
          {showLabel && (
            <span className="text-[11px] font-medium" style={{ color }}>
              {score.toFixed(1)}
            </span>
          )}
        </div>
      )}
      <div
        className={`w-full bg-[rgba(123,61,44,0.08)] rounded-full overflow-hidden ${sizeStyles[size]}`}
      >
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
  );
}

interface ScoreBreakdownProps {
  scores: Array<{
    label: string;
    value: number;
    maxValue?: number;
    weight?: number;
  }>;
  className?: string;
}

export function ScoreBreakdown({ scores, className = '' }: ScoreBreakdownProps) {
  return (
    <div className={`space-y-2.5 ${className}`}>
      {scores.map((score, index) => (
        <div key={index}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] text-[#a89588]">{score.label}</span>
            <div className="flex items-center gap-2">
              {score.weight && (
                <span className="text-[10px] text-[#c4b5a9]">
                  ×{score.weight.toFixed(2)}
                </span>
              )}
              <span className="text-[11px] font-medium text-[#7b3d2c]">
                {score.value.toFixed(1)}
              </span>
            </div>
          </div>
          <ScoreBar
            score={score.value}
            maxScore={score.maxValue || 100}
            size="sm"
            showLabel={false}
          />
        </div>
      ))}
    </div>
  );
}
