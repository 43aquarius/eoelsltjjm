'use client';

import { Star, MapPin, Users, Clock, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge, CategoryBadge } from '@/components/ui/Badge';
import { ScoreBar } from '@/components/ui/ScoreBar';
import type { MatchResult } from '@/types';

interface MerchantCardProps {
  match: MatchResult;
  isSelected?: boolean;
  onClick?: () => void;
  compact?: boolean;
}

export function MerchantCard({
  match,
  isSelected = false,
  onClick,
  compact = false,
}: MerchantCardProps) {
  const { merchant, scores, finalScore, rank, isRecommended, riskLevel, explanation } = match;

  const getRiskBadge = () => {
    switch (riskLevel) {
      case 'low':
        return <Badge variant="success">Low Risk</Badge>;
      case 'medium':
        return <Badge variant="warning">Medium Risk</Badge>;
      case 'high':
        return <Badge variant="error">High Risk</Badge>;
    }
  };

  const getAvailabilityColor = (value: number) => {
    if (value >= 70) return 'text-[#16a34a]';
    if (value >= 40) return 'text-[#d97706]';
    return 'text-[#dc2626]';
  };

  if (compact) {
    return (
      <Card
        padding="sm"
        hover
        onClick={onClick}
        className={`cursor-pointer transition-all ${
          isSelected ? 'ring-1 ring-[#7b3d2c] border-[#7b3d2c]' : ''
        }`}
      >
        <CardContent className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-[rgba(123,61,44,0.1)] flex items-center justify-center text-[#7b3d2c] font-medium text-xs">
                #{rank}
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-[#7b3d2c]">{merchant.name}</span>
                  {isRecommended && <Badge variant="primary">Recommended</Badge>}
                </div>
                <div className="flex items-center gap-2 text-[10px] text-[#a89588]">
                  <span>{merchant.cuisine}</span>
                  <span>·</span>
                  <span className="flex items-center gap-0.5">
                    <Star className="w-2.5 h-2.5 fill-[#e78745] text-[#e78745]" />
                    {merchant.rating}
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs font-medium text-[#7b3d2c]">{finalScore.toFixed(1)}</div>
              <div className="text-[10px] text-[#c4b5a9]">score</div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      padding="sm"
      hover
      onClick={onClick}
      className={`cursor-pointer transition-all animate-fadeIn ${
        isSelected ? 'ring-1 ring-[#7b3d2c] border-[#7b3d2c]' : ''
      }`}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[rgba(123,61,44,0.12)] to-[rgba(123,61,44,0.05)] flex items-center justify-center">
              <span className="text-base font-semibold text-[#7b3d2c]">#{rank}</span>
            </div>
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <h3 className="text-sm font-medium text-[#7b3d2c]">{merchant.name}</h3>
                {isRecommended && <Badge variant="primary">Recommended</Badge>}
              </div>
              <div className="flex items-center gap-2 text-[11px] text-[#a89588]">
                <CategoryBadge category={merchant.cuisine} />
                <span className="flex items-center gap-0.5">
                  <Star className="w-3 h-3 fill-[#e78745] text-[#e78745]" />
                  {merchant.rating}
                </span>
                <span className="flex items-center gap-0.5">
                  <MapPin className="w-3 h-3" />
                  {merchant.location.split(',')[0]}
                </span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-lg font-semibold text-[#7b3d2c]">{finalScore.toFixed(1)}</div>
            <div className="text-[10px] text-[#c4b5a9]">match score</div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="text-center p-1.5 rounded-lg bg-[rgba(123,61,44,0.04)]">
            <Users className={`w-3.5 h-3.5 mx-auto mb-0.5 ${getAvailabilityColor(merchant.currentStatus.seatAvailability)}`} />
            <div className="text-[11px] font-medium text-[#7b3d2c]">{merchant.currentStatus.seatAvailability}%</div>
            <div className="text-[9px] text-[#c4b5a9]">Seats</div>
          </div>
          <div className="text-center p-1.5 rounded-lg bg-[rgba(123,61,44,0.04)]">
            <div className={`text-[11px] font-medium mb-0.5 ${getAvailabilityColor(100 - merchant.currentStatus.kitchenLoad)}`}>
              {merchant.currentStatus.kitchenLoad}%
            </div>
            <div className="text-[9px] text-[#c4b5a9]">Kitchen</div>
          </div>
          <div className="text-center p-1.5 rounded-lg bg-[rgba(123,61,44,0.04)]">
            <Clock className={`w-3.5 h-3.5 mx-auto mb-0.5 ${merchant.currentStatus.queueTime > 20 ? 'text-[#dc2626]' : 'text-[#16a34a]'}`} />
            <div className="text-[11px] font-medium text-[#7b3d2c]">{merchant.currentStatus.queueTime}m</div>
            <div className="text-[9px] text-[#c4b5a9]">Queue</div>
          </div>
          <div className="text-center p-1.5 rounded-lg bg-[rgba(123,61,44,0.04)]">
            <div className={`text-[11px] font-medium ${getAvailabilityColor(merchant.metrics.fulfillmentRate)}`}>
              {merchant.metrics.fulfillmentRate}%
            </div>
            <div className="text-[9px] text-[#c4b5a9]">Fulfill</div>
          </div>
        </div>

        <div className="space-y-1.5 mb-3">
          <ScoreBar score={scores.demandFit} label="Demand Fit" size="sm" />
          <ScoreBar score={scores.fulfillment} label="Reliability" size="sm" />
          <ScoreBar score={scores.supplyIdle} label="Availability" size="sm" />
          <ScoreBar score={scores.price} label="Price" size="sm" />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getRiskBadge()}
            <span className="text-[11px] text-[#a89588]">{explanation}</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-[#c4b5a9]" />
        </div>
      </CardContent>
    </Card>
  );
}
