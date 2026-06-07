'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  Filter,
  TrendingUp,
  DollarSign,
  MapPin,
  Shield,
  Star,
  ChevronRight,
  Info,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge, CategoryBadge } from '@/components/ui/Badge';
import { ScoreBreakdown } from '@/components/ui/ScoreBar';
import { MerchantCard } from '@/components/features/MerchantCard';
import { useStore } from '@/lib/store';
import { useTranslation } from '@/lib/LanguageContext';
import type { MatchResult, Offer } from '@/types';

type SortOption = 'best' | 'cheapest' | 'stable' | 'environment';

export default function MatchResultsPage() {
  const router = useRouter();
  const t = useTranslation();
  const { matchResults, currentOffers, setSelectedMatch, setSelectedOffer } = useStore();
  const [selectedMatchIndex, setSelectedMatchIndex] = useState(0);
  const [sortBy, setSortBy] = useState<SortOption>('best');
  const [sortedResults, setSortedResults] = useState<MatchResult[]>([]);

  useEffect(() => {
    if (matchResults.length === 0) {
      const generateDemoMatches = async () => {
        const { matchMerchants } = await import('@/lib/matching');
        const { generateOffers } = await import('@/lib/offer-generator');
        const { merchants } = await import('@/lib/mock-data');
        const { parseDemand } = await import('@/lib/parser');

        const sampleDemand = parseDemand('今晚7点，4人，预算400，安静，不要辣，要包间');
        const matches = matchMerchants(merchants, sampleDemand);
        const demandId = `demand_${Date.now()}`;
        const offers = generateOffers(matches, sampleDemand, demandId);

        useStore.getState().setMatchResults(matches);
        useStore.getState().setCurrentOffers(offers);
        useStore.getState().setCurrentDemand(sampleDemand);
      };

      generateDemoMatches();
    }
  }, [matchResults.length]);

  useEffect(() => {
    if (matchResults.length === 0) return;

    const sorted = [...matchResults];
    switch (sortBy) {
      case 'best':
        sorted.sort((a, b) => b.finalScore - a.finalScore);
        break;
      case 'cheapest':
        sorted.sort((a, b) => {
          const aOffer = currentOffers.find((o) => o.merchantId === a.merchant.id);
          const bOffer = currentOffers.find((o) => o.merchantId === b.merchant.id);
          return (aOffer?.price || 0) - (bOffer?.price || 0);
        });
        break;
      case 'stable':
        sorted.sort((a, b) => b.scores.fulfillment - a.scores.fulfillment);
        break;
      case 'environment':
        sorted.sort((a, b) => b.scores.demandFit - a.scores.demandFit);
        break;
      default:
        break;
    }

    setSortedResults(sorted);
    setSelectedMatchIndex(0);
  }, [sortBy, matchResults, currentOffers]);

  const selectedMatch = sortedResults[selectedMatchIndex];
  const selectedOffer = selectedMatch
    ? currentOffers.find((o) => o.merchantId === selectedMatch.merchant.id)
    : null;

  const handleSelectMatch = (index: number) => {
    setSelectedMatchIndex(index);
    setSelectedMatch(sortedResults[index]);
  };

  const handleAcceptOffer = (offer: Offer) => {
    setSelectedOffer(offer);
    router.push(`/contract/${offer.id}`);
  };

  const sortOptions = [
    { value: 'best', label: t.matches.sortBest },
    { value: 'cheapest', label: t.matches.sortCheapest },
    { value: 'stable', label: t.matches.sortStable },
    { value: 'environment', label: t.matches.sortEnvironment },
  ];

  if (matchResults.length === 0) {
    return (
      <div className="max-w-5xl mx-auto flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-16 h-16 rounded-2xl bg-[rgba(231,135,69,0.1)] flex items-center justify-center mx-auto mb-4 animate-pulse">
            <TrendingUp className="w-8 h-8 text-[#e78745]" />
          </div>
          <h2 className="text-xl font-semibold text-[#7b3d2c]">
            {t.matches.analyzingMatches}
          </h2>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-10">
        <div>
          <button
            onClick={() => router.back()}
            className="flex items-center gap-2 text-[#9a7e72] hover:text-[#7b3d2c] transition-colors mb-3"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="text-base">{t.common.back}</span>
          </button>
          <h1 className="text-4xl font-normal text-[#7b3d2c] font-serif">{t.matches.title}</h1>
          <p className="text-base text-[#9a7e72] mt-2">
            {t.matches.subtitle.replace('{count}', sortedResults.length.toString())}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Filter className="w-5 h-5 text-[#9a7e72]" />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortOption)}
            className="px-4 py-2.5 rounded-xl bg-white border border-[rgba(123,61,44,0.08)] text-[#7b3d2c] text-base focus:border-[#e78745] transition-colors"
          >
            {sortOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto pr-2 hide-scrollbar">
          {sortedResults.map((match, index) => (
            <MerchantCard
              key={match.merchant.id}
              match={match}
              isSelected={index === selectedMatchIndex}
              onClick={() => handleSelectMatch(index)}
              compact
            />
          ))}
        </div>

        <div className="lg:col-span-2 space-y-8">
          {selectedMatch && (
            <>
              <Card padding="lg">
                <CardContent className="p-8">
                  <div className="flex items-start justify-between mb-8">
                    <div className="flex items-start gap-5">
                      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[rgba(123,61,44,0.1)] to-[rgba(123,61,44,0.03)] flex items-center justify-center flex-shrink-0">
                        <span className="text-3xl font-normal text-[#7b3d2c] font-serif">
                          #{selectedMatch.rank}
                        </span>
                      </div>
                      <div>
                        <div className="flex items-center gap-3 mb-3">
                          <h2 className="text-2xl font-normal text-[#7b3d2c] font-serif">
                            {selectedMatch.merchant.name}
                          </h2>
                          {selectedMatch.isRecommended && (
                            <Badge variant="primary">{t.matches.recommended}</Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-5 text-base text-[#9a7e72]">
                          <CategoryBadge category={selectedMatch.merchant.cuisine} />
                          <span className="flex items-center gap-1.5">
                            <Star className="w-4 h-4 fill-[#e78745] text-[#e78745]" />
                            {selectedMatch.merchant.rating}
                          </span>
                          <span className="flex items-center gap-1.5">
                            <MapPin className="w-4 h-4" />
                            {selectedMatch.merchant.location.split(',')[0]}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-4xl font-normal text-[#7b3d2c] font-serif">
                        {selectedMatch.finalScore.toFixed(1)}
                      </div>
                      <div className="text-base text-[#9a7e72]">
                        {t.matches.matchScore}
                      </div>
                    </div>
                  </div>

                  <div className="mb-8">
                    <h3 className="text-base font-medium text-[#7b3d2c] mb-4 flex items-center gap-2">
                      <Info className="w-5 h-5 text-[#e78745]" />
                      {t.matches.scoreBreakdown}
                    </h3>
                    <ScoreBreakdown
                      scores={[
                        { label: t.create.demandFit, value: selectedMatch.scores.demandFit, weight: 0.3 },
                        { label: t.create.fulfillmentRate, value: selectedMatch.scores.fulfillment, weight: 0.25 },
                        { label: t.create.supplyIdleScore, value: selectedMatch.scores.supplyIdle, weight: 0.2 },
                        { label: t.create.priceScore, value: selectedMatch.scores.price, weight: 0.15 },
                        { label: t.create.distanceScore, value: selectedMatch.scores.distance, weight: 0.1 },
                      ]}
                    />
                  </div>

                  <div className="p-5 rounded-xl bg-[rgba(231,135,69,0.05)] border border-[rgba(231,135,69,0.1)]">
                    <div className="flex items-start gap-4">
                      <Shield className="w-6 h-6 text-[#e78745] mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="text-base font-medium text-[#7b3d2c] mb-2">
                          {t.matches.whyMatch}
                        </div>
                        <p className="text-base text-[#9a7e72] leading-loose">
                          {selectedMatch.explanation}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {selectedOffer && (
                <Card className="border-[rgba(34,197,94,0.15)]" padding="lg">
                  <CardContent className="p-8">
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="text-xl font-normal text-[#7b3d2c] flex items-center gap-3 font-serif">
                        <DollarSign className="w-6 h-6 text-[#22c55e]" />
                        {t.matches.offer}
                      </h3>
                      <Badge variant="success">
                        {t.matches.validFor}
                      </Badge>
                    </div>

                    <div className="mb-6">
                      <div className="text-4xl font-normal text-[#7b3d2c] font-serif">
                        ¥{selectedOffer.price}
                        <span className="text-lg font-normal text-[#9a7e72] ml-2">
                          {selectedOffer.priceType === 'per_person' ? t.matches.perPerson : t.matches.total}
                        </span>
                      </div>
                    </div>

                    <div className="space-y-4 mb-6">
                      <h4 className="text-base font-medium text-[#7b3d2c]">
                        {t.matches.bindingPromises}
                      </h4>
                      {selectedOffer.promises.slice(0, 3).map((promise, index) => (
                        <div
                          key={index}
                          className="flex items-start gap-4 p-4 rounded-xl bg-[rgba(123,61,44,0.03)]"
                        >
                          <div className="w-6 h-6 rounded-full bg-[rgba(34,197,94,0.1)] flex items-center justify-center flex-shrink-0 mt-0.5">
                            <Shield className="w-4 h-4 text-[#22c55e]" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-base text-[#7b3d2c] leading-relaxed">
                              {promise.description}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <Button
                      className="w-full"
                      size="md"
                      onClick={() => handleAcceptOffer(selectedOffer)}
                      rightIcon={<ChevronRight className="w-5 h-5" />}
                    >
                      {t.matches.acceptOffer}
                    </Button>
                  </CardContent>
                </Card>
              )}

              <Card padding="lg">
                <CardContent className="p-8">
                  <h3 className="text-base font-medium text-[#7b3d2c] mb-6">
                    {t.matches.riskAssessment}
                  </h3>
                  <div className="grid grid-cols-4 gap-4">
                    <div className="text-center p-5 rounded-xl bg-[rgba(123,61,44,0.03)]">
                      <div
                        className={`text-2xl font-normal ${
                          selectedMatch.riskLevel === 'low'
                            ? 'text-[#22c55e]'
                            : selectedMatch.riskLevel === 'medium'
                            ? 'text-[#f59e0b]'
                            : 'text-[#ef4444]'
                        }`}
                      >
                        {selectedMatch.riskLevel === 'low'
                          ? t.matches.lowRisk
                          : selectedMatch.riskLevel === 'medium'
                          ? t.matches.mediumRisk
                          : t.matches.highRisk}
                      </div>
                      <div className="text-sm text-[#9a7e72] mt-2">
                        {t.matches.riskAssessment}
                      </div>
                    </div>
                    <div className="text-center p-5 rounded-xl bg-[rgba(123,61,44,0.03)]">
                      <div className="text-2xl font-normal text-[#7b3d2c]">
                        {selectedMatch.merchant.metrics.fulfillmentRate}%
                      </div>
                      <div className="text-sm text-[#9a7e72] mt-2">
                        {t.create.fulfillmentRate}
                      </div>
                    </div>
                    <div className="text-center p-5 rounded-xl bg-[rgba(123,61,44,0.03)]">
                      <div className="text-2xl font-normal text-[#7b3d2c]">
                        {selectedMatch.merchant.metrics.breachRate}%
                      </div>
                      <div className="text-sm text-[#9a7e72] mt-2">
                        {t.merchant?.breachRate || '违约率'}
                      </div>
                    </div>
                    <div className="text-center p-5 rounded-xl bg-[rgba(123,61,44,0.03)]">
                      <div className="text-2xl font-normal text-[#7b3d2c]">
                        {selectedMatch.merchant.currentStatus.queueTime}m
                      </div>
                      <div className="text-sm text-[#9a7e72] mt-2">
                        {t.matches.waitTime}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
