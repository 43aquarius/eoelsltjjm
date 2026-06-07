'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Sparkles,
  Send,
  ChevronRight,
  AlertCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { useStore } from '@/lib/store';
import { useTranslation } from '@/lib/LanguageContext';
import { parseDemand, formatParsedDemand } from '@/lib/parser';

export default function CreateDemandPage() {
  const router = useRouter();
  const t = useTranslation();
  const { demandInput, setDemandInput, setCurrentDemand, setMatchResults, setCurrentOffers } = useStore();
  const [parsedPreview, setParsedPreview] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    if (demandInput.trim()) {
      const parsed = parseDemand(demandInput);
      setParsedPreview(formatParsedDemand(parsed));
    } else {
      setParsedPreview('');
    }
  }, [demandInput]);

  const handleGenerateMatches = () => {
    setIsAnalyzing(true);
    const parsed = parseDemand(demandInput);
    setCurrentDemand(parsed);

    setTimeout(async () => {
      const { matchMerchants } = await import('@/lib/matching');
      const { generateOffers } = await import('@/lib/offer-generator');
      const { merchants } = await import('@/lib/mock-data');

      const matches = matchMerchants(merchants, parsed);
      setMatchResults(matches);

      const demandId = `demand_${Date.now()}`;
      const offers = generateOffers(matches, parsed, demandId);
      setCurrentOffers(offers);

      setIsAnalyzing(false);
      router.push('/matches');
    }, 1500);
  };

  const handleSampleDemand = (sample: string) => {
    setDemandInput(sample);
  };

  const chineseSampleDemands = [
    "今晚7点，4人，预算400，安静，不要辣，要包间",
    "明天午餐，2人，想吃五道口附近的寿司，人均200",
    "周五晚上，6人庆祝，需要包间，偏好中餐",
    "快速午餐1人，预算50，清淡健康",
    "约会晚餐，2人，浪漫氛围，西餐优先",
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-12">
        <h1 className="text-4xl font-normal text-[#7b3d2c] mb-4 font-serif">{t.create.title}</h1>
        <p className="text-lg text-[#9a7e72] leading-loose">
          {t.create.subtitle}
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-8">
          <Card padding="lg">
            <CardContent className="p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 rounded-xl bg-[rgba(231,135,69,0.1)] flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-[#e78745]" />
                </div>
                <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
                  {t.create.naturalInput}
                </h2>
              </div>

              <textarea
                value={demandInput}
                onChange={(e) => setDemandInput(e.target.value)}
                placeholder={t.create.placeholder}
                className="w-full h-40 px-5 py-4 rounded-xl bg-[rgba(123,61,44,0.03)] border border-[rgba(123,61,44,0.08)] text-[#7b3d2c] text-base placeholder-[#d4c8be] resize-none focus:border-[#e78745] focus:ring-4 focus:ring-[rgba(231,135,69,0.1)] transition-all leading-loose"
              />

              <div className="flex items-center justify-between mt-6">
                <div className="text-base text-[#9a7e72]">
                  {t.create.tryNatural}
                </div>
                <Button
                  onClick={handleGenerateMatches}
                  isLoading={isAnalyzing}
                  disabled={!demandInput.trim()}
                  rightIcon={<Send className="w-4 h-4" />}
                  size="md"
                >
                  {isAnalyzing ? t.create.analyzing : t.create.generateMatches}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card padding="lg">
            <CardContent className="p-8">
              <h3 className="text-lg font-normal text-[#7b3d2c] mb-6 font-serif">
                {t.create.quickExamples}
              </h3>
              <div className="space-y-3">
                {chineseSampleDemands.map((sample, index) => (
                  <button
                    key={index}
                    onClick={() => handleSampleDemand(sample)}
                    className="w-full text-left px-5 py-4 rounded-xl bg-[rgba(123,61,44,0.02)] hover:bg-[rgba(123,61,44,0.06)] border border-[rgba(123,61,44,0.06)] transition-colors group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-base text-[#7b3d2c] leading-relaxed">{sample}</span>
                      <ChevronRight className="w-5 h-5 text-[#d4c8be] group-hover:text-[#9a7e72] transition-colors flex-shrink-0 ml-2" />
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-8">
          <Card padding="lg">
            <CardContent className="p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-3.5 h-3.5 rounded-full bg-[#22c55e] animate-pulse" />
                <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
                  {t.create.livePreview}
                </h2>
              </div>

              {parsedPreview ? (
                <div className="space-y-6">
                  <pre className="text-base text-[#7b3d2c] whitespace-pre-wrap font-sans leading-loose bg-[rgba(123,61,44,0.02)] p-5 rounded-xl">
                    {parsedPreview}
                  </pre>

                  <div className="flex flex-wrap gap-3">
                    {(demandInput.includes('安静') || demandInput.toLowerCase().includes('quiet')) && (
                      <Badge variant="info">{t.create.quiet}</Badge>
                    )}
                    {(demandInput.includes('包间') || demandInput.toLowerCase().includes('private')) && (
                      <Badge variant="primary">{t.create.privateRoom}</Badge>
                    )}
                    {(demandInput.includes('辣') || demandInput.toLowerCase().includes('spice')) && (
                      <Badge variant="warning">{t.create.noSpice}</Badge>
                    )}
                    {(demandInput.includes('素') || demandInput.toLowerCase().includes('vegetarian')) && (
                      <Badge variant="success">{t.create.vegetarian}</Badge>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-[#9a7e72]">
                  <AlertCircle className="w-10 h-10 mx-auto mb-4 opacity-40" />
                  <p className="text-base">{t.create.enterDemand}</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card padding="lg">
            <CardContent className="p-8">
              <h3 className="text-lg font-normal text-[#7b3d2c] mb-6 font-serif">
                {t.create.matchingAlgorithm}
              </h3>

              <div className="space-y-4">
                {[
                  { label: t.create.demandFit, weight: '30%', color: '#7b3d2c' },
                  { label: t.create.fulfillmentRate, weight: '25%', color: '#22c55e' },
                  { label: t.create.supplyIdleScore, weight: '20%', color: '#e78745' },
                  { label: t.create.priceScore, weight: '15%', color: '#f59e0b' },
                  { label: t.create.distanceScore, weight: '10%', color: '#9a7e72' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-4">
                    <div
                      className="w-4 h-4 rounded-full flex-shrink-0"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="flex-1 text-base text-[#7b3d2c]">
                      {item.label}
                    </span>
                    <span className="text-sm text-[#b8a89c] font-medium">
                      {item.weight}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-[rgba(231,135,69,0.05)] to-transparent border-[rgba(231,135,69,0.1)]" padding="lg">
            <CardContent className="p-8">
              <h3 className="text-lg font-normal text-[#7b3d2c] mb-6 font-serif">
                {t.create.howItWorksTitle}
              </h3>
              <ul className="space-y-4 text-base text-[#9a7e72] leading-loose">
                <li className="flex items-start gap-3">
                  <span className="text-[#e78745] font-medium">1.</span>
                  <span>{t.create.howItWorks1}</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#e78745] font-medium">2.</span>
                  <span>{t.create.howItWorks2}</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#e78745] font-medium">3.</span>
                  <span>{t.create.howItWorks3}</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#e78745] font-medium">4.</span>
                  <span>{t.create.howItWorks4}</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
