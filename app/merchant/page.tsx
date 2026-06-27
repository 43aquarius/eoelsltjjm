'use client';

import { useState } from 'react';
import {
  Users,
  Clock,
  TrendingUp,
  CheckCircle,
  XCircle,
  Activity,
} from 'lucide-react';
import { useTranslation } from '@/lib/LanguageContext';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

type MetricColor = 'green' | 'yellow' | 'red';

export default function MerchantPage() {
  const t = useTranslation();
  const [seatAvailability] = useState(75);
  const [kitchenLoad] = useState(60);
  const [queueTime] = useState(15);

  const metrics: Array<{
    label: string;
    value: string;
    icon: typeof Users;
    color: MetricColor;
  }> = [
    {
      label: t.merchant?.seatAvailability || '座位可用性',
      value: `${seatAvailability}%`,
      icon: Users,
      color: seatAvailability > 70 ? 'green' : seatAvailability > 40 ? 'yellow' : 'red',
    },
    {
      label: t.merchant?.kitchenLoad || '厨房负载',
      value: `${kitchenLoad}%`,
      icon: TrendingUp,
      color: kitchenLoad < 70 ? 'green' : kitchenLoad < 90 ? 'yellow' : 'red',
    },
    {
      label: '排队时间',
      value: `${queueTime}分钟`,
      icon: Clock,
      color: queueTime < 20 ? 'green' : queueTime < 40 ? 'yellow' : 'red',
    },
  ];

  const opportunities = [
    {
      id: 1,
      customer: '张先生',
      demand: '今晚7点，4人，预算400',
      matchScore: 92,
      status: 'pending',
    },
    {
      id: 2,
      customer: '李女士',
      demand: '明天午餐，2人，人均150',
      matchScore: 87,
      status: 'pending',
    },
    {
      id: 3,
      customer: '王总',
      demand: '周五晚上，8人庆祝，需要包间',
      matchScore: 78,
      status: 'accepted',
    },
  ];

  const colorClasses: Record<MetricColor, string> = {
    green: 'text-[#16a34a]',
    yellow: 'text-[#d97706]',
    red: 'text-[#dc2626]',
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-16">
        <h1 className="text-5xl font-normal text-[#7b3d2c] mb-5 font-serif">
          {t.merchant?.title || '商家仪表板'}
        </h1>
        <p className="text-xl text-[#a89588] leading-loose">
          {t.merchant?.subtitle || '实时查看您的业务状态和合约机会'}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-10 mb-16">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <Card key={index} padding="lg">
              <CardContent className="p-8">
                <div className="flex items-center justify-between mb-5">
                  <span className="text-lg text-[#a89588]">{metric.label}</span>
                  <Icon className="w-6 h-6 text-[#c4b5a9]" />
                </div>
                <div className={`text-4xl font-normal ${colorClasses[metric.color]} font-serif`}>
                  {metric.value}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-3 gap-10 mb-16">
        <Card padding="lg">
          <CardContent className="p-8">
            <div className="text-lg text-[#a89588] mb-4">
              {t.merchant?.fulfillmentRate || '履约率'}
            </div>
            <div className="text-4xl font-normal text-[#16a34a] font-serif">98.5%</div>
          </CardContent>
        </Card>
        <Card padding="lg">
          <CardContent className="p-8">
            <div className="text-lg text-[#a89588] mb-4">
              {t.merchant?.customerSatisfaction || '客户满意度'}
            </div>
            <div className="text-4xl font-normal text-[#7b3d2c] font-serif">4.8/5.0</div>
          </CardContent>
        </Card>
        <Card padding="lg">
          <CardContent className="p-8">
            <div className="text-lg text-[#a89588] mb-4">
              {t.merchant?.breachRate || '违约率'}
            </div>
            <div className="text-4xl font-normal text-[#d97706] font-serif">0.3%</div>
          </CardContent>
        </Card>
      </div>

      <Card padding="lg">
        <CardContent className="p-10">
          <div className="flex items-center gap-5 mb-10">
            <Activity className="w-7 h-7 text-[#e78745]" />
            <h2 className="text-2xl font-normal text-[#7b3d2c] font-serif">
              {t.merchant?.contractOpportunities || '合约机会'}
            </h2>
          </div>
          <div className="space-y-6">
            {opportunities.map((opp) => (
              <div
                key={opp.id}
                className="flex items-center justify-between p-6 rounded-2xl bg-[rgba(123,61,44,0.03)] border border-[rgba(123,61,44,0.06)]"
              >
                <div className="flex-1 min-w-0 mr-6">
                  <div className="flex items-center gap-5 mb-4">
                    <span className="text-lg text-[#7b3d2c] font-medium">{opp.customer}</span>
                    <span
                      className={`px-3 py-1.5 rounded-xl text-base font-medium ${
                        opp.matchScore >= 85
                          ? 'bg-[rgba(74,222,128,0.12)] text-[#16a34a]'
                          : opp.matchScore >= 70
                          ? 'bg-[rgba(251,191,36,0.12)] text-[#d97706]'
                          : 'bg-[rgba(248,113,113,0.12)] text-[#dc2626]'
                      }`}
                    >
                      {opp.matchScore}%
                    </span>
                  </div>
                  <div className="text-lg text-[#a89588] truncate leading-relaxed">{opp.demand}</div>
                </div>
                <div className="flex items-center gap-4 flex-shrink-0">
                  {opp.status === 'pending' ? (
                    <>
                      <Button size="lg" variant="secondary">
                        <CheckCircle className="w-5 h-5" />
                        {t.merchant?.accept || '接受'}
                      </Button>
                      <Button size="lg" variant="danger">
                        <XCircle className="w-5 h-5" />
                        {t.merchant?.reject || '拒绝'}
                      </Button>
                    </>
                  ) : (
                    <span className="flex items-center gap-3 text-lg text-[#16a34a]">
                      <CheckCircle className="w-6 h-6" />
                      已接受
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
