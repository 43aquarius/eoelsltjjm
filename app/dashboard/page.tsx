'use client';

import {
  FileText,
  Users,
  TrendingUp,
  DollarSign,
  CheckCircle,
  AlertCircle,
  BarChart3,
} from 'lucide-react';
import { useTranslation } from '@/lib/LanguageContext';
import { Card, CardContent } from '@/components/ui/Card';

export default function DashboardPage() {
  const t = useTranslation();

  const stats = [
    {
      label: '活跃合约',
      value: '1,234',
      change: '+12%',
      icon: FileText,
      color: '#7b3d2c',
    },
    {
      label: '活跃商家',
      value: '8,456',
      change: '+8%',
      icon: Users,
      color: '#16a34a',
    },
    {
      label: t.dashboard?.revenueGenerated || '收入',
      value: '¥125,680',
      change: '+15%',
      icon: DollarSign,
      color: '#e78745',
    },
    {
      label: '用户满意度',
      value: '87.3',
      change: '+2.5%',
      icon: TrendingUp,
      color: '#8b5cf6',
    },
  ];

  const performance = [
    { label: '转化率', value: 78, target: 80 },
    { label: '履约率', value: 95, target: 98 },
    { label: '留存率', value: 92, target: 95 },
  ];

  const dailyTrend = [
    { day: '周一', contracts: 45 },
    { day: '周二', contracts: 52 },
    { day: '周三', contracts: 38 },
    { day: '周四', contracts: 65 },
    { day: '周五', contracts: 89 },
    { day: '周六', contracts: 95 },
    { day: '周日', contracts: 72 },
  ];

  const maxContracts = Math.max(...dailyTrend.map((d) => d.contracts));

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-16">
        <h1 className="text-5xl font-normal text-[#7b3d2c] mb-5 font-serif">
          {t.dashboard?.title || '平台仪表板'}
        </h1>
        <p className="text-xl text-[#a89588] leading-loose">
          {t.dashboard?.subtitle || '查看平台关键指标和性能数据'}
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-10 mb-16">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} padding="lg">
              <CardContent className="p-8">
                <div className="flex items-center justify-between mb-5">
                  <span className="text-lg text-[#a89588] truncate">{stat.label}</span>
                  <div
                    className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: `${stat.color}12` }}
                  >
                    <Icon className="w-7 h-7" style={{ color: stat.color }} />
                  </div>
                </div>
                <div className="text-4xl font-normal text-[#7b3d2c] mb-3 font-serif">{stat.value}</div>
                <div className="text-base text-[#16a34a] font-medium">{stat.change}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-16">
        <Card padding="lg">
          <CardContent className="p-8">
            <div className="flex items-center gap-5 mb-10">
              <BarChart3 className="w-7 h-7 text-[#e78745]" />
              <h2 className="text-2xl font-normal text-[#7b3d2c] font-serif">
                {t.dashboard?.performanceByCategory || '性能指标'}
              </h2>
            </div>
            <div className="space-y-6">
              {performance.map((perf, index) => {
                const percentage = (perf.value / perf.target) * 100;
                return (
                  <div key={index}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-lg text-[#a89588]">{perf.label}</span>
                      <span className="text-lg text-[#7b3d2c] font-medium">
                        {perf.value}% / {perf.target}%
                      </span>
                    </div>
                    <div className="h-3 bg-[rgba(123,61,44,0.06)] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${percentage}%`,
                          backgroundColor:
                            percentage >= 95 ? '#16a34a' : percentage >= 80 ? '#e78745' : '#dc2626',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card padding="lg">
          <CardContent className="p-8">
            <div className="flex items-center gap-5 mb-10">
              <TrendingUp className="w-7 h-7 text-[#e78745]" />
              <h2 className="text-2xl font-normal text-[#7b3d2c] font-serif">
                {t.dashboard?.dailyContracts || '每日合约趋势'}
              </h2>
            </div>
            <div className="flex items-end justify-between gap-4 h-48">
              {dailyTrend.map((day, index) => {
                const height = (day.contracts / maxContracts) * 100;
                return (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    <div className="w-full flex flex-col items-center justify-end h-40">
                      <div
                        className="w-full rounded-t-xl bg-[#7b3d2c] transition-all duration-300 hover:bg-[#6a3325]"
                        style={{ height: `${height}%` }}
                      />
                    </div>
                    <div className="text-base text-[#a89588] mt-4">{day.day}</div>
                    <div className="text-base text-[#7b3d2c] font-medium">{day.contracts}</div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card padding="lg">
        <CardContent className="p-10">
          <div className="flex items-center gap-5 mb-10">
            <AlertCircle className="w-7 h-7 text-[#e78745]" />
            <h2 className="text-2xl font-normal text-[#7b3d2c] font-serif">
              {t.dashboard?.systemInsights || '系统洞察'}
            </h2>
          </div>
          <div className="space-y-6">
            <div className="flex items-start gap-5 p-6 rounded-2xl bg-[rgba(231,135,69,0.06)] border border-[rgba(231,135,69,0.12)]">
              <TrendingUp className="w-6 h-6 text-[#e78745] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-lg text-[#7b3d2c] font-medium mb-3">周五晚上需求激增</div>
                <div className="text-lg text-[#a89588] leading-loose">
                  预计周五晚上7-9点需求将增长45%
                </div>
              </div>
            </div>
            <div className="flex items-start gap-5 p-6 rounded-2xl bg-[rgba(74,222,128,0.06)] border border-[rgba(74,222,128,0.12)]">
              <CheckCircle className="w-6 h-6 text-[#16a34a] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-lg text-[#7b3d2c] font-medium mb-3">匹配效率提升</div>
                <div className="text-lg text-[#a89588] leading-loose">本周平均匹配分数提升2.5%</div>
              </div>
            </div>
            <div className="flex items-start gap-5 p-6 rounded-2xl bg-[rgba(251,191,36,0.06)] border border-[rgba(251,191,36,0.12)]">
              <AlertCircle className="w-6 h-6 text-[#d97706] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-lg text-[#7b3d2c] font-medium mb-3">库存预警</div>
                <div className="text-lg text-[#a89588] leading-loose">3家商家库存水平较低</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
