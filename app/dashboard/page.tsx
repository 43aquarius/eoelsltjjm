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
      <div className="mb-12">
        <h1 className="text-4xl font-normal text-[#7b3d2c] mb-4 font-serif">
          {t.dashboard?.title || '平台仪表板'}
        </h1>
        <p className="text-lg text-[#a89588] leading-loose">
          {t.dashboard?.subtitle || '查看平台关键指标和性能数据'}
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} padding="md">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-base text-[#a89588] truncate">{stat.label}</span>
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: `${stat.color}12` }}
                  >
                    <Icon className="w-6 h-6" style={{ color: stat.color }} />
                  </div>
                </div>
                <div className="text-3xl font-normal text-[#7b3d2c] mb-2 font-serif">{stat.value}</div>
                <div className="text-sm text-[#16a34a]">{stat.change}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        <Card padding="md">
          <CardContent className="p-6">
            <div className="flex items-center gap-4 mb-8">
              <BarChart3 className="w-6 h-6 text-[#e78745]" />
              <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
                {t.dashboard?.performanceByCategory || '性能指标'}
              </h2>
            </div>
            <div className="space-y-5">
              {performance.map((perf, index) => {
                const percentage = (perf.value / perf.target) * 100;
                return (
                  <div key={index}>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-base text-[#a89588]">{perf.label}</span>
                      <span className="text-base text-[#7b3d2c] font-medium">
                        {perf.value}% / {perf.target}%
                      </span>
                    </div>
                    <div className="h-2.5 bg-[rgba(123,61,44,0.06)] rounded-full overflow-hidden">
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

        <Card padding="md">
          <CardContent className="p-6">
            <div className="flex items-center gap-4 mb-8">
              <TrendingUp className="w-6 h-6 text-[#e78745]" />
              <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
                {t.dashboard?.dailyContracts || '每日合约趋势'}
              </h2>
            </div>
            <div className="flex items-end justify-between gap-3 h-44">
              {dailyTrend.map((day, index) => {
                const height = (day.contracts / maxContracts) * 100;
                return (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    <div className="w-full flex flex-col items-center justify-end h-36">
                      <div
                        className="w-full rounded-t-lg bg-[#7b3d2c] transition-all duration-300 hover:bg-[#6a3325]"
                        style={{ height: `${height}%` }}
                      />
                    </div>
                    <div className="text-sm text-[#a89588] mt-3">{day.day}</div>
                    <div className="text-sm text-[#7b3d2c] font-medium">{day.contracts}</div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card padding="lg">
        <CardContent className="p-8">
          <div className="flex items-center gap-4 mb-8">
            <AlertCircle className="w-6 h-6 text-[#e78745]" />
            <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
              {t.dashboard?.systemInsights || '系统洞察'}
            </h2>
          </div>
          <div className="space-y-5">
            <div className="flex items-start gap-4 p-5 rounded-xl bg-[rgba(231,135,69,0.06)] border border-[rgba(231,135,69,0.12)]">
              <TrendingUp className="w-5 h-5 text-[#e78745] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-base text-[#7b3d2c] font-medium mb-2">周五晚上需求激增</div>
                <div className="text-base text-[#a89588] leading-relaxed">
                  预计周五晚上7-9点需求将增长45%
                </div>
              </div>
            </div>
            <div className="flex items-start gap-4 p-5 rounded-xl bg-[rgba(74,222,128,0.06)] border border-[rgba(74,222,128,0.12)]">
              <CheckCircle className="w-5 h-5 text-[#16a34a] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-base text-[#7b3d2c] font-medium mb-2">匹配效率提升</div>
                <div className="text-base text-[#a89588] leading-relaxed">本周平均匹配分数提升2.5%</div>
              </div>
            </div>
            <div className="flex items-start gap-4 p-5 rounded-xl bg-[rgba(251,191,36,0.06)] border border-[rgba(251,191,36,0.12)]">
              <AlertCircle className="w-5 h-5 text-[#d97706] flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-base text-[#7b3d2c] font-medium mb-2">库存预警</div>
                <div className="text-base text-[#a89588] leading-relaxed">3家商家库存水平较低</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
