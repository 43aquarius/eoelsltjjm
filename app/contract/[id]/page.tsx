'use client';

import { useState } from 'react';
import { useParams } from 'next/navigation';
import {
  FileCheck,
  Clock,
  MapPin,
  Users,
  DollarSign,
  Shield,
  CheckCircle,
  AlertCircle,
  QrCode,
  ChevronRight,
} from 'lucide-react';
import { useTranslation } from '@/lib/LanguageContext';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

export default function ContractPage() {
  const params = useParams();
  const t = useTranslation();
  const [currentStatus, setCurrentStatus] = useState<'pending' | 'confirmed' | 'arrived' | 'completed'>('pending');

  const contractId = params.id || 'demo';

  const contract = {
    id: contractId,
    merchant: '樱花日料',
    cuisine: '日本料理',
    date: '2026-06-07',
    time: '19:00',
    partySize: 4,
    budget: 400,
    location: '北京 · 五道口',
    price: 380,
    promises: [
      t.contract?.compensationRules || '价格保证：锁定价格，无隐藏费用',
      '预约承诺：保证预留座位',
      '服务质量：如有问题自动赔偿',
      '违约赔偿：如商家违约，全额退款',
    ],
    compensation: [
      '迟到超过15分钟：补偿50元',
      '服务质量问题：补偿100元',
      '商家无法接待：全额退款+补偿200元',
    ],
  };

  const statusSteps = [
    { key: 'pending', label: t.contract?.draft || '待确认', icon: Clock },
    { key: 'confirmed', label: t.contract?.confirmed || '已确认', icon: CheckCircle },
    { key: 'arrived', label: t.contract?.arrived || '已到达', icon: MapPin },
    { key: 'completed', label: t.contract?.completed || '已完成', icon: CheckCircle },
  ];

  const currentStepIndex = statusSteps.findIndex(s => s.key === currentStatus);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-12">
        <div className="flex items-center gap-2 text-base text-[#a89588] mb-4">
          <span>{t.nav?.matchResults || '匹配结果'}</span>
          <ChevronRight className="w-5 h-5" />
          <span className="text-[#7b3d2c]">{t.contract?.title || '合约详情'}</span>
        </div>
        <h1 className="text-4xl font-normal text-[#7b3d2c] mb-3 font-serif">
          {t.contract?.title || '合约详情'} #{contractId}
        </h1>
        <p className="text-base text-[#a89588]">
          {contract.merchant} · {contract.cuisine}
        </p>
      </div>

      <Card padding="md" className="mb-8">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-8">
            <FileCheck className="w-6 h-6 text-[#e78745]" />
            <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
              {t.contract?.contractSummary || '合约状态'}
            </h2>
          </div>
          <div className="flex items-center justify-between mb-8">
            {statusSteps.map((step, index) => {
              const Icon = step.icon;
              const isActive = index <= currentStepIndex;
              const isCurrent = index === currentStepIndex;
              return (
                <div key={step.key} className="flex flex-col items-center flex-1">
                  <div
                    className={`w-11 h-11 rounded-full flex items-center justify-center mb-3 ${
                      isActive
                        ? 'bg-[#7b3d2c] text-white'
                        : 'bg-[rgba(123,61,44,0.06)] text-[#c4b5a9]'
                    } ${isCurrent ? 'ring-2 ring-[#7b3d2c] ring-offset-2 ring-offset-white' : ''}`}
                  >
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className={`text-sm ${isActive ? 'text-[#7b3d2c]' : 'text-[#c4b5a9]'}`}>
                    {step.label}
                  </span>
                </div>
              );
            })}
          </div>
          {currentStepIndex < statusSteps.length - 1 && (
            <div className="flex justify-center">
              <Button
                size="md"
                onClick={() => setCurrentStatus(statusSteps[currentStepIndex + 1].key as typeof currentStatus)}
              >
                {t.contract?.advanceTo?.replace('{status}', '') || '推进状态'}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-8 mb-8">
        <Card padding="md">
          <CardContent className="p-6">
            <h2 className="text-xl font-normal text-[#7b3d2c] mb-6 font-serif">
              {t.contract?.contractSummary || '合约摘要'}
            </h2>
            <div className="space-y-0">
              <div className="flex items-center justify-between py-4 border-b border-[rgba(123,61,44,0.08)]">
                <span className="text-base text-[#a89588] flex items-center gap-3">
                  <Users className="w-5 h-5" />
                  {t.contract?.partySize || '人数'}
                </span>
                <span className="text-base text-[#7b3d2c]">{contract.partySize} 人</span>
              </div>
              <div className="flex items-center justify-between py-4 border-b border-[rgba(123,61,44,0.08)]">
                <span className="text-base text-[#a89588] flex items-center gap-3">
                  <Clock className="w-5 h-5" />
                  {t.contract?.timeSlot || '时间'}
                </span>
                <span className="text-base text-[#7b3d2c]">{contract.time}</span>
              </div>
              <div className="flex items-center justify-between py-4 border-b border-[rgba(123,61,44,0.08)]">
                <span className="text-base text-[#a89588] flex items-center gap-3">
                  <MapPin className="w-5 h-5" />
                  {t.contract?.location || '地点'}
                </span>
                <span className="text-base text-[#7b3d2c]">{contract.location}</span>
              </div>
              <div className="flex items-center justify-between py-4">
                <span className="text-base text-[#a89588] flex items-center gap-3">
                  <DollarSign className="w-5 h-5" />
                  {t.contract?.totalPrice || '价格'}
                </span>
                <span className="text-base text-[#16a34a] font-medium">¥{contract.price}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card padding="md">
          <CardContent className="p-6">
            <h2 className="text-xl font-normal text-[#7b3d2c] mb-6 font-serif">
              {t.contract?.qrCode || '验证二维码'}
            </h2>
            <div className="flex flex-col items-center justify-center py-8">
              <div className="w-28 h-28 bg-[#7b3d2c] bg-opacity-10 rounded-xl flex items-center justify-center mb-4">
                <QrCode className="w-18 h-18 text-[#7b3d2c]" />
              </div>
              <p className="text-sm text-[#a89588] text-center">
                到店后出示此二维码
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card padding="md" className="mb-8">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-6">
            <Shield className="w-6 h-6 text-[#e78745]" />
            <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
              {t.contract?.servicePromises || '服务承诺'}
            </h2>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {contract.promises.map((promise, index) => (
              <div key={index} className="flex items-start gap-4 p-4 rounded-xl bg-[rgba(231,135,69,0.06)] border border-[rgba(231,135,69,0.12)]">
                <CheckCircle className="w-5 h-5 text-[#e78745] flex-shrink-0 mt-0.5" />
                <span className="text-base text-[#8b6e62] leading-relaxed">{promise}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card padding="md">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-6">
            <AlertCircle className="w-6 h-6 text-[#d97706]" />
            <h2 className="text-xl font-normal text-[#7b3d2c] font-serif">
              {t.contract?.compensationRules || '赔偿规则'}
            </h2>
          </div>
          <div className="space-y-4">
            {contract.compensation.map((rule, index) => (
              <div key={index} className="flex items-start gap-4 p-4 rounded-xl bg-[rgba(251,191,36,0.06)] border border-[rgba(251,191,36,0.12)]">
                <AlertCircle className="w-5 h-5 text-[#d97706] flex-shrink-0 mt-0.5" />
                <span className="text-base text-[#8b6e62] leading-relaxed">{rule}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
