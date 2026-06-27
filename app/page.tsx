'use client';

import { useRouter } from 'next/navigation';
import {
  ArrowRight,
  Zap,
  Shield,
  TrendingUp,
  Clock,
  CheckCircle,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { useTranslation } from '@/lib/LanguageContext';

export default function HomePage() {
  const router = useRouter();
  const t = useTranslation();

  const features = [
    { icon: Shield, title: t.home.guaranteeTitle, desc: t.home.guaranteeDesc, color: '#7b3d2c' },
    { icon: TrendingUp, title: t.home.matchingTitle, desc: t.home.matchingDesc, color: '#e78745' },
    { icon: Clock, title: t.home.realtimeTitle, desc: t.home.realtimeDesc, color: '#22c55e' },
  ];

  const steps = [
    { step: '01', title: t.home.step1Title, desc: t.home.step1Desc },
    { step: '02', title: t.home.step2Title, desc: t.home.step2Desc },
    { step: '03', title: t.home.step3Title, desc: t.home.step3Desc },
    { step: '04', title: t.home.step4Title, desc: t.home.step4Desc },
  ];

  const stats = [
    { label: t.home.activeContracts, value: '4,023', change: '+12%' },
    { label: t.home.fulfillmentRate, value: '92.8%', change: '+3.2%' },
    { label: t.home.merchantPartner, value: '156', change: '+8%' },
    { label: t.home.avgMatchScore, value: '84.2', change: '+5.1%' },
  ];

  return (
    <div className="max-w-5xl mx-auto">
      {/* Hero */}
      <section className="text-center py-24 mb-20">
        <Badge variant="primary" className="mb-8">
          <Zap className="w-4 h-4" />
          {t.home.badge}
        </Badge>

        <h1 className="text-5xl sm:text-7xl font-normal text-[#7b3d2c] mb-8 leading-snug tracking-tight font-serif">
          {t.home.title1}
          <br />
          <span className="text-gradient">{t.home.title2}</span>
        </h1>

        <p className="text-xl text-[#9a7e72] max-w-2xl mx-auto mb-14 leading-loose">
          {t.home.subtitle}
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-5">
          <Button size="lg" onClick={() => router.push('/create')} rightIcon={<ArrowRight className="w-5 h-5" />}>
            {t.home.cta1}
          </Button>
          <Button variant="secondary" size="lg" onClick={() => router.push('/matches')}>
            {t.home.cta2}
          </Button>
        </div>
      </section>

      {/* Features */}
      <section className="grid sm:grid-cols-3 gap-10 mb-28">
        {features.map((f) => (
          <Card key={f.title} padding="lg">
            <CardContent className="p-10">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-8" style={{ backgroundColor: `${f.color}12` }}>
                <f.icon className="w-8 h-8" style={{ color: f.color }} />
              </div>
              <h3 className="text-2xl font-normal text-[#7b3d2c] mb-5 font-serif">{f.title}</h3>
              <p className="text-lg text-[#9a7e72] leading-loose">{f.desc}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* How it works */}
      <section className="mb-28">
        <h2 className="text-4xl font-normal text-[#7b3d2c] text-center mb-16 font-serif">
          {t.home.howItWorks}
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((s) => (
            <Card key={s.step} padding="md">
              <CardContent className="p-8">
                <div className="text-base font-medium text-[#e78745] mb-5">STEP {s.step}</div>
                <h3 className="text-xl font-normal text-[#7b3d2c] mb-4 font-serif">{s.title}</h3>
                <p className="text-lg text-[#9a7e72] leading-loose">{s.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Comparison */}
      <section className="grid md:grid-cols-2 gap-10 mb-28">
        <Card className="border-[rgba(239,68,68,0.15)]" padding="lg">
          <CardContent className="p-10">
            <div className="flex items-center gap-5 mb-8">
              <div className="w-12 h-12 rounded-xl bg-[rgba(239,68,68,0.08)] flex items-center justify-center">
                <X className="w-6 h-6 text-[#dc2626]" />
              </div>
              <h3 className="text-2xl font-normal text-[#7b3d2c] font-serif">{t.home.traditionalTitle}</h3>
            </div>
            <ul className="space-y-5">
              {[t.home.traditional1, t.home.traditional2, t.home.traditional3, t.home.traditional4, t.home.traditional5].map((item, i) => (
                <li key={i} className="flex items-start gap-4 text-lg text-[#9a7e72] leading-loose">
                  <span className="text-[#dc2626] mt-1">×</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card className="border-[rgba(34,197,94,0.15)]" padding="lg">
          <CardContent className="p-10">
            <div className="flex items-center gap-5 mb-8">
              <div className="w-12 h-12 rounded-xl bg-[rgba(34,197,94,0.08)] flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-[#22c55e]" />
              </div>
              <h3 className="text-2xl font-normal text-[#7b3d2c] font-serif">{t.home.contractTitle}</h3>
            </div>
            <ul className="space-y-5">
              {[t.home.contract1, t.home.contract2, t.home.contract3, t.home.contract4, t.home.contract5].map((item, i) => (
                <li key={i} className="flex items-start gap-4 text-lg text-[#9a7e72] leading-loose">
                  <span className="text-[#22c55e] mt-1">✓</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-28">
        {stats.map((s) => (
          <Card key={s.label} padding="md">
            <CardContent className="p-8 text-center">
              <div className="text-4xl font-normal text-[#7b3d2c] mb-3 font-serif">{s.value}</div>
              <div className="text-lg text-[#9a7e72] mb-4">{s.label}</div>
              <div className="text-base text-[#22c55e] font-medium">{s.change}</div>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* CTA */}
      <section className="text-center mb-28">
        <Card className="inline-block bg-gradient-to-br from-[rgba(231,135,69,0.06)] to-transparent border-[rgba(231,135,69,0.12)]" padding="lg">
          <CardContent className="p-16">
            <h2 className="text-4xl font-normal text-[#7b3d2c] mb-6 font-serif">{t.home.readyTitle}</h2>
            <p className="text-lg text-[#9a7e72] mb-10 max-w-md mx-auto leading-loose">{t.home.readyDesc}</p>
            <Button size="lg" onClick={() => router.push('/create')} rightIcon={<ArrowRight className="w-5 h-5" />}>
              {t.home.getStarted}
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* Footer */}
      <footer className="text-center py-12 border-t border-[rgba(123,61,44,0.08)]">
        <p className="text-base text-[#b8a89c]">{t.home.footer}</p>
      </footer>
    </div>
  );
}
