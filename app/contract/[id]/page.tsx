import ContractContent from './ContractContent';

// 静态导出所需的参数生成
export function generateStaticParams() {
  return [{ id: 'demo' }];
}

export default function ContractPage({ params }: { params: { id: string } }) {
  return <ContractContent id={params.id} />;
}
