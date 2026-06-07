const http = require('http');

function test(url, description) {
  return new Promise((resolve) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const hasError = data.includes('Error') && !data.includes('lucide');
        const status = res.statusCode;
        const icon = status === 200 && !hasError ? '✓' : '❌';
        console.log(`${icon} ${description}: ${status}`);
        resolve({ status, hasError });
      });
    }).on('error', (e) => {
      console.log(`❌ ${description}: NETWORK ERROR - ${e.message}`);
      resolve({ status: 0, error: true });
    });
  });
}

async function runTests() {
  console.log('=== 页面可访问性检查 ===\n');

  const tests = [
    { url: 'http://127.0.0.1:3003/', name: '首页' },
    { url: 'http://127.0.0.1:3003/create', name: '创建需求页' },
    { url: 'http://127.0.0.1:3003/matches', name: '匹配结果页' },
    { url: 'http://127.0.0.1:3003/merchant', name: '商家仪表板' },
    { url: 'http://127.0.0.1:3003/dashboard', name: '平台仪表板' },
  ];

  const results = [];
  for (const t of tests) {
    const result = await test(t.url, t.name);
    results.push({ name: t.name, ...result });
  }

  console.log('\n=== 检查结果汇总 ===');
  const passed = results.filter(r => r.status === 200 && !r.hasError).length;
  const failed = results.length - passed;
  console.log(`通过: ${passed}/${results.length}`);
  console.log(`失败: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log('\n=== 失败的页面 ===');
    results.filter(r => r.status !== 200 || r.hasError).forEach(r => {
      console.log(`  - ${r.name}: ${r.status || 'NETWORK ERROR'}`);
    });
  }
}

runTests();
