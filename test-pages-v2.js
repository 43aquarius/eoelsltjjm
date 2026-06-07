const http = require('http');

function test(url, description, expectedContent) {
  return new Promise((resolve) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const status = res.statusCode;
        const hasContent = data.includes(expectedContent);
        const icon = status === 200 && hasContent ? '✓' : '❌';
        console.log(`${icon} ${description}: ${status} ${hasContent ? '(内容正确)' : '(缺少内容: ' + expectedContent + ')'}`);
        resolve({ status, hasContent });
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
    { url: 'http://127.0.0.1:3003/', name: '首页', expected: '从推荐' },
    { url: 'http://127.0.0.1:3003/create', name: '创建需求页', expected: '创建需求' },
    { url: 'http://127.0.0.1:3003/matches', name: '匹配结果页', expected: '匹配结果' },
    { url: 'http://127.0.0.1:3003/merchant', name: '商家仪表板', expected: '商家仪表板' },
    { url: 'http://127.0.0.1:3003/dashboard', name: '平台仪表板', expected: '平台仪表板' },
  ];

  const results = [];
  for (const t of tests) {
    const result = await test(t.url, t.name, t.expected);
    results.push({ name: t.name, ...result });
  }

  console.log('\n=== 检查结果汇总 ===');
  const passed = results.filter(r => r.status === 200 && r.hasContent).length;
  const failed = results.length - passed;
  console.log(`通过: ${passed}/${results.length}`);
  console.log(`失败: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log('\n=== 失败的页面 ===');
    results.filter(r => r.status !== 200 || !r.hasContent).forEach(r => {
      console.log(`  - ${r.name}: ${r.status || 'NETWORK ERROR'}`);
    });
  } else {
    console.log('\n🎉 所有页面都正常！');
  }
}

runTests();
