export type Language = 'zh' | 'en';

export interface TranslationKeys {
  common: {
    loading: string;
    error: string;
    retry: string;
    confirm: string;
    cancel: string;
    save: string;
    delete: string;
    edit: string;
    back: string;
    next: string;
    previous: string;
    close: string;
    search: string;
    filter: string;
    sort: string;
    all: string;
    none: string;
  };
  nav: {
    home: string;
    createDemand: string;
    matchResults: string;
    contract: string;
    merchant: string;
    dashboard: string;
  };
  home: {
    badge: string;
    title1: string;
    title2: string;
    subtitle: string;
    cta1: string;
    cta2: string;
    guaranteeTitle: string;
    guaranteeDesc: string;
    matchingTitle: string;
    matchingDesc: string;
    realtimeTitle: string;
    realtimeDesc: string;
    howItWorks: string;
    step1Title: string;
    step1Desc: string;
    step2Title: string;
    step2Desc: string;
    step3Title: string;
    step3Desc: string;
    step4Title: string;
    step4Desc: string;
    traditionalTitle: string;
    contractTitle: string;
    traditional1: string;
    traditional2: string;
    traditional3: string;
    traditional4: string;
    traditional5: string;
    contract1: string;
    contract2: string;
    contract3: string;
    contract4: string;
    contract5: string;
    activeContracts: string;
    fulfillmentRate: string;
    merchantPartner: string;
    avgMatchScore: string;
    readyTitle: string;
    readyDesc: string;
    getStarted: string;
    footer: string;
    footerDesc: string;
  };
  create: {
    title: string;
    subtitle: string;
    naturalInput: string;
    placeholder: string;
    tryNatural: string;
    generateMatches: string;
    analyzing: string;
    quickExamples: string;
    structuredForm: string;
    partySize: string;
    timeSlot: string;
    budget: string;
    total: string;
    perPerson: string;
    location: string;
    preferences: string;
    constraints: string;
    quiet: string;
    romantic: string;
    privateRoom: string;
    outdoor: string;
    wifi: string;
    parking: string;
    noSpice: string;
    vegetarian: string;
    vegan: string;
    noPork: string;
    halal: string;
    livePreview: string;
    enterDemand: string;
    matchingAlgorithm: string;
    demandFit: string;
    fulfillmentRate: string;
    supplyIdleScore: string;
    priceScore: string;
    distanceScore: string;
    howItWorksTitle: string;
    howItWorks1: string;
    howItWorks2: string;
    howItWorks3: string;
    howItWorks4: string;
  };
  matches: {
    title: string;
    subtitle: string;
    analyzingMatches: string;
    sortBest: string;
    sortCheapest: string;
    sortStable: string;
    sortEnvironment: string;
    recommended: string;
    matchScore: string;
    seats: string;
    kitchen: string;
    queue: string;
    fulfill: string;
    whyMatch: string;
    scoreBreakdown: string;
    offer: string;
    validFor: string;
    total: string;
    perPerson: string;
    bindingPromises: string;
    binding: string;
    acceptOffer: string;
    riskAssessment: string;
    lowRisk: string;
    mediumRisk: string;
    highRisk: string;
    waitTime: string;
  };
  contract: {
    title: string;
    subtitle: string;
    loading: string;
    draft: string;
    confirmed: string;
    accepted: string;
    arrived: string;
    inProgress: string;
    completed: string;
    breached: string;
    compensated: string;
    cancelled: string;
    contractSummary: string;
    partySize: string;
    timeSlot: string;
    location: string;
    totalPrice: string;
    servicePromises: string;
    compensationRules: string;
    specialRequests: string;
    qrCode: string;
    demoControls: string;
    demoDesc: string;
    simulateBreach: string;
    advanceTo: string;
    processCompensation: string;
    contractCompleted: string;
    allPromises: string;
  };
  merchant: {
    title: string;
    subtitle: string;
    selectMerchant: string;
    businessStatus: string;
    performanceMetrics: string;
    contractOpportunities: string;
    seatAvailability: string;
    kitchenLoad: string;
    currentQueue: string;
    inventoryLevel: string;
    fulfillmentRate: string;
    customerSatisfaction: string;
    breachRate: string;
    totalContracts: string;
    avgServiceTime: string;
    demand: string;
    suggestedOffer: string;
    potentialRevenue: string;
    accept: string;
    reject: string;
    contractAccepted: string;
    noOpportunities: string;
    opportunitiesDesc: string;
  };
  dashboard: {
    title: string;
    subtitle: string;
    revenueGenerated: string;
    compensationPaid: string;
    dailyContracts: string;
    performanceByCategory: string;
    systemInsights: string;
    peakHours: string;
    peakHoursDesc: string;
    topCategory: string;
    topCategoryDesc: string;
    capacityOptimization: string;
    capacityOptimizationDesc: string;
    attentionRequired: string;
    attentionRequiredDesc: string;
    valueCreation: string;
    activeMerchants: string;
    thisMonth: string;
    activeContracts: string;
    thisWeek: string;
    industryLeading: string;
    footerStats: string;
    avgResponseTime: string;
    userSatisfaction: string;
    merchantRetention: string;
  };
  commandMenu: {
    placeholder: string;
    search: string;
    navigation: string;
    demoActions: string;
    runDemo: string;
    runDemoDesc: string;
    quickMatch: string;
    quickMatchDesc: string;
    noResults: string;
    noResultsDesc: string;
    result: string;
    results: string;
  };
  footer: {
    demoMode: string;
    location: string;
    searchPlaceholder: string;
  };
}

export const translations: Record<Language, TranslationKeys> = {
  zh: {
    common: {
      loading: '加载中...',
      error: '出错了',
      retry: '重试',
      confirm: '确认',
      cancel: '取消',
      save: '保存',
      delete: '删除',
      edit: '编辑',
      back: '返回',
      next: '下一步',
      previous: '上一步',
      close: '关闭',
      search: '搜索',
      filter: '筛选',
      sort: '排序',
      all: '全部',
      none: '无',
    },
    nav: {
      home: '首页',
      createDemand: '创建需求',
      matchResults: '匹配结果',
      contract: '合约详情',
      merchant: '商家中心',
      dashboard: '平台仪表板',
    },
    home: {
      badge: '黑客松 Demo 2026',
      title1: '从推荐',
      title2: '到承诺',
      subtitle: 'LocalContract 将传统的推荐式本地服务转变为承诺式合约，提供保价、预约和违约赔偿。',
      cta1: '创建需求',
      cta2: '查看演示',
      guaranteeTitle: '有保障的承诺',
      guaranteeDesc: '每份合约都包含有约束力的承诺：价格锁定、预约保证、服务质量承诺，以及违约赔偿。',
      matchingTitle: '智能匹配',
      matchingDesc: '我们的算法综合考虑需求匹配度、商家可靠性、产能利用率和价格，为您找到最佳匹配。',
      realtimeTitle: '实时可用',
      realtimeDesc: '实时追踪座位可用性、厨房负载和排队时间，确保精准匹配，减少等待时间。',
      howItWorks: '工作流程',
      step1Title: '创建需求',
      step1Desc: '用自然语言告诉我们您需要什么',
      step2Title: '智能解析',
      step2Desc: 'AI 将您的需求解析为结构化数据',
      step3Title: '匹配评分',
      step3Desc: '算法找到并排名最佳商家匹配',
      step4Title: '获取合约',
      step4Desc: '接收有保障的绑定要约',
      traditionalTitle: '传统模式',
      contractTitle: 'LocalContract 模式',
      traditional1: '推荐无保障',
      traditional2: '无价格保护',
      traditional3: '可用性不确定',
      traditional4: '无违约赔偿',
      traditional5: '闲置产能浪费',
      contract1: '绑定服务合约',
      contract2: '价格有保障',
      contract3: '座位已预约',
      contract4: '自动赔偿',
      contract5: '优化产能利用',
      activeContracts: '活跃合约',
      fulfillmentRate: '履约率',
      merchantPartner: '合作商家',
      avgMatchScore: '平均匹配分数',
      readyTitle: '准备好体验未来了吗？',
      readyDesc: '创建您的第一个需求，查看匹配算法的实际效果。',
      getStarted: '开始使用',
      footer: 'LocalContract · 为黑客松 2026 构建',
      footerDesc: '将本地服务从推荐转变为承诺',
    },
    create: {
      title: '创建需求',
      subtitle: '告诉我们您需要什么，我们会找到完美的商家匹配并提供有保障的承诺。',
      naturalInput: '自然语言输入',
      placeholder: '示例：今晚7点，4人，预算400，安静，不要辣，要包间',
      tryNatural: '尝试自然语言 - AI 将解析您的需求',
      generateMatches: '生成匹配',
      analyzing: '分析中...',
      quickExamples: '快速示例',
      structuredForm: '或使用结构化表单',
      partySize: '人数',
      timeSlot: '时间段',
      budget: '预算',
      total: '总计',
      perPerson: '人均',
      location: '位置',
      preferences: '偏好',
      constraints: '限制',
      quiet: '安静',
      romantic: '浪漫',
      privateRoom: '包间',
      outdoor: '户外',
      wifi: 'WiFi',
      parking: '停车',
      noSpice: '不要辣',
      vegetarian: '素食',
      vegan: '纯素',
      noPork: '不要猪肉',
      halal: '清真',
      livePreview: '实时解析预览',
      enterDemand: '输入需求以查看实时解析',
      matchingAlgorithm: '匹配算法',
      demandFit: '需求匹配度',
      fulfillmentRate: '履约率',
      supplyIdleScore: '闲置产能',
      priceScore: '价格评分',
      distanceScore: '距离评分',
      howItWorksTitle: '工作原理',
      howItWorks1: '您的输入被解析为结构化需求数据',
      howItWorks2: '算法在多个维度上为所有商家评分',
      howItWorks3: '最佳匹配会排名并附带详细解释',
      howItWorks4: '生成有保障的绑定要约',
    },
    matches: {
      title: '匹配结果',
      subtitle: '找到 {count} 个匹配的商家',
      analyzingMatches: '正在分析匹配...',
      sortBest: '最佳匹配',
      sortCheapest: '最便宜',
      sortStable: '最可靠',
      sortEnvironment: '最佳环境',
      recommended: '推荐',
      matchScore: '匹配分数',
      seats: '座位',
      kitchen: '厨房',
      queue: '排队',
      fulfill: '履约',
      whyMatch: '为什么选择这家？',
      scoreBreakdown: '分数明细',
      offer: '生成的要约',
      validFor: '30分钟内有效',
      total: '总计',
      perPerson: '人均',
      bindingPromises: '绑定承诺',
      binding: '绑定',
      acceptOffer: '接受此要约',
      riskAssessment: '风险评估',
      lowRisk: '低风险',
      mediumRisk: '中风险',
      highRisk: '高风险',
      waitTime: '预计等待时间',
    },
    contract: {
      title: '合约详情',
      subtitle: '合约ID：{id}',
      loading: '正在加载合约...',
      draft: '草稿',
      confirmed: '已确认',
      accepted: '已接受',
      arrived: '已到达',
      inProgress: '进行中',
      completed: '已完成',
      breached: '已违约',
      compensated: '已赔偿',
      cancelled: '已取消',
      contractSummary: '合约摘要',
      partySize: '人数',
      timeSlot: '时间段',
      location: '地点',
      totalPrice: '总价',
      servicePromises: '服务承诺',
      compensationRules: '违约赔偿规则',
      specialRequests: '特殊要求',
      qrCode: '验证二维码',
      demoControls: '演示控制',
      demoDesc: '模拟合约生命周期以进行演示',
      simulateBreach: '模拟违约',
      advanceTo: '推进到 {status}',
      processCompensation: '处理赔偿',
      contractCompleted: '合约已完成！',
      allPromises: '所有承诺已成功履行',
    },
    merchant: {
      title: '商家仪表板',
      subtitle: '实时业务状态和合约机会',
      selectMerchant: '选择商家',
      businessStatus: '业务状态',
      performanceMetrics: '绩效指标',
      contractOpportunities: '合约机会',
      seatAvailability: '座位可用性',
      kitchenLoad: '厨房负载',
      currentQueue: '当前排队',
      inventoryLevel: '库存水平',
      fulfillmentRate: '履约率',
      customerSatisfaction: '顾客满意度',
      breachRate: '违约率',
      totalContracts: '总合约数',
      avgServiceTime: '平均服务时间',
      demand: '需求',
      suggestedOffer: '建议报价',
      potentialRevenue: '潜在收入',
      accept: '接受',
      reject: '拒绝',
      contractAccepted: '合约已接受并确认',
      noOpportunities: '暂无待处理机会',
      opportunitiesDesc: '新的合约机会将显示在这里',
    },
    dashboard: {
      title: '平台仪表板',
      subtitle: 'LocalContract 平台的实时数据和洞察',
      revenueGenerated: '已产生收入',
      compensationPaid: '已支付赔偿',
      dailyContracts: '每日合约（最近7天）',
      performanceByCategory: '按类别的性能',
      systemInsights: '系统洞察',
      peakHours: '高峰时段已识别',
      peakHoursDesc: '晚上7-9点需求高45%，建议商家调整产能。',
      topCategory: '最佳表现类别',
      topCategoryDesc: '法餐满意度最高（4.7★），履约率97%。',
      capacityOptimization: '产能优化',
      capacityOptimizationDesc: '23家餐厅午餐时段闲置产能>70%，可增加156份合约。',
      attentionRequired: '需要关注',
      attentionRequiredDesc: '5家商家违约率>5%，建议审查合约要求。',
      valueCreation: '价值创造总结',
      activeMerchants: '活跃商家',
      thisMonth: '本月 +8%',
      activeContracts: '活跃合约',
      thisWeek: '本周 +23%',
      industryLeading: '行业领先',
      footerStats: 'LocalContract 已促成 {contracts} 份合约，履约率 {rate}%，为商家创造 {revenue} 价值，为用户提供 {experiences} 次有保障的服务体验。',
      avgResponseTime: '平均响应时间',
      userSatisfaction: '用户满意度',
      merchantRetention: '商家留存率',
    },
    commandMenu: {
      placeholder: '输入命令或搜索...',
      search: '搜索...',
      navigation: '导航',
      demoActions: '演示操作',
      runDemo: '运行演示流程',
      runDemoDesc: '执行完整的演示场景',
      quickMatch: '快速匹配',
      quickMatchDesc: '使用示例数据运行匹配算法',
      noResults: '未找到结果',
      noResultsDesc: '尝试不同的搜索词',
      result: '个结果',
      results: '个结果',
    },
    footer: {
      demoMode: '演示模式',
      location: '北京 · 五道口',
      searchPlaceholder: '搜索...',
    },
  },
  en: {
    common: {
      loading: 'Loading...',
      error: 'Error',
      retry: 'Retry',
      confirm: 'Confirm',
      cancel: 'Cancel',
      save: 'Save',
      delete: 'Delete',
      edit: 'Edit',
      back: 'Back',
      next: 'Next',
      previous: 'Previous',
      close: 'Close',
      search: 'Search',
      filter: 'Filter',
      sort: 'Sort',
      all: 'All',
      none: 'None',
    },
    nav: {
      home: 'Home',
      createDemand: 'Create Demand',
      matchResults: 'Match Results',
      contract: 'Contract',
      merchant: 'Merchant',
      dashboard: 'Dashboard',
    },
    home: {
      badge: 'Hackathon Demo 2026',
      title1: 'From Recommendation',
      title2: 'to Commitment',
      subtitle: 'LocalContract transforms traditional recommendation-based local services into commitment-based contracts with guaranteed prices, reservations, and breach compensation.',
      cta1: 'Create a Demand',
      cta2: 'View Demo Matches',
      guaranteeTitle: 'Guaranteed Promises',
      guaranteeDesc: 'Every contract comes with binding promises: price locks, reservation guarantees, and service quality commitments backed by compensation.',
      matchingTitle: 'Smart Matching',
      matchingDesc: 'Our algorithm considers demand fit, merchant reliability, capacity utilization, and price to find the perfect match for your needs.',
      realtimeTitle: 'Real-time Availability',
      realtimeDesc: 'Live tracking of seat availability, kitchen load, and queue times ensures accurate matching with minimal wait times.',
      howItWorks: 'How It Works',
      step1Title: 'Create Demand',
      step1Desc: 'Tell us what you need in natural language',
      step2Title: 'Smart Parse',
      step2Desc: 'AI parses your requirements into structured data',
      step3Title: 'Match & Score',
      step3Desc: 'Algorithm finds and ranks best merchant matches',
      step4Title: 'Get Contract',
      step4Desc: 'Receive binding offers with guarantees',
      traditionalTitle: 'Traditional Model',
      contractTitle: 'LocalContract Model',
      traditional1: 'Recommendations without guarantees',
      traditional2: 'No price protection',
      traditional3: 'Uncertain availability',
      traditional4: 'No breach compensation',
      traditional5: 'Wasted idle capacity',
      contract1: 'Binding service contracts',
      contract2: 'Guaranteed prices',
      contract3: 'Reserved seating',
      contract4: 'Automatic compensation',
      contract5: 'Optimized capacity utilization',
      activeContracts: 'Active Contracts',
      fulfillmentRate: 'Fulfillment Rate',
      merchantPartner: 'Merchant Partners',
      avgMatchScore: 'Avg Match Score',
      readyTitle: 'Ready to experience the future?',
      readyDesc: 'Create your first demand and see the matching algorithm in action.',
      getStarted: 'Get Started',
      footer: 'LocalContract · Built for Hackathon 2026',
      footerDesc: 'Transforming local services from recommendation to commitment',
    },
    create: {
      title: 'Create Demand',
      subtitle: 'Tell us what you need, and we\'ll find the perfect merchant match with guaranteed promises.',
      naturalInput: 'Natural Language Input',
      placeholder: 'Example: Tonight 7pm, 4 people, budget 400, quiet, no spice, private room',
      tryNatural: 'Try natural language - our AI will parse your requirements',
      generateMatches: 'Generate Matches',
      analyzing: 'Analyzing...',
      quickExamples: 'Quick Examples',
      structuredForm: 'Or Use Structured Form',
      partySize: 'Party Size',
      timeSlot: 'Time Slot',
      budget: 'Budget',
      total: 'Total',
      perPerson: 'Per Person',
      location: 'Location',
      preferences: 'Preferences',
      constraints: 'Constraints',
      quiet: 'quiet',
      romantic: 'romantic',
      privateRoom: 'private room',
      outdoor: 'outdoor',
      wifi: 'wifi',
      parking: 'parking',
      noSpice: 'no spice',
      vegetarian: 'vegetarian',
      vegan: 'vegan',
      noPork: 'no pork',
      halal: 'halal',
      livePreview: 'Live Parsing Preview',
      enterDemand: 'Enter your demand to see real-time parsing',
      matchingAlgorithm: 'Matching Algorithm',
      demandFit: 'Demand Fit',
      fulfillmentRate: 'Fulfillment Rate',
      supplyIdleScore: 'Supply-Idle Score',
      priceScore: 'Price Score',
      distanceScore: 'Distance Score',
      howItWorksTitle: 'How It Works',
      howItWorks1: 'Your input is parsed into structured demand data',
      howItWorks2: 'Algorithm scores all merchants on multiple dimensions',
      howItWorks3: 'Best matches are ranked with detailed explanations',
      howItWorks4: 'Generate binding offers with guaranteed promises',
    },
    matches: {
      title: 'Match Results',
      subtitle: 'Found {count} matching merchants',
      analyzingMatches: 'Analyzing Matches...',
      sortBest: 'Best Match',
      sortCheapest: 'Cheapest',
      sortStable: 'Most Reliable',
      sortEnvironment: 'Best Environment',
      recommended: 'Recommended',
      matchScore: 'Match Score',
      seats: 'Seats',
      kitchen: 'Kitchen',
      queue: 'Queue',
      fulfill: 'Fulfill',
      whyMatch: 'Why this match?',
      scoreBreakdown: 'Score Breakdown',
      offer: 'Generated Offer',
      validFor: 'Valid for 30 minutes',
      total: 'total',
      perPerson: 'per person',
      bindingPromises: 'Binding Promises',
      binding: 'Binding',
      acceptOffer: 'Accept This Offer',
      riskAssessment: 'Risk Assessment',
      lowRisk: 'Low Risk',
      mediumRisk: 'Medium Risk',
      highRisk: 'High Risk',
      waitTime: 'Est. Wait Time',
    },
    contract: {
      title: 'Contract Details',
      subtitle: 'Contract ID: {id}',
      loading: 'Loading Contract...',
      draft: 'Draft',
      confirmed: 'Confirmed',
      accepted: 'Accepted',
      arrived: 'Arrived',
      inProgress: 'In Progress',
      completed: 'Completed',
      breached: 'Breached',
      compensated: 'Compensated',
      cancelled: 'Cancelled',
      contractSummary: 'Contract Summary',
      partySize: 'Party Size',
      timeSlot: 'Time Slot',
      location: 'Location',
      totalPrice: 'Total Price',
      servicePromises: 'Service Promises',
      compensationRules: 'Compensation Rules',
      specialRequests: 'Special Requests',
      qrCode: 'Verification QR Code',
      demoControls: 'Demo Controls',
      demoDesc: 'Simulate contract lifecycle for demo purposes',
      simulateBreach: 'Simulate Breach',
      advanceTo: 'Advance to {status}',
      processCompensation: 'Process Compensation',
      contractCompleted: 'Contract Completed!',
      allPromises: 'All promises fulfilled successfully',
    },
    merchant: {
      title: 'Merchant Dashboard',
      subtitle: 'Real-time business status and contract opportunities',
      selectMerchant: 'Select Merchant',
      businessStatus: 'Business Status',
      performanceMetrics: 'Performance Metrics',
      contractOpportunities: 'Contract Opportunities',
      seatAvailability: 'Seat Availability',
      kitchenLoad: 'Kitchen Load',
      currentQueue: 'Current Queue',
      inventoryLevel: 'Inventory Level',
      fulfillmentRate: 'Fulfillment Rate',
      customerSatisfaction: 'Customer Satisfaction',
      breachRate: 'Breach Rate',
      totalContracts: 'Total Contracts',
      avgServiceTime: 'Avg Service Time',
      demand: 'Demand',
      suggestedOffer: 'Suggested Offer',
      potentialRevenue: 'Potential Revenue',
      accept: 'Accept',
      reject: 'Reject',
      contractAccepted: 'Contract accepted and confirmed',
      noOpportunities: 'No pending opportunities',
      opportunitiesDesc: 'New contract opportunities will appear here',
    },
    dashboard: {
      title: 'Platform Dashboard',
      subtitle: 'Real-time metrics and insights for the LocalContract platform',
      revenueGenerated: 'Revenue Generated',
      compensationPaid: 'Compensation Paid',
      dailyContracts: 'Daily Contracts (Last 7 Days)',
      performanceByCategory: 'Performance by Category',
      systemInsights: 'System Insights',
      peakHours: 'Peak Hours Identified',
      peakHoursDesc: '7PM-9PM shows 45% higher demand. Recommend merchant capacity adjustments.',
      topCategory: 'Top Performing Category',
      topCategoryDesc: 'French cuisine has highest satisfaction (4.7★) and fulfillment rate (97%).',
      capacityOptimization: 'Capacity Optimization',
      capacityOptimizationDesc: '23 restaurants have >70% idle capacity during lunch. Potential for 156 additional contracts.',
      attentionRequired: 'Attention Required',
      attentionRequiredDesc: '5 merchants have breach rate >5%. Recommend contract requirement review.',
      valueCreation: 'Value Creation Summary',
      activeMerchants: 'Active Merchants',
      thisMonth: '+8% this month',
      activeContracts: 'Active Contracts',
      thisWeek: '+23% this week',
      industryLeading: 'Industry leading',
      footerStats: 'LocalContract has facilitated {contracts} contracts with a {rate}% fulfillment rate, creating {revenue} in value for merchants and {experiences} guaranteed service experiences for users.',
      avgResponseTime: 'Average Response Time',
      userSatisfaction: 'User Satisfaction',
      merchantRetention: 'Merchant Retention',
    },
    commandMenu: {
      placeholder: 'Type a command or search...',
      search: 'Search...',
      navigation: 'Navigation',
      demoActions: 'Demo Actions',
      runDemo: 'Run Demo Flow',
      runDemoDesc: 'Execute complete demo scenario',
      quickMatch: 'Quick Match',
      quickMatchDesc: 'Run matching algorithm with sample data',
      noResults: 'No results found',
      noResultsDesc: 'Try a different search term',
      result: 'result',
      results: 'results',
    },
    footer: {
      demoMode: 'Demo Mode',
      location: 'Beijing · Wudaokou',
      searchPlaceholder: 'Search...',
    },
  },
};
