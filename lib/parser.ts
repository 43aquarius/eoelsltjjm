import type { ParsedDemand } from '@/types';

export function parseDemand(input: string): ParsedDemand {
  const normalizedInput = input.toLowerCase().trim();

  // Extract party size (支持中英文) - 必须在时间之前提取
  const partySizePatterns = [
    /(\d+)\s*(人|个人|位)/,
    /(\d+)\s*(people|person|guests|guest|ppl|pax)/,
  ];
  let partySize = 2;
  for (const pattern of partySizePatterns) {
    const match = normalizedInput.match(pattern);
    if (match) {
      partySize = parseInt(match[1]);
      break;
    }
  }

  // Extract time (支持中英文)
  let timeSlot = '19:00'; // 默认晚餐时间

  // 中文时间格式：19点、19:30、19点30分（排除人数后的匹配）
  const chineseTimeMatch = normalizedInput.match(/(\d{1,2})[点时:：](\d{1,2})?分?/);
  if (chineseTimeMatch) {
    const hour = parseInt(chineseTimeMatch[1]);
    // 只有合理的小时才作为时间（排除人数）
    if (hour >= 0 && hour <= 23) {
      const minute = chineseTimeMatch[2] || '00';
      const adjustedHour = hour < 12 && hour < 7 ? hour + 12 : hour;
      timeSlot = `${adjustedHour.toString().padStart(2, '0')}:${minute}`;
    }
  }

  // 英文时间格式
  const timeMatch = normalizedInput.match(/(\d{1,2}):(\d{2})\s*(pm|am)?/i);
  const simpleTimeMatch = normalizedInput.match(/(\d{1,2})\s*(pm|am)/i);
  const relativeTimeMatch = normalizedInput.match(
    /(tonight|today|tomorrow|dinner|lunch|breakfast|brunch|今晚|今天|明天|晚餐|午餐|早餐)/i
  );

  if (timeMatch) {
    const hour = parseInt(timeMatch[1]);
    const minute = timeMatch[2] || '00';
    const period = timeMatch[3]?.toLowerCase();
    let adjustedHour = hour;

    if (period === 'pm' && hour !== 12) adjustedHour += 12;
    if (period === 'am' && hour === 12) adjustedHour = 0;
    if (!period && hour < 12 && hour < 7) adjustedHour += 12; // Assume evening for small numbers

    timeSlot = `${adjustedHour.toString().padStart(2, '0')}:${minute}`;
  } else if (simpleTimeMatch) {
    let hour = parseInt(simpleTimeMatch[1]);
    const period = simpleTimeMatch[2]?.toLowerCase();
    if (period === 'pm' && hour !== 12) hour += 12;
    if (period === 'am' && hour === 12) hour = 0;
    timeSlot = `${hour.toString().padStart(2, '0')}:00`;
  } else if (relativeTimeMatch) {
    const timeWord = relativeTimeMatch[1].toLowerCase();
    if (timeWord === 'tonight' || timeWord === 'dinner' || timeWord === '今晚' || timeWord === '晚餐') timeSlot = '19:00';
    else if (timeWord === 'today' || timeWord === '今天') timeSlot = '18:00';
    else if (timeWord === 'tomorrow' || timeWord === '明天') timeSlot = '18:00';
    else if (timeWord === 'lunch' || timeWord === '午餐' || timeWord === '午饭') timeSlot = '12:00';
    else if (timeWord === 'breakfast' || timeWord === '早餐' || timeWord === '早饭') timeSlot = '08:00';
    else if (timeWord === 'brunch' || timeWord === '早午餐') timeSlot = '10:30';
  }

  // Extract budget
  const budgetPerPersonMatch = normalizedInput.match(
    /budget\s*(\d+)\s*(per\s*person|each|pp)/i
  );
  const budgetTotalMatch = normalizedInput.match(/budget\s*(\d+)/i);
  const priceMatch = normalizedInput.match(/(\d+)\s*(元|rmb|yuan)/i);

  let budget = 0;
  let budgetType: 'total' | 'per_person' = 'total';

  if (budgetPerPersonMatch) {
    budget = parseInt(budgetPerPersonMatch[1]);
    budgetType = 'per_person';
  } else if (budgetTotalMatch) {
    budget = parseInt(budgetTotalMatch[1]);
    budgetType = 'total';
  } else if (priceMatch) {
    budget = parseInt(priceMatch[1]);
    budgetType = 'total';
  } else {
    // Try to find any number that might be budget
    const numbers = normalizedInput.match(/\d+/g);
    if (numbers) {
      const potentialBudget = numbers.find((n) => parseInt(n) > 30 && parseInt(n) < 10000);
      if (potentialBudget) {
        budget = parseInt(potentialBudget);
        budgetType = budget < 100 ? 'per_person' : 'total';
      }
    }
  }

  // Default budget if none found
  if (budget === 0) {
    budget = 100 * partySize;
    budgetType = 'total';
  }

  // Extract preferences and constraints
  const preferences: string[] = [];
  const constraints: string[] = [];

  const preferenceKeywords = [
    { keyword: 'quiet', preference: 'quiet environment' },
    { keyword: 'romantic', preference: 'romantic atmosphere' },
    { keyword: 'private', preference: 'private room' },
    { keyword: 'outdoor', preference: 'outdoor seating' },
    { keyword: 'wifi', preference: 'wifi access' },
    { keyword: 'parking', preference: 'parking available' },
    { keyword: 'counter', preference: 'counter seating' },
    { keyword: 'window', preference: 'window seat' },
  ];

  const constraintKeywords = [
    { keyword: 'no spice', constraint: 'no spicy food' },
    { keyword: 'not spicy', constraint: 'no spicy food' },
    { keyword: 'vegetarian', constraint: 'vegetarian only' },
    { keyword: 'vegan', constraint: 'vegan only' },
    { keyword: 'no pork', constraint: 'no pork' },
    { keyword: 'halal', constraint: 'halal only' },
    { keyword: 'no seafood', constraint: 'no seafood' },
    { keyword: 'gluten free', constraint: 'gluten free' },
  ];

  preferenceKeywords.forEach(({ keyword, preference }) => {
    if (normalizedInput.includes(keyword)) {
      preferences.push(preference);
    }
  });

  constraintKeywords.forEach(({ keyword, constraint }) => {
    if (normalizedInput.includes(keyword)) {
      constraints.push(constraint);
    }
  });

  // Extract cuisine preferences
  const cuisineKeywords = [
    'sushi', 'japanese', 'chinese', 'korean', 'thai', 'italian',
    'french', 'mediterranean', 'indian', 'mexican', 'american',
    'bbq', 'hotpot', 'seafood', 'vegetarian', 'vegan', 'ramen',
    'pizza', 'burger', 'noodles', 'dumplings'
  ];

  let cuisine: string | undefined;
  cuisineKeywords.forEach((keyword) => {
    if (normalizedInput.includes(keyword)) {
      cuisine = keyword;
    }
  });

  // Extract occasion
  const occasionKeywords = [
    { keyword: 'birthday', occasion: 'birthday celebration' },
    { keyword: 'anniversary', occasion: 'anniversary' },
    { keyword: 'date', occasion: 'date night' },
    { keyword: 'business', occasion: 'business meeting' },
    { keyword: 'celebration', occasion: 'celebration' },
    { keyword: 'party', occasion: 'party' },
  ];

  let occasion: string | undefined;
  occasionKeywords.forEach(({ keyword, occasion: occ }) => {
    if (normalizedInput.includes(keyword)) {
      occasion = occ;
    }
  });

  // Default location
  const location = 'Wudaokou, Beijing';

  return {
    partySize,
    timeSlot,
    budget,
    budgetType,
    preferences,
    constraints,
    location,
    cuisine,
    occasion,
  };
}

export function formatParsedDemand(parsed: ParsedDemand): string {
  const lines: string[] = [];

  lines.push(`📊 Party Size: ${parsed.partySize} people`);
  lines.push(`🕐 Time: ${parsed.timeSlot}`);

  if (parsed.budgetType === 'per_person') {
    lines.push(`💰 Budget: ¥${parsed.budget} per person`);
  } else {
    lines.push(`💰 Budget: ¥${parsed.budget} total`);
  }

  if (parsed.cuisine) {
    lines.push(`🍽️ Cuisine: ${parsed.cuisine}`);
  }

  if (parsed.preferences.length > 0) {
    lines.push(`✨ Preferences: ${parsed.preferences.join(', ')}`);
  }

  if (parsed.constraints.length > 0) {
    lines.push(`⚠️ Constraints: ${parsed.constraints.join(', ')}`);
  }

  if (parsed.occasion) {
    lines.push(`🎉 Occasion: ${parsed.occasion}`);
  }

  return lines.join('\n');
}
