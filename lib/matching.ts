import type { Merchant, ParsedDemand, MatchResult, MatchScore } from '@/types';

// Calculate demand fit score (0-100)
function calculateDemandFit(merchant: Merchant, demand: ParsedDemand): number {
  let score = 70; // Base score

  // Cuisine match
  if (demand.cuisine) {
    const merchantCuisines = merchant.cuisine.toLowerCase();
    const merchantFeatures = merchant.features.map(f => f.toLowerCase());
    const demandCuisine = demand.cuisine.toLowerCase();

    if (merchantCuisines.includes(demandCuisine) || demandCuisine.includes(merchantCuisines)) {
      score += 20;
    } else if (merchantFeatures.some(f => f.includes(demandCuisine))) {
      score += 10;
    } else {
      score -= 30; // Cuisine mismatch
    }
  }

  // Price range compatibility
  const avgPricePerPerson = demand.budgetType === 'per_person'
    ? demand.budget
    : demand.budget / demand.partySize;

  const priceRanges = {
    budget: [0, 80],
    moderate: [80, 150],
    premium: [150, 300],
    luxury: [300, 1000],
  };

  const merchantRange = priceRanges[merchant.priceRange];
  if (avgPricePerPerson >= merchantRange[0] && avgPricePerPerson <= merchantRange[1]) {
    score += 10;
  } else if (avgPricePerPerson < merchantRange[0]) {
    score += 5; // Cheaper than budget is ok
  } else {
    score -= 10; // More expensive than budget
  }

  // Capacity check
  if (merchant.capacity >= demand.partySize * 2) {
    score += 5; // Adequate capacity
  } else if (merchant.capacity < demand.partySize) {
    score -= 20; // Not enough capacity
  }

  // Private room requirement
  const needsPrivateRoom = demand.preferences.some(p =>
    p.toLowerCase().includes('private') || p.toLowerCase().includes('private room')
  );
  if (needsPrivateRoom) {
    if (merchant.privateRooms) {
      score += 10;
    } else {
      score -= 25; // Critical requirement not met
    }
  }

  // Feature matching
  const featureMatches = demand.preferences.filter(pref =>
    merchant.features.some(f =>
      f.toLowerCase().includes(pref.toLowerCase()) ||
      pref.toLowerCase().includes(f.toLowerCase())
    )
  );
  score += featureMatches.length * 3;

  // Constraint checking
  const hasNoSpice = demand.constraints.some(c =>
    c.toLowerCase().includes('spice') || c.toLowerCase().includes('spicy')
  );
  if (hasNoSpice && merchant.features.some(f =>
    f.toLowerCase().includes('spicy') || f.toLowerCase().includes('spice')
  )) {
    score -= 15;
  }

  const hasVegetarian = demand.constraints.some(c =>
    c.toLowerCase().includes('vegetarian') || c.toLowerCase().includes('vegan')
  );
  if (hasVegetarian && !merchant.features.some(f =>
    f.toLowerCase().includes('vegetarian') || f.toLowerCase().includes('vegan')
  )) {
    score -= 10;
  }

  return Math.max(0, Math.min(100, score));
}

// Calculate fulfillment score based on merchant metrics (0-100)
function calculateFulfillment(merchant: Merchant): number {
  const metrics = merchant.metrics;

  // Weighted average of fulfillment metrics
  const fulfillmentScore =
    metrics.fulfillmentRate * 0.4 +
    (metrics.customerSatisfaction / 5) * 100 * 0.3 +
    (100 - metrics.breachRate) * 0.2 +
    Math.min(100, (metrics.totalContracts / 500) * 100) * 0.1;

  return Math.max(0, Math.min(100, fulfillmentScore));
}

// Calculate supply-idle score - how well can merchant absorb demand (0-100)
function calculateSupplyIdle(merchant: Merchant): number {
  const status = merchant.currentStatus;

  // Higher availability = better score
  const seatScore = status.seatAvailability;
  const kitchenScore = 100 - status.kitchenLoad;
  const queueScore = Math.max(0, 100 - status.queueTime * 2);
  const inventoryScore = status.inventoryLevel;

  // Weighted combination
  const idleScore =
    seatScore * 0.35 +
    kitchenScore * 0.3 +
    queueScore * 0.2 +
    inventoryScore * 0.15;

  return Math.max(0, Math.min(100, idleScore));
}

// Calculate price competitiveness score (0-100)
function calculatePriceScore(merchant: Merchant, demand: ParsedDemand): number {
  const baseScore = 70;

  // Adjust based on price range
  const priceRangeScores = {
    budget: 85,
    moderate: 75,
    premium: 60,
    luxury: 45,
  };

  let score = priceRangeScores[merchant.priceRange];

  // Bonus for high seat availability (can offer better deals)
  if (merchant.currentStatus.seatAvailability > 70) {
    score += 10;
  }

  // Penalty for high kitchen load (less room for negotiation)
  if (merchant.currentStatus.kitchenLoad > 70) {
    score -= 10;
  }

  return Math.max(0, Math.min(100, score));
}

// Calculate distance score (0-100) - simplified for demo
function calculateDistanceScore(merchant: Merchant): number {
  // In real app, would calculate actual distance
  // For demo, assume all merchants are in Wudaokou area
  return 85; // Good default score
}

// Calculate risk penalty (0-30)
function calculateRiskPenalty(merchant: Merchant): number {
  let penalty = 0;

  // High queue time
  if (merchant.currentStatus.queueTime > 20) {
    penalty += 5;
  }

  // Low inventory
  if (merchant.currentStatus.inventoryLevel < 50) {
    penalty += 8;
  }

  // High kitchen load
  if (merchant.currentStatus.kitchenLoad > 80) {
    penalty += 7;
  }

  // Low fulfillment rate
  if (merchant.metrics.fulfillmentRate < 85) {
    penalty += 10;
  }

  // High breach rate
  if (merchant.metrics.breachRate > 5) {
    penalty += 10;
  }

  return Math.min(30, penalty);
}

// Main matching function
export function matchMerchants(
  merchants: Merchant[],
  demand: ParsedDemand
): MatchResult[] {
  const results: MatchResult[] = merchants.map((merchant) => {
    const scores: MatchScore = {
      demandFit: calculateDemandFit(merchant, demand),
      fulfillment: calculateFulfillment(merchant),
      supplyIdle: calculateSupplyIdle(merchant),
      price: calculatePriceScore(merchant, demand),
      distance: calculateDistanceScore(merchant),
      riskPenalty: calculateRiskPenalty(merchant),
    };

    // Calculate final score with weights
    const finalScore =
      scores.demandFit * 0.30 +
      scores.fulfillment * 0.25 +
      scores.supplyIdle * 0.20 +
      scores.price * 0.15 +
      scores.distance * 0.10 -
      scores.riskPenalty;

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' = 'low';
    if (scores.riskPenalty > 15) riskLevel = 'high';
    else if (scores.riskPenalty > 8) riskLevel = 'medium';

    // Generate explanation
    const explanationParts: string[] = [];

    if (scores.demandFit >= 80) {
      explanationParts.push('Excellent demand match');
    } else if (scores.demandFit >= 60) {
      explanationParts.push('Good demand match');
    } else {
      explanationParts.push('Moderate demand match');
    }

    if (scores.supplyIdle >= 80) {
      explanationParts.push('high availability');
    } else if (scores.supplyIdle < 50) {
      explanationParts.push('limited availability');
    }

    if (scores.fulfillment >= 90) {
      explanationParts.push('proven track record');
    }

    if (scores.price >= 80) {
      explanationParts.push('competitive pricing');
    }

    const explanation = explanationParts.join(', ') + '.';

    return {
      merchant,
      scores,
      finalScore: Math.round(finalScore * 10) / 10,
      rank: 0, // Will be set after sorting
      isRecommended: false, // Will be set after sorting
      riskLevel,
      explanation,
    };
  });

  // Sort by final score (descending)
  results.sort((a, b) => b.finalScore - a.finalScore);

  // Assign ranks and recommendations
  results.forEach((result, index) => {
    result.rank = index + 1;
    result.isRecommended = index < 3; // Top 3 are recommended
  });

  return results;
}

// Get top N matches
export function getTopMatches(
  merchants: Merchant[],
  demand: ParsedDemand,
  count: number = 5
): MatchResult[] {
  const allMatches = matchMerchants(merchants, demand);
  return allMatches.slice(0, count);
}
