import type { Merchant, MatchResult, Offer, ServicePromise, ParsedDemand } from '@/types';

// Generate unique ID
function generateId(): string {
  return `offer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Generate service promises based on merchant capabilities and demand
function generatePromises(
  merchant: Merchant,
  demand: ParsedDemand,
  matchResult: MatchResult
): ServicePromise[] {
  const promises: ServicePromise[] = [];

  // Always include price guarantee
  promises.push({
    type: 'price_guarantee',
    description: 'Price locked at quoted amount',
    value: 'No surprise charges',
    binding: true,
  });

  // Reservation promise based on seat availability
  if (merchant.currentStatus.seatAvailability > 50) {
    promises.push({
      type: 'reservation',
      description: 'Guaranteed seating reservation',
      value: `${demand.partySize} seats at ${demand.timeSlot}`,
      binding: true,
    });
  }

  // Time guarantee based on kitchen load
  if (merchant.currentStatus.kitchenLoad < 70) {
    promises.push({
      type: 'time_guarantee',
      description: 'Service time commitment',
      value: `Food served within ${merchant.metrics.avgServiceTime} minutes`,
      binding: true,
    });
  }

  // Quality promise based on rating
  if (merchant.rating >= 4.5) {
    promises.push({
      type: 'service_quality',
      description: 'Premium service guarantee',
      value: `${merchant.rating}+ star experience`,
      binding: true,
    });
  }

  // Private room promise
  const needsPrivateRoom = demand.preferences.some(p =>
    p.toLowerCase().includes('private') || p.toLowerCase().includes('private room')
  );
  if (needsPrivateRoom && merchant.privateRooms) {
    promises.push({
      type: 'service_quality',
      description: 'Private room reserved',
      value: 'Exclusive dining space',
      binding: true,
    });
  }

  // Compensation promise based on risk level
  if (matchResult.riskLevel === 'low') {
    promises.push({
      type: 'compensation',
      description: 'Breach protection guarantee',
      value: '200% compensation for service failure',
      binding: true,
    });
  } else if (matchResult.riskLevel === 'medium') {
    promises.push({
      type: 'compensation',
      description: 'Service guarantee',
      value: '100% refund for major issues',
      binding: true,
    });
  }

  // Add special promises for high-availability merchants
  if (merchant.currentStatus.seatAvailability > 80) {
    promises.push({
      type: 'service_quality',
      description: 'VIP treatment',
      value: 'Complimentary appetizer',
      binding: false,
    });
  }

  // Fresh food promise for high inventory
  if (merchant.currentStatus.inventoryLevel > 85) {
    promises.push({
      type: 'service_quality',
      description: 'Fresh ingredients guarantee',
      value: 'Same-day fresh preparation',
      binding: true,
    });
  }

  return promises;
}

// Calculate offer price based on merchant state and demand
function calculatePrice(
  merchant: Merchant,
  demand: ParsedDemand,
  matchResult: MatchResult
): { price: number; priceType: 'total' | 'per_person' } {
  // Base price estimation
  const priceMultipliers = {
    budget: 60,
    moderate: 120,
    premium: 220,
    luxury: 450,
  };

  const basePricePerPerson = priceMultipliers[merchant.priceRange];

  // Adjust based on availability (higher availability = better deal)
  let discountFactor = 1.0;
  if (merchant.currentStatus.seatAvailability > 70) {
    discountFactor = 0.9; // 10% discount for high availability
  } else if (merchant.currentStatus.seatAvailability > 50) {
    discountFactor = 0.95; // 5% discount
  }

  // Adjust based on kitchen load (lower load = better deal)
  if (merchant.currentStatus.kitchenLoad < 40) {
    discountFactor *= 0.95; // Additional 5% discount
  }

  // Adjust based on match score (better match = better deal)
  if (matchResult.finalScore > 85) {
    discountFactor *= 0.98;
  }

  const pricePerPerson = Math.round(basePricePerPerson * discountFactor);

  // Determine price type
  const priceType: 'total' | 'per_person' =
    demand.budgetType === 'per_person' ? 'per_person' : 'total';

  if (priceType === 'total') {
    return {
      price: pricePerPerson * demand.partySize,
      priceType: 'total',
    };
  }

  return {
    price: pricePerPerson,
    priceType: 'per_person',
  };
}

// Generate a single offer
export function generateOffer(
  merchant: Merchant,
  demand: ParsedDemand,
  matchResult: MatchResult,
  demandId: string
): Offer {
  const { price, priceType } = calculatePrice(merchant, demand, matchResult);
  const promises = generatePromises(merchant, demand, matchResult);

  // Offer valid for 30 minutes
  const validUntil = Date.now() + 30 * 60 * 1000;

  return {
    id: generateId(),
    merchantId: merchant.id,
    demandId,
    price,
    priceType,
    promises,
    validUntil,
    status: 'pending',
    createdAt: Date.now(),
  };
}

// Generate offers for multiple merchants
export function generateOffers(
  matchResults: MatchResult[],
  demand: ParsedDemand,
  demandId: string
): Offer[] {
  return matchResults.map((matchResult) =>
    generateOffer(matchResult.merchant, demand, matchResult, demandId)
  );
}

// Get offer summary for display
export function getOfferSummary(offer: Offer): string {
  const priceDisplay =
    offer.priceType === 'per_person'
      ? `¥${offer.price}/person`
      : `¥${offer.price} total`;

  const promiseCount = offer.promises.filter((p) => p.binding).length;

  return `${priceDisplay} · ${promiseCount} guarantees`;
}
