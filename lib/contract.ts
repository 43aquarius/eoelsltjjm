import type {
  Contract,
  ContractStatus,
  ContractTerms,
  ContractEvent,
  CompensationRule,
  Offer,
  Merchant,
  ParsedDemand,
} from '@/types';

// Generate unique ID
function generateId(): string {
  return `contract_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Generate QR code data (simplified for demo)
function generateQRCode(contractId: string): string {
  return `data:image/svg+xml,${encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <rect fill="#000" width="100" height="100"/>
      <rect fill="#fff" x="10" y="10" width="80" height="80"/>
      <text x="50" y="55" text-anchor="middle" font-size="8" font-family="monospace">${contractId.slice(0, 12)}</text>
    </svg>
  `)}`;
}

// Generate compensation rules based on merchant risk
function generateCompensationRules(
  merchant: Merchant,
  offer: Offer
): CompensationRule[] {
  const rules: CompensationRule[] = [];

  // Standard breach rules
  rules.push({
    trigger: 'Reservation not honored',
    compensation: 'Full refund + 50% compensation',
    type: 'refund',
    amount: offer.price * 1.5,
  });

  rules.push({
    trigger: 'Price increased after booking',
    compensation: 'Price difference refunded 2x',
    type: 'refund',
    amount: offer.price * 0.3,
  });

  rules.push({
    trigger: 'Wait time exceeds 30 minutes past reserved time',
    compensation: '20% discount on total bill',
    type: 'discount',
    amount: offer.price * 0.2,
  });

  // Quality-based rules
  if (merchant.rating >= 4.5) {
    rules.push({
      trigger: 'Service quality below 4-star standard',
      compensation: '30% refund',
      type: 'refund',
      amount: offer.price * 0.3,
    });
  }

  // High breach rate merchants have stricter penalties
  if (merchant.metrics.breachRate > 3) {
    rules.push({
      trigger: 'Any contract violation',
      compensation: 'Full refund + 100% compensation',
      type: 'refund',
      amount: offer.price * 2,
    });
  }

  // Add voucher option for minor issues
  rules.push({
    trigger: 'Minor service issues (e.g., slight delay)',
    compensation: '¥50 voucher for next visit',
    type: 'voucher',
    amount: 50,
  });

  return rules;
}

// Create a new contract from an offer
export function createContract(
  offer: Offer,
  merchant: Merchant,
  demand: ParsedDemand,
  userId: string = 'user_demo'
): Contract {
  const contractId = generateId();

  const terms: ContractTerms = {
    price: offer.price,
    partySize: demand.partySize,
    timeSlot: demand.timeSlot,
    location: merchant.location,
    promises: offer.promises,
    compensationRules: generateCompensationRules(merchant, offer),
    specialRequests: [...demand.preferences, ...demand.constraints],
  };

  const now = Date.now();
  const initialEvent: ContractEvent = {
    status: 'draft',
    timestamp: now,
    description: 'Contract created, awaiting merchant confirmation',
    actor: 'system',
  };

  return {
    id: contractId,
    offerId: offer.id,
    merchantId: merchant.id,
    demandId: `demand_${now}`,
    userId,
    status: 'draft',
    terms,
    timeline: [initialEvent],
    createdAt: now,
    updatedAt: now,
    qrCode: generateQRCode(contractId),
  };
}

// Update contract status
export function updateContractStatus(
  contract: Contract,
  newStatus: ContractStatus,
  actor: 'user' | 'merchant' | 'system',
  description?: string
): Contract {
  const statusDescriptions: Record<ContractStatus, string> = {
    draft: 'Contract created',
    confirmed: 'Contract confirmed by merchant',
    merchant_accepted: 'Merchant accepted the contract',
    user_arrived: 'User arrived at venue',
    service_started: 'Service has begun',
    completed: 'Service completed successfully',
    breached: 'Contract terms violated',
    compensated: 'Compensation processed',
    cancelled: 'Contract cancelled',
  };

  const event: ContractEvent = {
    status: newStatus,
    timestamp: Date.now(),
    description: description || statusDescriptions[newStatus],
    actor,
  };

  return {
    ...contract,
    status: newStatus,
    timeline: [...contract.timeline, event],
    updatedAt: Date.now(),
  };
}

// Check if contract can transition to next status
export function canTransitionTo(
  currentStatus: ContractStatus,
  nextStatus: ContractStatus
): boolean {
  const validTransitions: Record<ContractStatus, ContractStatus[]> = {
    draft: ['confirmed', 'cancelled'],
    confirmed: ['merchant_accepted', 'cancelled'],
    merchant_accepted: ['user_arrived', 'cancelled'],
    user_arrived: ['service_started', 'breached'],
    service_started: ['completed', 'breached'],
    completed: [],
    breached: ['compensated'],
    compensated: [],
    cancelled: [],
  };

  return validTransitions[currentStatus].includes(nextStatus);
}

// Get next possible statuses
export function getNextStatuses(currentStatus: ContractStatus): ContractStatus[] {
  const validTransitions: Record<ContractStatus, ContractStatus[]> = {
    draft: ['confirmed', 'cancelled'],
    confirmed: ['merchant_accepted', 'cancelled'],
    merchant_accepted: ['user_arrived', 'cancelled'],
    user_arrived: ['service_started', 'breached'],
    service_started: ['completed', 'breached'],
    completed: [],
    breached: ['compensated'],
    compensated: [],
    cancelled: [],
  };

  return validTransitions[currentStatus];
}

// Get status display info
export function getStatusInfo(status: ContractStatus): {
  label: string;
  color: string;
  description: string;
} {
  const statusInfo: Record<ContractStatus, { label: string; color: string; description: string }> = {
    draft: {
      label: 'Draft',
      color: 'gray',
      description: 'Awaiting merchant confirmation',
    },
    confirmed: {
      label: 'Confirmed',
      color: 'blue',
      description: 'Merchant has confirmed',
    },
    merchant_accepted: {
      label: 'Accepted',
      color: 'indigo',
      description: 'Ready for arrival',
    },
    user_arrived: {
      label: 'Arrived',
      color: 'purple',
      description: 'Guest has arrived',
    },
    service_started: {
      label: 'In Progress',
      color: 'amber',
      description: 'Service underway',
    },
    completed: {
      label: 'Completed',
      color: 'green',
      description: 'Successfully completed',
    },
    breached: {
      label: 'Breached',
      color: 'red',
      description: 'Terms violated',
    },
    compensated: {
      label: 'Compensated',
      color: 'orange',
      description: 'Compensation processed',
    },
    cancelled: {
      label: 'Cancelled',
      color: 'gray',
      description: 'Contract cancelled',
    },
  };

  return statusInfo[status];
}

// Calculate contract progress percentage
export function getContractProgress(contract: Contract): number {
  const progressMap: Record<ContractStatus, number> = {
    draft: 10,
    confirmed: 25,
    merchant_accepted: 40,
    user_arrived: 60,
    service_started: 80,
    completed: 100,
    breached: 50,
    compensated: 100,
    cancelled: 0,
  };

  return progressMap[contract.status];
}

// Get breach compensation amount
export function getBreachCompensation(
  contract: Contract,
  breachType: string
): number {
  const rule = contract.terms.compensationRules.find(
    (r) => r.trigger.toLowerCase().includes(breachType.toLowerCase())
  );

  return rule?.amount || contract.terms.price * 0.5; // Default 50% refund
}
