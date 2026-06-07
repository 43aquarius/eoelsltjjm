// Core type definitions for Local Contract system

export interface UserDemand {
  id: string;
  rawInput: string;
  parsed: ParsedDemand;
  timestamp: number;
  userId: string;
}

export interface ParsedDemand {
  partySize: number;
  timeSlot: string;
  budget: number;
  budgetType: 'total' | 'per_person';
  preferences: string[];
  constraints: string[];
  location: string;
  cuisine?: string;
  occasion?: string;
}

export interface Merchant {
  id: string;
  name: string;
  category: string;
  cuisine: string;
  location: string;
  coordinates: { lat: number; lng: number };
  rating: number;
  priceRange: 'budget' | 'moderate' | 'premium' | 'luxury';
  capacity: number;
  privateRooms: boolean;
  features: string[];
  currentStatus: MerchantStatus;
  metrics: MerchantMetrics;
}

export interface MerchantStatus {
  seatAvailability: number; // 0-100%
  kitchenLoad: number; // 0-100%
  queueTime: number; // minutes
  inventoryLevel: number; // 0-100%
  isAcceptingOrders: boolean;
}

export interface MerchantMetrics {
  fulfillmentRate: number; // 0-100%
  avgServiceTime: number; // minutes
  customerSatisfaction: number; // 0-5
  totalContracts: number;
  breachRate: number; // 0-100%
}

export interface MatchResult {
  merchant: Merchant;
  scores: MatchScore;
  finalScore: number;
  rank: number;
  isRecommended: boolean;
  riskLevel: 'low' | 'medium' | 'high';
  explanation: string;
}

export interface MatchScore {
  demandFit: number; // 0-100
  fulfillment: number; // 0-100
  supplyIdle: number; // 0-100
  price: number; // 0-100
  distance: number; // 0-100
  riskPenalty: number; // 0-30
}

export interface Offer {
  id: string;
  merchantId: string;
  demandId: string;
  price: number;
  priceType: 'total' | 'per_person';
  promises: ServicePromise[];
  validUntil: number;
  status: 'pending' | 'accepted' | 'rejected' | 'expired';
  createdAt: number;
}

export interface ServicePromise {
  type: 'reservation' | 'price_guarantee' | 'service_quality' | 'time_guarantee' | 'compensation';
  description: string;
  value: string;
  binding: boolean;
}

export interface Contract {
  id: string;
  offerId: string;
  merchantId: string;
  demandId: string;
  userId: string;
  status: ContractStatus;
  terms: ContractTerms;
  timeline: ContractEvent[];
  createdAt: number;
  updatedAt: number;
  qrCode: string;
}

export type ContractStatus =
  | 'draft'
  | 'confirmed'
  | 'merchant_accepted'
  | 'user_arrived'
  | 'service_started'
  | 'completed'
  | 'breached'
  | 'compensated'
  | 'cancelled';

export interface ContractTerms {
  price: number;
  partySize: number;
  timeSlot: string;
  location: string;
  promises: ServicePromise[];
  compensationRules: CompensationRule[];
  specialRequests: string[];
}

export interface CompensationRule {
  trigger: string;
  compensation: string;
  type: 'refund' | 'discount' | 'voucher';
  amount: number;
}

export interface ContractEvent {
  status: ContractStatus;
  timestamp: number;
  description: string;
  actor: 'user' | 'merchant' | 'system';
}

export interface PlatformMetrics {
  totalContracts: number;
  activeContracts: number;
  conversionRate: number; // 0-100
  fulfillmentRate: number; // 0-100
  breachRate: number; // 0-100
  avgMatchScore: number;
  totalUsers: number;
  totalMerchants: number;
  revenueGenerated: number;
  compensationPaid: number;
  contractsByDay: Array<{ date: string; count: number }>;
  metricsByCategory: Array<{ category: string; contracts: number; satisfaction: number }>;
}

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: string;
  shortcut?: string;
  action: () => void;
  category: 'navigation' | 'action' | 'demo';
}
