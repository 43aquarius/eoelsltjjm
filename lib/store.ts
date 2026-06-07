'use client';

import { create } from 'zustand';
import type {
  UserDemand,
  ParsedDemand,
  MatchResult,
  Offer,
  Contract,
  Merchant,
  PlatformMetrics,
} from '@/types';
import { merchants as mockMerchants, platformMetrics as mockMetrics } from './mock-data';

interface AppState {
  // Navigation
  currentPage: string;
  setCurrentPage: (page: string) => void;

  // Demand creation
  currentDemand: ParsedDemand | null;
  demandInput: string;
  setDemandInput: (input: string) => void;
  setCurrentDemand: (demand: ParsedDemand | null) => void;

  // Matching
  matchResults: MatchResult[];
  selectedMatch: MatchResult | null;
  setMatchResults: (results: MatchResult[]) => void;
  setSelectedMatch: (match: MatchResult | null) => void;

  // Offers
  currentOffers: Offer[];
  selectedOffer: Offer | null;
  setCurrentOffers: (offers: Offer[]) => void;
  setSelectedOffer: (offer: Offer | null) => void;

  // Contract
  currentContract: Contract | null;
  setCurrentContract: (contract: Contract | null) => void;

  // Merchant
  merchants: Merchant[];
  selectedMerchant: Merchant | null;
  setSelectedMerchant: (merchant: Merchant | null) => void;

  // Platform metrics
  metrics: PlatformMetrics;
  updateMetrics: (metrics: Partial<PlatformMetrics>) => void;

  // Command menu
  isCommandMenuOpen: boolean;
  setCommandMenuOpen: (open: boolean) => void;

  // Demo mode
  isDemoMode: boolean;
  simulationTime: Date;
  advanceSimulation: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  // Navigation
  currentPage: 'home',
  setCurrentPage: (page) => set({ currentPage: page }),

  // Demand creation
  currentDemand: null,
  demandInput: '',
  setDemandInput: (input) => set({ demandInput: input }),
  setCurrentDemand: (demand) => set({ currentDemand: demand }),

  // Matching
  matchResults: [],
  selectedMatch: null,
  setMatchResults: (results) => set({ matchResults: results }),
  setSelectedMatch: (match) => set({ selectedMatch: match }),

  // Offers
  currentOffers: [],
  selectedOffer: null,
  setCurrentOffers: (offers) => set({ currentOffers: offers }),
  setSelectedOffer: (offer) => set({ selectedOffer: offer }),

  // Contract
  currentContract: null,
  setCurrentContract: (contract) => set({ currentContract: contract }),

  // Merchant
  merchants: mockMerchants,
  selectedMerchant: null,
  setSelectedMerchant: (merchant) => set({ selectedMerchant: merchant }),

  // Platform metrics
  metrics: mockMetrics,
  updateMetrics: (newMetrics) =>
    set((state) => ({
      metrics: { ...state.metrics, ...newMetrics },
    })),

  // Command menu
  isCommandMenuOpen: false,
  setCommandMenuOpen: (open) => set({ isCommandMenuOpen: open }),

  // Demo mode
  isDemoMode: true,
  simulationTime: new Date('2026-06-07T18:00:00'),
  advanceSimulation: () =>
    set((state) => ({
      simulationTime: new Date(state.simulationTime.getTime() + 15 * 60 * 1000),
    })),
}));
