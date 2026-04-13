/**
 * File: src/state/TripsStore.tsx
 * Purpose: Centralized state for trips, selection, and mutations (create trip, select, add member/item, add contribution, add expense, edit itinerary item).
 */
import React, { createContext, useContext, useMemo, useState, useCallback } from 'react';
import { trips as initialTrips, Trip, TripMember, BudgetItem, Contribution, Expense } from '../data/sample';

export type TripsState = {
  trips: Trip[];
  selectedTripId?: string;
  selectTrip: (tripId: string) => void;
  createTrip: (trip: Pick<Trip, 'name' | 'destination' | 'dateRange'> & { goalBudget?: number; startDate?: string; endDate?: string; members?: TripMember[]; itinerary?: BudgetItem[] }) => Trip;
  addMember: (tripId: string, member: TripMember) => void;
  addItineraryItem: (tripId: string, item: BudgetItem) => void;
  addContribution: (tripId: string, contribution: Contribution) => void;
  addExpense: (tripId: string, expense: Expense) => void;
  editItineraryItem: (tripId: string, item: Partial<BudgetItem> & { id: string }) => void;
};

const TripsContext = createContext<TripsState | null>(null);

export function useTrips() {
  const ctx = useContext(TripsContext);
  if (!ctx) throw new Error('useTrips must be used within TripsProvider');
  return ctx;
}

export function TripsProvider({ children }: { children: React.ReactNode }) {
  const [trips, setTrips] = useState<Trip[]>(initialTrips);
  const [selectedTripId, setSelectedTripId] = useState<string | undefined>(trips[0]?.id);

  const selectTrip = useCallback((tripId: string) => setSelectedTripId(tripId), []);

  const createTrip: TripsState['createTrip'] = useCallback((input) => {
    const newTrip: Trip = {
      id: Math.random().toString(36).slice(2),
      name: input.name,
      destination: input.destination,
      dateRange: input.dateRange,
      goalBudget: input.goalBudget,
      startDate: input.startDate,
      endDate: input.endDate,
      members: input.members ?? [],
      itinerary: input.itinerary ?? [],
    };
    setTrips(prev => [newTrip, ...prev]);
    setSelectedTripId(newTrip.id);
    return newTrip;
  }, []);

  const addMember: TripsState['addMember'] = useCallback((tripId, member) => {
    setTrips(prev => prev.map(t => t.id === tripId ? { ...t, members: [ ...t.members, member ] } : t));
  }, []);

  const addItineraryItem: TripsState['addItineraryItem'] = useCallback((tripId, item) => {
    setTrips(prev => prev.map(t => t.id === tripId ? { ...t, itinerary: [ ...t.itinerary, item ] } : t));
  }, []);

  const addContribution: TripsState['addContribution'] = useCallback((tripId, contribution) => {
    setTrips(prev => prev.map(t => t.id === tripId ? { ...t, contributions: [ ...(t.contributions ?? []), contribution ] } : t));
  }, []);

  const addExpense: TripsState['addExpense'] = useCallback((tripId, expense) => {
    setTrips(prev => prev.map(t => t.id === tripId ? { ...t, expenses: [ ...(t.expenses ?? []), expense ] } : t));
  }, []);

  const editItineraryItem: TripsState['editItineraryItem'] = useCallback((tripId, item) => {
    setTrips(prev => prev.map(t => {
      if (t.id !== tripId) return t;
      return { ...t, itinerary: t.itinerary.map(it => it.id === item.id ? { ...it, ...item } as BudgetItem : it) };
    }));
  }, []);

  const value = useMemo(() => ({ trips, selectedTripId, selectTrip, createTrip, addMember, addItineraryItem, addContribution, addExpense, editItineraryItem }), [trips, selectedTripId, selectTrip, createTrip, addMember, addItineraryItem, addContribution, addExpense, editItineraryItem]);

  return <TripsContext.Provider value={value}>{children}</TripsContext.Provider>;
}


