/**
 * File: src/components/trip/TripOverviewCard.tsx
 * Purpose: Overview card showing destination details, budget chart, and category totals for a trip.
 */
import React from 'react';
import { View, Text } from 'react-native';
import { Trip } from '../../data/sample';
import { useTheme } from '../../theme/ThemeProvider';
import { Card } from '../Card';
import { PieChart } from '../PieChart';
import { categoryColor, categoryTotals } from '../../utils/budget';

type Props = { trip: Trip };

export function TripOverviewCard({ trip }: Props) {
  const theme = useTheme();
  const total = trip.itinerary.reduce((s, i) => s + i.total, 0);
  const goal = trip.goalBudget ?? total;
  const progress = goal > 0 ? Math.min(1, total / goal) : 1;
  return (
    <>
      <Card style={{ marginTop: 16 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Summary</Text>
        <View style={{ height: 8 }} />
        <Text style={{ color: theme.colors.textSecondary }}>Destination: {trip.destination}</Text>
        <Text style={{ color: theme.colors.textSecondary }}>Dates: {trip.dateRange}</Text>
        <Text style={{ color: theme.colors.textSecondary }}>Members: {trip.members.length}</Text>
        <Text style={{ color: theme.colors.textSecondary }}>Planned Items: {trip.itinerary.length}</Text>
      </Card>
      <Card style={{ marginTop: 16, alignItems: 'center' }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Budget Breakdown</Text>
        <View style={{ height: 12 }} />
        <PieChart size={180} thickness={20} segments={trip.itinerary.map(i => ({ value: i.total, color: categoryColor(i.category) }))} />
        <Text style={{ color: theme.colors.textSecondary, marginTop: 12 }}>Total plan: ${total.toLocaleString()} {goal ? `of $${goal.toLocaleString()}` : ''}</Text>
        <View style={{ height: 12 }} />
        <View>
          {categoryTotals(trip.itinerary).map(ct => (
            <View key={ct.category} style={{ flexDirection: 'row', alignItems: 'center', marginTop: 6 }}>
              <View style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: categoryColor(ct.category as any), marginRight: 8 }} />
              <Text style={{ color: theme.colors.textSecondary, flex: 1 }}>{ct.category}</Text>
              <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>${ct.total.toLocaleString()}</Text>
            </View>
          ))}
        </View>
        <View style={{ height: 8 }} />
        <Text style={{ color: theme.colors.textSecondary }}>Goal progress: {Math.round(progress * 100)}%</Text>
      </Card>
    </>
  );
}


