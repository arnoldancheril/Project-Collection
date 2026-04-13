/**
 * File: src/components/trip/ItineraryTimeline.tsx
 * Purpose: Timeline of itinerary items grouped by day with times for quick scanning.
 */
import React from 'react';
import { View, Text } from 'react-native';
import { BudgetItem } from '../../data/sample';
import { useTheme } from '../../theme/ThemeProvider';

type Props = { items: BudgetItem[] };

export function ItineraryTimeline({ items }: Props) {
  const theme = useTheme();
  const withDates = items.filter(i => i.startAt);
  const sorted = [...withDates].sort((a, b) => Date.parse(a.startAt!) - Date.parse(b.startAt!));
  const groups: Record<string, BudgetItem[]> = {};
  for (const it of sorted) {
    const dateKey = new Date(it.startAt as string).toDateString();
    groups[dateKey] = groups[dateKey] || [];
    groups[dateKey].push(it);
  }

  const dayKeys = Object.keys(groups);
  if (dayKeys.length === 0) {
    return <Text style={{ color: theme.colors.textSecondary }}>No dated items yet.</Text>;
  }

  return (
    <View>
      {dayKeys.map(day => (
        <View key={day} style={{ marginTop: 12 }}>
          <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>{day}</Text>
          {groups[day].map(item => {
            const time = new Date(item.startAt as string).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            return (
              <View key={item.id} style={{ flexDirection: 'row', alignItems: 'center', marginTop: 6 }}>
                <View style={{ width: 64 }}>
                  <Text style={{ color: theme.colors.textSecondary }}>{time}</Text>
                </View>
                <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: theme.colors.primary, marginRight: 8 }} />
                <View style={{ flex: 1 }}>
                  <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{item.label}</Text>
                  <Text style={{ color: theme.colors.textSecondary }}>{item.category} • ${item.total.toLocaleString()}</Text>
                </View>
              </View>
            );
          })}
        </View>
      ))}
    </View>
  );
}


