/**
 * File: app/(tabs)/trips.tsx
 * Purpose: Trips tab now mirrors the previous Home page: large trip cards grid/list with quick access
 *          to details and a floating + button for creating a trip.
 */
import React from 'react';
import { View, Text, StyleSheet, FlatList, Pressable } from 'react-native';
import { Screen } from '../../src/components/Screen';
import { useTheme } from '../../src/theme/ThemeProvider';
import { useTrips } from '../../src/state/TripsStore';
import { Link } from 'expo-router';
import { TripCard } from '../../src/components/TripCard';
import { FAB } from '../../src/components/FAB';

export default function TripsTab() {
  const theme = useTheme();
  const { trips } = useTrips();
  const sorted = React.useMemo(() => {
    const parseDate = (t: string | undefined) => (t ? Date.parse(t) : Number.MAX_SAFE_INTEGER);
    return [...trips].sort((a, b) => parseDate(a.startDate) - parseDate(b.startDate));
  }, [trips]);
  return (
    <Screen>
      <Text style={[styles.title, { color: theme.colors.textPrimary }]}>Upcoming Trips</Text>
      <FlatList
        contentContainerStyle={{ paddingTop: 12, paddingBottom: 90 }}
        data={sorted}
        keyExtractor={(t) => t.id}
        renderItem={({ item }) => (
          <Link href={{ pathname: '/trip/[id]', params: { id: item.id } }} asChild>
            <Pressable>
              <TripCard
                name={item.name}
                destination={item.destination}
                dateRange={item.dateRange}
                memberNames={(item.members || []).map(m => m.name)}
              />
            </Pressable>
          </Link>
        )}
      />
      <Link href="/trips/create" asChild>
        <FAB onPress={() => {}} />
      </Link>
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: { fontSize: 22, fontWeight: '700' },
});


