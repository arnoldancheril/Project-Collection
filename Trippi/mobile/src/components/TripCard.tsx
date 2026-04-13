/**
 * File: src/components/TripCard.tsx
 * Purpose: Hero-style trip card showing destination photo, dates, member avatars, and quick stats.
 */
import React from 'react';
import { View, Text, ImageBackground } from 'react-native';
import { Card } from './Card';
import { useTheme } from '../theme/ThemeProvider';
import { Avatar } from './Avatar';

type Props = { name: string; destination: string; dateRange: string; memberNames?: string[] };

export function TripCard({ name, destination, dateRange, memberNames = [] }: Props) {
  const theme = useTheme();
  return (
    <Card style={{ marginBottom: 16, padding: 0 }}>
      <ImageBackground source={require('../../assets/Trippi_logo.png')} imageStyle={{ opacity: 0.08 }} style={{ padding: 20 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '800', fontSize: 22 }}>{name}</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6, fontSize: 14 }}>{destination} • {dateRange}</Text>
        <View style={{ height: 14 }} />
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ flexDirection: 'row' }}>
            {memberNames.slice(0, 4).map((n, i) => (
              <View key={n + i} style={{ marginRight: 6 }}>
                <Avatar name={n} size={28} />
              </View>
            ))}
          </View>
          <View style={{ flex: 1 }} />
          <View style={{ paddingVertical: 6, paddingHorizontal: 10, backgroundColor: theme.colors.surface, borderRadius: 999, borderWidth: 1, borderColor: theme.colors.border }}>
            <Text style={{ color: theme.colors.textSecondary, fontSize: 12 }}>Tap to view details</Text>
          </View>
        </View>
      </ImageBackground>
    </Card>
  );
}


