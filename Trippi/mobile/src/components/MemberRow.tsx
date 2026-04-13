/**
 * File: src/components/MemberRow.tsx
 * Purpose: Detailed member row with avatar, role/bio, and an action button.
 */
import React from 'react';
import { View, Text, Pressable } from 'react-native';
import { Avatar } from './Avatar';
import { useTheme } from '../theme/ThemeProvider';

type Props = { id: string; name: string; avatarColor?: string; onView?: (id: string) => void };

export function MemberRow({ id, name, avatarColor, onView }: Props) {
  const theme = useTheme();
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 12, borderTopWidth: 1, borderTopColor: theme.colors.border, paddingTop: 8 }}>
      <Avatar name={name} color={avatarColor} size={40} />
      <View style={{ marginLeft: 12, flex: 1 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{name}</Text>
        <Text style={{ color: theme.colors.textSecondary }}>Role: Traveler • Pref: Even split</Text>
      </View>
      <Pressable onPress={() => onView && onView(id)} style={({ pressed }) => ({ opacity: pressed ? 0.7 : 1 })}>
        <Text style={{ color: theme.colors.primary, fontWeight: '600' }}>View</Text>
      </Pressable>
    </View>
  );
}


