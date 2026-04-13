/**
 * File: app/profile.tsx
 * Purpose: Modern profile screen with richer insights, quick actions, preferences, and secure account controls.
 */
import React from 'react';
import { View, Text, StyleSheet, Switch } from 'react-native';
import { useTheme } from '../src/theme/ThemeProvider';
import { Card } from '../src/components/Card';
import { Screen } from '../src/components/Screen';
import { Avatar } from '../src/components/Avatar';
import { ListItem } from '../src/components/ListItem';
import { ProgressBar } from '../src/components/ProgressBar';
import { useTrips } from '../src/state/TripsStore';
import { useAuth } from '../src/state/AuthContext';
import { computeBalances } from '../src/utils/balances';
import { Button } from '../src/components/Button';

export default function ProfileScreen() {
  const theme = useTheme();
  const { signOut } = useAuth();
  const [notifications, setNotifications] = React.useState(true);
  const { trips, selectedTripId } = useTrips();
  const current = trips.find(t => t.id === selectedTripId) || trips[0];
  const balances = current ? computeBalances(current) : [];
  return (
    <Screen scroll>
      <Text style={[styles.title, { color: theme.colors.textPrimary }]}>Profile</Text>
      <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>Manage preferences, payments, and privacy settings.</Text>

      <Card style={{ marginTop: 16 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Avatar name="You" size={56} />
          <View style={{ marginLeft: 12 }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '700', fontSize: 16 }}>You</Text>
            <Text style={{ color: theme.colors.textSecondary }}>Member since 2025</Text>
          </View>
        </View>
        <View style={{ height: 12 }} />
        <Text style={{ color: theme.colors.textSecondary, marginBottom: 4 }}>Overall savings progress</Text>
        <ProgressBar progress={0.35} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <ListItem title="Notifications" right={<Switch value={notifications} onValueChange={setNotifications} />} />
        <ListItem title="Payment methods" subtitle="Manage cards for contributions" onPress={() => {}} />
        <ListItem title="Privacy" subtitle="Control invite and visibility settings" onPress={() => {}} />
        <ListItem title="Export data" subtitle="Get a copy of your trip data" onPress={() => {}} />
        <ListItem title="Security" subtitle="Change password and 2FA" onPress={() => {}} />
        <ListItem title="Sign out" subtitle="Log out of this device" onPress={signOut} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700', marginBottom: 8 }}>Trip preferences</Text>
        <ListItem title="Default split method" subtitle="Even split" onPress={() => {}} />
        <ListItem title="Currency" subtitle="USD ($)" onPress={() => {}} />
        <ListItem title="Default tip" subtitle="18%" onPress={() => {}} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700', marginBottom: 8 }}>Current Trip Balances</Text>
        {balances.length === 0 ? (
          <Text style={{ color: theme.colors.textSecondary }}>No balances yet.</Text>
        ) : (
          balances.map(b => (
            <ListItem key={b.memberId} title={b.name} subtitle={b.balance >= 0 ? `Should receive $${b.balance.toFixed(2)}` : `Owes $${Math.abs(b.balance).toFixed(2)}`} />
          ))
        )}
      </Card>
      <Card style={{ marginTop: 16 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700', marginBottom: 8 }}>Support</Text>
        <ListItem title="Help Center" subtitle="FAQs and guides" onPress={() => {}} />
        <ListItem title="Contact Us" subtitle="Get in touch" onPress={() => {}} />
      </Card>

      <View style={{ height: 24 }} />
      <Button label="Log out" onPress={signOut} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 22, fontWeight: '700' },
});


