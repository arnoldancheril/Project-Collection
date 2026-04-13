/**
 * File: app/budget.tsx
 * Purpose: Budget screen with all-trips overview, annual goals, per-trip goal progress, category breakdown, per-person allocations, and contribution actions.
 */
import React from 'react';
import { View, Text, StyleSheet, Modal } from 'react-native';
import { useTheme } from '../src/theme/ThemeProvider';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { Screen } from '../src/components/Screen';
import { ProgressBar } from '../src/components/ProgressBar';
import { TextField } from '../src/components/TextField';
import { useTrips } from '../src/state/TripsStore';
import { PieChart } from '../src/components/PieChart';
import { formatCurrency } from '../src/utils/format';
import { categoryTotals, categoryColor } from '../src/utils/budget';
import { CategoryLegend } from '../src/components/CategoryLegend';
import { computeBalances } from '../src/utils/balances';
import { TripProgressRow } from '../src/components/trip/TripProgressRow';

export default function BudgetScreen() {
  const theme = useTheme();
  const [donateOpen, setDonateOpen] = React.useState(false);
  const [giftOpen, setGiftOpen] = React.useState(false);
  const [amount, setAmount] = React.useState('25');
  const { trips, selectedTripId } = useTrips();
  const trip = trips.find(t => t.id === selectedTripId) || trips[0];
  const total = trip?.itinerary.reduce((s, i) => s + i.total, 0) || 0;
  const perPerson = trip && trip.members.length > 0 ? total / trip.members.length : 0;
  const categories = categoryTotals(trip?.itinerary || []);
  const contributionsTotal = (trip?.contributions || []).reduce((s, c) => s + c.amount, 0);
  const balance = total - contributionsTotal;
  const annualGoal = trips.reduce((s, t) => s + (t.goalBudget || 0), 0);
  const annualPlanned = trips.reduce((s, t) => s + t.itinerary.reduce((ss, i) => ss + i.total, 0), 0);
  const annualContrib = trips.reduce((s, t) => s + (t.contributions || []).reduce((ss, c) => ss + c.amount, 0), 0);
  return (
    <Screen scroll>
      <Text style={[styles.title, { color: theme.colors.textPrimary }]}>Budget</Text>
      <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>Overview for: {trip?.name}</Text>
      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>All Trips Overview</Text>
        {trips.map(t => {
          const tTotal = t.itinerary.reduce((s, i) => s + i.total, 0);
          const tBal = computeBalances(t).reduce((s, b) => s + b.balance, 0);
          const pct = tTotal > 0 ? Math.min(1, Math.max(0, (t.contributions || []).reduce((s,c)=>s+c.amount,0) / tTotal)) : 0;
          return (
            <View key={t.id} style={{ borderTopColor: theme.colors.border, borderTopWidth: 1, paddingTop: 8, marginTop: 8 }}>
              <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{t.name}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>{formatCurrency(tTotal)} total • Progress {Math.round(pct*100)}%</Text>
              <View style={{ height: 8 }} />
              <ProgressBar progress={pct} />
            </View>
          );
        })}
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Annual Goals</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Goal: {formatCurrency(annualGoal)} • Planned: {formatCurrency(annualPlanned)} • Contributions: {formatCurrency(annualContrib)}</Text>
        <View style={{ height: 8 }} />
        <ProgressBar progress={annualGoal > 0 ? Math.min(1, annualPlanned / annualGoal) : 0} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Per Trip Goal Progress</Text>
        {trips.map(t => (
          <TripProgressRow key={t.id} trip={t} />
        ))}
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Overall</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>{formatCurrency(total)} total • {formatCurrency(perPerson)} per person</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Contributions: {formatCurrency(contributionsTotal)} • Balance: {formatCurrency(balance)}</Text>
        <View style={{ height: 12 }} />
        <ProgressBar progress={0.33} />
        <View style={{ height: 12 }} />
        <Button label={`Donate $${amount}`} onPress={() => setDonateOpen(true)} />
      </Card>
      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Breakdown</Text>
        <View style={{ height: 12 }} />
        <View style={{ alignItems: 'center' }}>
          <PieChart size={180} thickness={20} segments={(trip?.itinerary || []).map(i => ({ value: i.total, color: categoryColor(i.category) }))} />
        </View>
        <View style={{ height: 12 }} />
        <CategoryLegend items={categories} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Gift a Friend</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Send a contribution to someone’s item. Requires their confirmation.</Text>
        <Button label="Gift $50" onPress={() => setGiftOpen(true)} style={{ marginTop: 12 }} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Recent Contributions</Text>
        {(trip?.contributions || []).length === 0 ? (
          <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>No contributions yet.</Text>
        ) : (
          (trip?.contributions || []).map(c => (
            <View key={c.id} style={{ borderTopColor: theme.colors.border, borderTopWidth: 1, paddingTop: 8, marginTop: 8 }}>
              <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{c.label}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>{formatCurrency(c.amount)} • {c.date || '—'}</Text>
            </View>
          ))
        )}
      </Card>
      <Modal visible={donateOpen} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Donate to Overall</Text>
            <TextField label="Amount ($)" keyboardType="numeric" value={amount} onChangeText={setAmount} />
            <Button label="Confirm" onPress={() => setDonateOpen(false)} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setDonateOpen(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={giftOpen} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Gift a Friend</Text>
            <TextField label="Amount ($)" keyboardType="numeric" value={amount} onChangeText={setAmount} />
            <Button label="Send Gift" onPress={() => setGiftOpen(false)} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setGiftOpen(false)} />
          </View>
        </View>
      </Modal>
    </Screen>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 22, fontWeight: '700' },
  sectionTitle: { fontSize: 16, fontWeight: '600' },
});


