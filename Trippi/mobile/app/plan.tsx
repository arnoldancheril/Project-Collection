/**
 * File: app/plan.tsx
 * Purpose: Planning screen: AI Trip Planner entry (top), select active trip, manage itinerary items with category and totals, and invite friends via a proper modal.
 */
import React from 'react';
import { View, Text, StyleSheet, Modal } from 'react-native';
import { useTheme } from '../src/theme/ThemeProvider';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { Screen } from '../src/components/Screen';
import { TextField } from '../src/components/TextField';
import { useTrips } from '../src/state/TripsStore';
import { Segmented } from '../src/components/Segmented';
import { categoryColor } from '../src/utils/budget';
import { computeBalances } from '../src/utils/balances';
import { TripSwitcher } from '../src/components/TripSwitcher';
import { useRouter } from 'expo-router';

export default function PlanScreen() {
  const theme = useTheme();
  const router = useRouter();
  const [showAdd, setShowAdd] = React.useState(false);
  const [showInvite, setShowInvite] = React.useState(false);
  const [label, setLabel] = React.useState('');
  const [total, setTotal] = React.useState('');
  const [category, setCategory] = React.useState<'Lodging' | 'Flights' | 'Transport' | 'Activities' | 'Food' | 'Other'>('Other');
  const [invitee, setInvitee] = React.useState('');
  const { trips, selectedTripId, selectTrip, addItineraryItem } = useTrips();
  const selectedTrip = trips.find(t => t.id === selectedTripId) || trips[0];
  const balances = selectedTrip ? computeBalances(selectedTrip) : [];
  const [switcherOpen, setSwitcherOpen] = React.useState(false);
  return (
    <Screen scroll>
      <Text style={[styles.title, { color: theme.colors.textPrimary }]}>Plan</Text>
      <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>AI-assisted planning, itinerary building, and invites.</Text>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>AI Trip Planner</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Ask about itinerary, splits, tips, and planning details.</Text>
        <Button label="Try AI Trip Planner" onPress={() => router.push('/ai')} style={{ marginTop: 12 }} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Select Trip</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>{selectedTrip?.name} • {selectedTrip?.destination}</Text>
        <Button label="Switch Trip" onPress={() => setSwitcherOpen(true)} style={{ marginTop: 12 }} />
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Itinerary</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Add flights, stays, and activities. Set split preferences per item.</Text>
        <Button label="Add Item" onPress={() => setShowAdd(true)} style={{ marginTop: 12 }} />
        {selectedTrip?.itinerary.map(item => (
          <View key={item.id} style={{ marginTop: 12, borderTopWidth: 1, borderTopColor: theme.colors.border, paddingTop: 8 }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{item.label}</Text>
            <Text style={{ color: theme.colors.textSecondary }}>{item.category} • ${item.total.toLocaleString()}</Text>
            <View style={{ width: 48, height: 4, backgroundColor: categoryColor(item.category), borderRadius: 2, marginTop: 6 }} />
          </View>
        ))}
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Split Costs</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Track who paid, who owes, and settlements.</Text>
        {balances.length === 0 ? (
          <Text style={{ color: theme.colors.textSecondary, marginTop: 8 }}>No balances yet. Add expenses from Trip Itinerary.</Text>
        ) : (
          balances.map(b => (
            <View key={b.memberId} style={{ borderTopColor: theme.colors.border, borderTopWidth: 1, paddingTop: 8, marginTop: 8 }}>
              <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{b.name}</Text>
              <Text style={{ color: b.balance >= 0 ? '#22C55E' : '#EF4444' }}>{b.balance >= 0 ? `Should receive $${b.balance.toFixed(2)}` : `Owes $${Math.abs(b.balance).toFixed(2)}`}</Text>
            </View>
          ))
        )}
      </Card>
      

      <Card style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Invite Friends</Text>
        <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>
          Share a link or invite by username. Assign owner permissions as needed.
        </Text>
        <Button label="Invite" onPress={() => setShowInvite(true)} style={{ marginTop: 12 }} />
      </Card>
      <Modal visible={showAdd} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Add Itinerary Item</Text>
            <TextField label="Label" placeholder="Airbnb" value={label} onChangeText={setLabel} />
            <TextField label="Total ($)" placeholder="500" keyboardType="numeric" value={total} onChangeText={setTotal} />
            <Text style={{ color: theme.colors.textSecondary, marginBottom: 6 }}>Category</Text>
            <Segmented options={[ 'Lodging','Flights','Transport','Activities','Food','Other' ]} value={category} onChange={(v) => setCategory(v as any)} />
            <View style={{ height: 12 }} />
            <Button label="Save" onPress={() => { if (selectedTrip) { addItineraryItem(selectedTrip.id, { id: Math.random().toString(36).slice(2), label, total: Number(total || 0), category }); } setShowAdd(false); setLabel(''); setTotal(''); }} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setShowAdd(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={showInvite} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Invite Friends</Text>
            <TextField label="Username or email" placeholder="friend@example.com" value={invitee} onChangeText={setInvitee} />
            <Button label="Send Invite" onPress={() => { setShowInvite(false); setInvitee(''); }} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setShowInvite(false)} />
          </View>
        </View>
      </Modal>

      <TripSwitcher visible={switcherOpen} onClose={() => setSwitcherOpen(false)} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 22, fontWeight: '700' },
  sectionTitle: { fontSize: 16, fontWeight: '600' },
});


