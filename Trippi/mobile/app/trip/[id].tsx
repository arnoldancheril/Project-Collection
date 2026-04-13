/**
 * File: app/trip/[id].tsx
 * Purpose: Trip detail screen with tabs: Overview (rich details), Itinerary, Members, Expenses. Uses centralized TripsStore.
 */
import React from 'react';
import { View, Text, StyleSheet, Modal, Pressable } from 'react-native';
import { Screen } from '../../src/components/Screen';
import { useTheme } from '../../src/theme/ThemeProvider';
import { useTrips } from '../../src/state/TripsStore';
import { useLocalSearchParams } from 'expo-router';
import { Avatar } from '../../src/components/Avatar';
import { Card } from '../../src/components/Card';
import { Segmented } from '../../src/components/Segmented';
import { Button } from '../../src/components/Button';
import { TextField } from '../../src/components/TextField';
import { useTrips as useTripsStore } from '../../src/state/TripsStore';
import { ItineraryTimeline } from '../../src/components/trip/ItineraryTimeline';
import { MemberRow } from '../../src/components/MemberRow';
import { Link, useRouter } from 'expo-router';
import { ExpenseForm } from '../../src/components/ExpenseForm';
import { TripOverviewCard } from '../../src/components/trip/TripOverviewCard';
import { ItineraryList } from '../../src/components/trip/ItineraryList';
import { MemberDetailsModal } from '../../src/components/trip/MemberDetailsModal';
import { ExpensesSection } from '../../src/components/trip/ExpensesSection';

export default function TripDetail() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const { trips, selectTrip } = useTrips();
  const trip = trips.find(t => t.id === id) || trips[0];
  React.useEffect(() => { if (trip) selectTrip(trip.id); }, [trip, selectTrip]);
  const total = trip.itinerary.reduce((s, i) => s + i.total, 0);

  const [tab, setTab] = React.useState<'Overview' | 'Itinerary' | 'Members' | 'Expenses'>('Overview');
  const [addMember, setAddMember] = React.useState(false);
  const [addItem, setAddItem] = React.useState(false);
  const [newMember, setNewMember] = React.useState('');
  const [newLabel, setNewLabel] = React.useState('');
  const [newTotal, setNewTotal] = React.useState('');
  const [newCategory, setNewCategory] = React.useState<'Lodging' | 'Flights' | 'Transport' | 'Activities' | 'Food' | 'Other'>('Other');
  const [newStartAt, setNewStartAt] = React.useState('');
  const [editingItemId, setEditingItemId] = React.useState<string | null>(null);
  const [expenseOpen, setExpenseOpen] = React.useState(false);
  const [expenseAmount, setExpenseAmount] = React.useState('');
  const [expensePaidBy, setExpensePaidBy] = React.useState<string>('');
  const [expenseSplitWith, setExpenseSplitWith] = React.useState<string>('');
  const { addExpense, editItineraryItem, addItineraryItem } = useTripsStore();
  const [memberOpen, setMemberOpen] = React.useState(false);
  const [viewedMember, setViewedMember] = React.useState<string | undefined>(undefined);

  return (
    <Screen scroll>
      <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
        <Pressable onPress={() => router.back()} style={({ pressed }) => ({ opacity: pressed ? 0.6 : 1, marginRight: 12 })}>
          <Text style={{ color: theme.colors.primary }}>{'< Back'}</Text>
        </Pressable>
      </View>
      <Text style={[styles.title, { color: theme.colors.textPrimary }]}>{trip.name}</Text>
      <Text style={{ color: theme.colors.textSecondary }}>{trip.destination} • {trip.dateRange}</Text>

      <View style={{ height: 12 }} />
      <Segmented
        options={[
          { label: 'Overview', value: 'Overview' },
          { label: 'Itinerary', value: 'Itinerary' },
          { label: 'Members', value: 'Members' },
          { label: 'Expenses', value: 'Expenses' },
        ]}
        value={tab}
        onChange={(v) => setTab(v as any)}
      />

      {tab === 'Overview' && (
        <TripOverviewCard trip={trip} />
      )}

      {tab === 'Itinerary' && (
        <>
          <ItineraryList
            items={trip.itinerary}
            onAddNew={() => setAddItem(true)}
            onEdit={(item) => { setEditingItemId(item.id); setNewLabel(item.label); setNewTotal(String(item.total)); setAddItem(true); }}
            onAddExpense={(item) => { setEditingItemId(item.id); setExpenseOpen(true); }}
          />
          <Card style={{ marginTop: 16 }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Timeline</Text>
            <ItineraryTimeline items={trip.itinerary} />
          </Card>
        </>
      )}

      {tab === 'Members' && (
        <Card style={{ marginTop: 16 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Members</Text>
            <Button label="Add" onPress={() => setAddMember(true)} style={{ width: 90 }} />
          </View>
          {trip.members.map(m => (
            <MemberRow key={m.id} id={m.id} name={m.name} avatarColor={m.avatarColor} onView={() => { setViewedMember(m.id); setMemberOpen(true); }} />
          ))}
        </Card>
      )}

      {tab === 'Expenses' && (
        <>
          <ExpensesSection trip={trip} onAddExpense={() => { setEditingItemId(null); setExpenseOpen(true); }} />
          <Card style={{ marginTop: 16 }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Needs Your Input</Text>
            {trip.itinerary.filter(i => i.startAt && Date.parse(i.startAt) < Date.now() && !(trip.expenses || []).some(e => e.itemId === i.id)).map(i => (
              <View key={i.id} style={{ borderTopWidth: 1, borderTopColor: theme.colors.border, paddingTop: 8, marginTop: 8 }}>
                <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }}>{i.label}</Text>
                <Text style={{ color: theme.colors.textSecondary }}>Event finished. Add actual expense?</Text>
                <Button label="Add Expense" onPress={() => { setEditingItemId(i.id); setExpenseOpen(true); }} style={{ marginTop: 8 }} />
              </View>
            ))}
          </Card>
        </>
      )}

      <Modal visible={addMember} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Add Member</Text>
            <TextField label="Name or username" value={newMember} onChangeText={setNewMember} />
            <Button label="Invite" onPress={() => setAddMember(false)} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setAddMember(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={addItem} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>{editingItemId ? 'Edit Itinerary Item' : 'Add Itinerary Item'}</Text>
            <TextField label="Label" value={newLabel} onChangeText={setNewLabel} />
            <TextField label="Budget Total ($)" keyboardType="numeric" value={newTotal} onChangeText={setNewTotal} />
            <Text style={{ color: theme.colors.textSecondary, marginBottom: 6 }}>Category</Text>
            <Segmented options={[ 'Lodging','Flights','Transport','Activities','Food','Other' ]} value={newCategory} onChange={(v) => setNewCategory(v as any)} />
            <View style={{ height: 12 }} />
            <TextField label="Start (YYYY-MM-DD HH:mm)" placeholder="2025-09-21 13:00" value={newStartAt} onChangeText={setNewStartAt} />
            <Button label={editingItemId ? 'Save' : 'Add'} onPress={() => {
              if (editingItemId) {
                editItineraryItem(trip.id, { id: editingItemId, label: newLabel, total: Number(newTotal || 0) });
              } else {
                const id = Math.random().toString(36).slice(2);
                const start = newStartAt ? new Date(newStartAt.replace(' ', 'T')).toISOString() : undefined;
                const item = { id, label: newLabel || 'New item', total: Number(newTotal || 0), category: newCategory, startAt: start } as any;
                addItineraryItem(trip.id, item);
              }
              setAddItem(false); setEditingItemId(null); setNewLabel(''); setNewTotal(''); setNewStartAt('');
            }} />
            <View style={{ height: 8 }} />
            <Button label="Cancel" variant="secondary" onPress={() => setAddItem(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={expenseOpen} animationType="slide" transparent>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' }}>
          <View style={{ backgroundColor: theme.colors.surface, padding: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20, borderColor: theme.colors.border, borderWidth: 1, maxHeight: 520 }}>
            <Text style={{ color: theme.colors.textPrimary, fontSize: 18, fontWeight: '700' }}>Add Expense</Text>
            <ExpenseForm
              amount={expenseAmount}
              onAmount={setExpenseAmount}
              paidBy={expensePaidBy}
              onPaidBy={setExpensePaidBy}
              splitWith={expenseSplitWith}
              onSplitWith={setExpenseSplitWith}
              onSubmit={() => {
                const exp = { id: Math.random().toString(36).slice(2), itemId: editingItemId || undefined, label: 'Expense', amount: Number(expenseAmount || 0), paidBy: expensePaidBy, splitWith: expenseSplitWith.split(',').map(s => s.trim()).filter(Boolean), createdAt: new Date().toISOString() };
                addExpense(trip.id, exp);
                setExpenseOpen(false);
                setExpenseAmount(''); setExpensePaidBy(''); setExpenseSplitWith('');
              }}
              onCancel={() => setExpenseOpen(false)}
            />
          </View>
        </View>
      </Modal>

      <MemberDetailsModal
        visible={memberOpen}
        onClose={() => setMemberOpen(false)}
        member={trip.members.find(m => m.id === viewedMember)}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: { fontSize: 22, fontWeight: '700' },
});


