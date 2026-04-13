/**
 * File: app/trips/create.tsx
 * Purpose: Multi-step wizard for creating a new trip (details, dates, budget, members) with keyboard-safe fields and navigation to the created trip.
 */
import React from 'react';
import { View, Text, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Screen } from '../../src/components/Screen';
import { useTheme } from '../../src/theme/ThemeProvider';
import { Segmented } from '../../src/components/Segmented';
import { TextField } from '../../src/components/TextField';
import { Button } from '../../src/components/Button';
import { useRouter } from 'expo-router';
import { useTrips } from '../../src/state/TripsStore';
import { Header } from '../../src/components/Header';

export default function CreateTrip() {
  const theme = useTheme();
  const router = useRouter();
  const [step, setStep] = React.useState<'Details' | 'Dates' | 'Budget' | 'Members' | 'Review'>('Details');
  const [name, setName] = React.useState('');
  const [destination, setDestination] = React.useState('');
  const [members, setMembers] = React.useState<string>('You, Nate, Ava');
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');
  const [goalBudget, setGoalBudget] = React.useState('');
  const { createTrip } = useTrips();

  return (
    <Screen>
      <KeyboardAvoidingView behavior={Platform.select({ ios: 'padding', android: undefined })} style={{ flex: 1 }}>
        <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={{ paddingBottom: 32 }}>
          <Header title="Create a Trip" left={<Header.Back onPress={() => router.back()} />} />
          <View style={{ height: 12 }} />
          <Segmented options={[ {label:'Details',value:'Details'}, {label:'Dates',value:'Dates'}, {label:'Budget',value:'Budget'}, {label:'Members',value:'Members'}, {label:'Review',value:'Review'} ]} value={step} onChange={(v) => setStep(v as any)} />
          <View style={{ height: 16 }} />
          {step === 'Details' && (
            <View>
              <TextField label="Trip name" placeholder="Chicago City Break" value={name} onChangeText={setName} />
              <TextField label="Destination(s)" placeholder="Chicago, IL" value={destination} onChangeText={setDestination} />
              <Button label="Next" onPress={() => setStep('Dates')} />
            </View>
          )}
          {step === 'Dates' && (
            <View>
              <TextField label="Start Date (YYYY-MM-DD)" placeholder="2025-09-20" value={startDate} onChangeText={setStartDate} />
              <TextField label="End Date (YYYY-MM-DD)" placeholder="2025-09-25" value={endDate} onChangeText={setEndDate} />
              <Button label="Next" onPress={() => setStep('Budget')} />
            </View>
          )}
          {step === 'Budget' && (
            <View>
              <TextField label="Goal Budget ($)" placeholder="4000" keyboardType="numeric" value={goalBudget} onChangeText={setGoalBudget} />
              <Button label="Next" onPress={() => setStep('Members')} />
            </View>
          )}
          {step === 'Members' && (
            <View>
              <TextField label="Members" placeholder="Comma separated" value={members} onChangeText={setMembers} />
              <Button label="Next" onPress={() => setStep('Review')} />
            </View>
          )}
          {step === 'Review' && (
            <View>
              <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>Review</Text>
              <Text style={{ color: theme.colors.textSecondary, marginTop: 6 }}>Name: {name || '—'}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>Destination: {destination || '—'}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>Start: {startDate || '—'} • End: {endDate || '—'}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>Goal Budget: {goalBudget ? `$${goalBudget}` : '—'}</Text>
              <Text style={{ color: theme.colors.textSecondary }}>Members: {members || '—'}</Text>
              <View style={{ height: 12 }} />
              <Button label="Create" onPress={() => {
                const memberList = (members || '')
                  .split(',')
                  .map(m => m.trim())
                  .filter(Boolean)
                  .map((m, idx) => ({ id: Math.random().toString(36).slice(2), name: m, avatarColor: ['#7C5CFF','#22C55E','#F59E0B','#0EA5E9'][idx % 4] }));
                const range = startDate && endDate ? `${new Date(startDate).toLocaleDateString()} - ${new Date(endDate).toLocaleDateString()}` : 'TBD';
                const newTrip = createTrip({ name: name || 'New Trip', destination: destination || 'TBD', dateRange: range, startDate: startDate || undefined, endDate: endDate || undefined, goalBudget: goalBudget ? Number(goalBudget) : undefined, members: memberList });
                router.replace({ pathname: '/trip/[id]', params: { id: newTrip.id } });
              }} />
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  );
}


