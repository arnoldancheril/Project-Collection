/**
 * File: app/(tabs)/index.tsx
 * Purpose: Home tab with personal greeting and savings overview. Shows top progress and prioritized
 *          savings list UI (static demo data for visualization only).
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Screen } from '../../src/components/Screen';
import { useTheme } from '../../src/theme/ThemeProvider';
import { useTrips } from '../../src/state/TripsStore';
import { ProgressBar } from '../../src/components/ProgressBar';
import { Modal, Pressable, Animated } from 'react-native';
import { SavingsPieChart } from '../../src/components/SavingsPieChart';

export default function HomeTab() {
  const theme = useTheme();
  const { trips } = useTrips();
  const userName = 'Traveler';
  const perTrip = [
    { id: 't1', name: 'Chicago City Break', saved: 300, goal: 500, color: '#7C5CFF' },
    { id: 't2', name: 'Ski Weekend', saved: 200, goal: 300, color: '#22C55E' },
    { id: 't3', name: 'Beach Getaway', saved: 100, goal: 200, color: '#F59E0B' },
  ];
  const totalSaved = perTrip.reduce((s, t) => s + t.saved, 0);
  const totalGoal = perTrip.reduce((s, t) => s + t.goal, 0);
  // Demo prioritized items (independent of total-saved chart). Edit values as needed.
  const prioritized = [
    { id: 'p1', label: 'Lodging Fund', trip: 'Chicago City Break', saved: 180, goal: 300 },
    { id: 'p2', label: 'Lift Tickets', trip: 'Ski Weekend', saved: 120, goal: 180 },
    { id: 'p3', label: 'Activities', trip: 'Beach Getaway', saved: 60, goal: 120 },
  ];
  const [tileWidth, setTileWidth] = React.useState<number>(0);

  return (
    <Screen scroll contentContainerStyle={{ paddingBottom: theme.spacing(3) }}>
      <Text style={[styles.greeting, { color: theme.colors.textPrimary }]}>Hi {userName}</Text>

      <View onLayout={(e) => setTileWidth(e.nativeEvent.layout.width)} style={{ marginTop: 12, backgroundColor: theme.colors.surface, borderRadius: theme.radius.lg, padding: 16, borderWidth: 1, borderColor: theme.colors.border, alignItems: 'center' }}>
        <Text style={{ color: theme.colors.textSecondary }}>Total Saved</Text>
        <Text style={{ color: theme.colors.textPrimary, fontWeight: '800', fontSize: 20, marginTop: 4 }}>
          ${totalSaved} / ${totalGoal}
        </Text>
        <View style={{ height: 12 }} />
        {/** Chart renders its own tooltip inside its container for correct positioning */}
        <ChartWithOverlay size={160} thickness={22} frameWidth={tileWidth} data={perTrip} />
      </View>

      <View style={{ marginTop: 16 }}>
        <Text style={[styles.sectionTitle, { color: theme.colors.textPrimary }]}>Prioritized Savings</Text>
        {prioritized.map(item => (
          <View key={item.id} style={{ borderTopColor: theme.colors.border, borderTopWidth: 1, paddingTop: 10, marginTop: 10 }}>
            <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>{item.label}
              <Text style={{ color: theme.colors.textSecondary }}> • {item.trip}</Text>
            </Text>
            <View style={{ height: 8 }} />
            <ProgressBar progress={Math.min(1, item.saved / item.goal)} />
            <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>${item.saved} / ${item.goal}</Text>
          </View>
        ))}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  greeting: { fontSize: 22, fontWeight: '800' },
  sectionTitle: { fontSize: 16, fontWeight: '700' },
});

function ChartWithOverlay({ size, thickness, data, frameWidth }: { size: number; thickness: number; data: { name: string; saved: number; goal: number; color: string }[]; frameWidth: number }) {
  const center = size / 2;
  const totalGoal = Math.max(1, data.reduce((s, t) => s + Math.max(0, t.goal), 0));
  const radius = (size - thickness) / 2;
  const inner = radius - thickness / 2 - 4;
  const outer = radius + thickness / 2 + 4;
  const [selectedIdx, setSelectedIdx] = React.useState<number | null>(null);
  const [anchor, setAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const fade = React.useRef(new Animated.Value(0)).current;

  const getSliceIndexFromPoint = (x: number, y: number) => {
    const dx = x - center;
    const dy = y - center;
    const r = Math.sqrt(dx * dx + dy * dy);
    if (r < inner || r > outer) return -1; // outside ring
    // atan2 uses +X as 0°, screen Y grows down which effectively makes clockwise positive.
    // Normalize to 0..360, then shift so 0 starts at top (90°) and proceeds clockwise.
    let angle = Math.atan2(dy, dx) * (180 / Math.PI); // -180..180, 0 at +X
    angle = (angle + 360) % 360; // 0..360, 0 at +X
    const frac = ((angle + 90) % 360) / 360; // 0 at top, clockwise increasing
    let acc = 0;
    for (let i = 0; i < data.length; i++) {
      const seg = Math.max(0, data[i].goal) / totalGoal;
      if (frac >= acc && frac < acc + seg) return i;
      acc += seg;
    }
    return data.length - 1;
  };

  const getAnchorForSlice = (idx: number) => {
    let acc = 0;
    for (let i = 0; i < idx; i++) acc += Math.max(0, data[i].goal) / totalGoal;
    const seg = Math.max(0, data[idx].goal) / totalGoal;
    const mid = acc + seg / 2; // fraction around circle
    const angleDeg = (mid * 360 + 270) % 360; // +270 because our 0 is at top (-90)
    const rad = (angleDeg * Math.PI) / 180;
    const r = radius + thickness + 60; // push further outward for clearer placement
    const x = center + Math.cos(rad) * r;
    const y = center + Math.sin(rad) * r;
    return { x, y };
  };

  return (
    <View style={{ width: frameWidth || size, height: size, position: 'relative', alignItems: 'center', justifyContent: 'center', overflow: 'visible' }}>
      <SavingsPieChart size={size} thickness={thickness} slices={data.map(t => ({ saved: t.saved, goal: t.goal, color: t.color }))} />
      <Pressable
        style={{ position: 'absolute', top: 0, left: (frameWidth ? (frameWidth - size) / 2 : 0), width: size, height: size }}
        onPress={(e) => {
          const { locationX, locationY } = (e.nativeEvent as any);
          const idx = getSliceIndexFromPoint(locationX, locationY);
          if (idx >= 0) {
            const pos = getAnchorForSlice(idx);
            setSelectedIdx(idx);
            setAnchor(pos);
            fade.setValue(0);
            Animated.timing(fade, { toValue: 1, duration: 120, useNativeDriver: true }).start();
          } else {
            setSelectedIdx(null);
          }
        }}
      />
      {selectedIdx !== null && anchor && (
        <InlinePopover
          opacity={fade}
          x={(frameWidth ? (frameWidth - size) / 2 : 0) + anchor.x}
          y={anchor.y}
          title={data[selectedIdx].name}
          saved={data[selectedIdx].saved}
          goal={data[selectedIdx].goal}
          containerSize={frameWidth || size}
        />
      )}
    </View>
  );
}

function InlinePopover({ opacity, x, y, title, saved, goal, containerSize }: { opacity: Animated.Value; x: number; y: number; title: string; saved: number; goal: number; containerSize: number }) {
  const theme = useTheme();
  const translate = opacity.interpolate({ inputRange: [0, 1], outputRange: [6, 0] });
  // Clamp within the chart container (assumes container ~ size x size)
  const width = 220; const height = 56; const pad = 8;
  // Position card to the left or right of the anchor based on which side of the circle was tapped
  const cx = containerSize / 2;
  const cy = containerSize / 2;
  const isLeftSide = x < cx;
  const gap = 12;
  const baseLeft = isLeftSide ? (x - width - gap) : (x + gap);
  const baseTop = y - height / 2; // center vertically on anchor
  // Blend between where it is now and the container center to pull it on-screen
  const centerLeft = containerSize / 2 - width / 2;
  const midLeft = (baseLeft + centerLeft) / 2;
  const left = Math.max(pad, Math.min(containerSize - width - pad, midLeft));
  // Keep vertical within the card bounds
  const top = Math.max(pad, Math.min(containerSize - height - pad, baseTop));
  return (
    <Animated.View style={{ position: 'absolute', left, top, opacity, transform: [{ translateY: translate }], backgroundColor: theme.colors.surface, padding: 10, borderRadius: 10, borderColor: theme.colors.border, borderWidth: 1, minWidth: width, shadowColor: '#000', shadowOpacity: 0.2, shadowRadius: 8, elevation: 4 }} pointerEvents="none">
      <Text style={{ color: theme.colors.textPrimary, fontWeight: '700' }}>{title}</Text>
      <Text style={{ color: theme.colors.textSecondary, marginTop: 4 }}>${saved} / ${goal}</Text>
    </Animated.View>
  );
}


