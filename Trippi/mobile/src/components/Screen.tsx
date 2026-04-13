/**
 * File: src/components/Screen.tsx
 * Purpose: Safe-area aware screen wrapper applying consistent padding and background color. Supports optional scrolling.
 */
import React from 'react';
import { View, StyleSheet, ScrollView, ScrollViewProps } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeProvider';

type Props = React.PropsWithChildren<{ scroll?: boolean; style?: any; contentContainerStyle?: any }> & Partial<ScrollViewProps>;

export function Screen({ children, style, contentContainerStyle, scroll }: Props) {
  const theme = useTheme();
  return (
    <SafeAreaView style={[styles.root, { backgroundColor: theme.colors.background }, style]} edges={["top","left","right","bottom"]}>
      {scroll ? (
        <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={[styles.inner, { paddingHorizontal: theme.spacing(2), paddingTop: theme.spacing(2) }, contentContainerStyle]}>
          {children}
        </ScrollView>
      ) : (
        <View style={[styles.inner, { paddingHorizontal: theme.spacing(2), paddingTop: theme.spacing(2) }, contentContainerStyle]}>
          {children}
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  inner: { flexGrow: 1 },
});


