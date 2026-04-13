import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import HavynLogo from './HavynLogo';
import { colors, spacing } from '../../styles/theme';

interface HomeHeaderProps {
  onFilterPress: () => void;
}

const HomeHeader = ({ onFilterPress }: HomeHeaderProps) => {
  return (
    <View style={styles.container}>
      {/* Left-aligned Havyn logo */}
      <View style={styles.logoContainer}>
        <HavynLogo size={24} showNameLogo={true} showPulse={false} />
      </View>
      
      {/* Right-aligned filter icon */}
      <TouchableOpacity 
        style={styles.filterButton}
        onPress={onFilterPress}
        accessibilityRole="button"
        accessibilityLabel="Open filters"
      >
        <Ionicons 
          name="options-outline" 
          size={24} 
          color={colors.primaryProfile} 
        />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    height: 56,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    backgroundColor: 'transparent',
    marginTop: spacing.md,
  },
  logoContainer: {
    flex: 1,
  },
  filterButton: {
    padding: spacing.sm,
    borderRadius: 8,
  },
});

export default HomeHeader; 