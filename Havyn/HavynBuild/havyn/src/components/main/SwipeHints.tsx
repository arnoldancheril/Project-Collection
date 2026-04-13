import React from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, fontSizes } from '../../styles/theme';

interface SwipeHintsProps {
  leftOpacity: Animated.Value;
  rightOpacity: Animated.Value;
}

const SwipeHints = ({ leftOpacity, rightOpacity }: SwipeHintsProps) => {
  return (
    <View style={styles.container}>
      {/* Left skip hint */}
      <Animated.View 
        style={[
          styles.hintContainer, 
          styles.leftHint,
          { opacity: leftOpacity }
        ]}
      >
        <Ionicons 
          name="chevron-back" 
          size={24} 
          color={colors.text.secondary} 
          style={styles.icon}
        />
        <Text style={styles.hintText}>Skip</Text>
      </Animated.View>

      {/* Right connect hint */}
      <Animated.View 
        style={[
          styles.hintContainer, 
          styles.rightHint,
          { opacity: rightOpacity }
        ]}
      >
        <Text style={styles.hintText}>Connect</Text>
        <Ionicons 
          name="chevron-forward" 
          size={24} 
          color={colors.text.secondary} 
          style={styles.icon}
        />
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: '50%',
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xl,
    zIndex: 10,
    pointerEvents: 'none',
  },
  hintContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    opacity: 0.25, // Default low opacity as specified
  },
  leftHint: {
    alignSelf: 'flex-start',
  },
  rightHint: {
    alignSelf: 'flex-end',
  },
  hintText: {
    fontSize: fontSizes.md,
    color: colors.text.secondary,
    fontWeight: '500',
  },
  icon: {
    marginHorizontal: spacing.xs,
  },
});

export default SwipeHints; 