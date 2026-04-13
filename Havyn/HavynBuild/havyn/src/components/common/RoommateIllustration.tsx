import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Path, Circle, Ellipse } from 'react-native-svg';
import { colors } from '../../styles/theme';

interface RoommateIllustrationProps {
  width?: number;
  height?: number;
}

const RoommateIllustration: React.FC<RoommateIllustrationProps> = ({
  width = 200,
  height = 150,
}) => {
  return (
    <View style={styles.container}>
      <Svg width={width} height={height} viewBox="0 0 200 150">
        {/* Background Buildings (faint) */}
        <Path
          d="M20 120 L20 80 L40 80 L40 120 Z"
          fill="rgba(183, 201, 229, 0.15)"
        />
        <Path
          d="M50 120 L50 60 L70 60 L70 120 Z"
          fill="rgba(183, 201, 229, 0.2)"
        />
        <Path
          d="M130 120 L130 70 L150 70 L150 120 Z"
          fill="rgba(183, 201, 229, 0.15)"
        />
        <Path
          d="M160 120 L160 85 L180 85 L180 120 Z"
          fill="rgba(183, 201, 229, 0.18)"
        />
        
        {/* Person 1 (left - man in navy) */}
        {/* Head */}
        <Circle cx="70" cy="45" r="12" fill="#F4C2A1" />
        {/* Hair */}
        <Path
          d="M58 40 Q70 30 82 40 Q82 35 70 35 Q58 35 58 40"
          fill="#8B4513"
        />
        {/* Body */}
        <Path
          d="M60 60 L80 60 L78 100 L62 100 Z"
          fill={colors.primary}
        />
        {/* Arms */}
        <Path
          d="M60 70 L45 75 L48 85 L62 80"
          fill="#F4C2A1"
        />
        <Path
          d="M80 70 L95 75 L92 85 L78 80"
          fill="#F4C2A1"
        />
        {/* Legs */}
        <Path
          d="M62 100 L68 130 L72 130 L72 100"
          fill="#2C3E50"
        />
        <Path
          d="M68 100 L74 130 L78 130 L78 100"
          fill="#2C3E50"
        />
        
        {/* Person 2 (right - woman in light blue) */}
        {/* Head */}
        <Circle cx="130" cy="45" r="12" fill="#F4C2A1" />
        {/* Hair */}
        <Path
          d="M118 40 Q130 32 142 40 Q142 50 130 50 Q118 50 118 40"
          fill="#654321"
        />
        {/* Body */}
        <Path
          d="M120 60 L140 60 L138 100 L122 100 Z"
          fill="#87CEEB"
        />
        {/* Arms */}
        <Path
          d="M120 70 L105 75 L108 85 L122 80"
          fill="#F4C2A1"
        />
        <Path
          d="M140 70 L155 75 L152 85 L138 80"
          fill="#F4C2A1"
        />
        {/* Legs */}
        <Path
          d="M122 100 L128 130 L132 130 L132 100"
          fill="#2C3E50"
        />
        <Path
          d="M128 100 L134 130 L138 130 L138 100"
          fill="#2C3E50"
        />
        
        {/* Handshake area */}
        <Ellipse
          cx="100"
          cy="77"
          rx="8"
          ry="6"
          fill="#F4C2A1"
          opacity="0.9"
        />
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default RoommateIllustration; 