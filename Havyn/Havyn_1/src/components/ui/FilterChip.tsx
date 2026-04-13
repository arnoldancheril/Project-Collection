import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface FilterChipProps {
  label: string;
  icon?: string;
  isSelected?: boolean;
  onPress: () => void;
}

const FilterChip: React.FC<FilterChipProps> = ({ 
  label, 
  icon, 
  isSelected = false, 
  onPress 
}) => {
  return (
    <TouchableOpacity
      style={[
        styles.container,
        isSelected ? styles.selectedContainer : {}
      ]}
      onPress={onPress}
    >
      {icon && (
        <Ionicons 
          name={icon as any} 
          size={16} 
          color={isSelected ? 'white' : '#333'} 
          style={styles.icon} 
        />
      )}
      <Text style={[
        styles.label,
        isSelected ? styles.selectedLabel : {}
      ]}>
        {label}
      </Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  selectedContainer: {
    backgroundColor: '#3498db',
  },
  icon: {
    marginRight: 4,
  },
  label: {
    fontSize: 14,
    color: '#333',
  },
  selectedLabel: {
    color: 'white',
    fontWeight: '500',
  },
});

export default FilterChip; 