import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { FilterChip } from './ui';
import { User } from '../models/User';

interface FilterBarProps {
  onFilterChange?: (filters: FilterState, filteredUsers: User[]) => void;
  allUsers: User[];
}

interface FilterState {
  gender: string;
  budget: string;
  pets: boolean;
  cleanliness: boolean;
  moveInDate: boolean;
  location: string;
}

const FilterBar: React.FC<FilterBarProps> = ({ onFilterChange, allUsers }) => {
  const [filters, setFilters] = useState<FilterState>({
    gender: '',
    budget: '',
    pets: false,
    cleanliness: false,
    moveInDate: false,
    location: '',
  });

  // Handle gender filter selection
  const handleGenderFilter = (gender: string) => {
    const newGender = filters.gender === gender ? '' : gender;
    const updatedFilters = {
      ...filters,
      gender: newGender
    };
    applyFilters(updatedFilters);
  };

  // Handle budget filter selection
  const handleBudgetFilter = (budget: string) => {
    const newBudget = filters.budget === budget ? '' : budget;
    const updatedFilters = {
      ...filters,
      budget: newBudget
    };
    applyFilters(updatedFilters);
  };

  // Toggle boolean filters
  const toggleFilter = (filterName: keyof FilterState) => {
    if (typeof filters[filterName] === 'boolean') {
      const updatedFilters = {
        ...filters,
        [filterName]: !filters[filterName]
      };
      applyFilters(updatedFilters);
    }
  };

  // Apply filters to users and call the onChange callback
  const applyFilters = (updatedFilters: FilterState) => {
    setFilters(updatedFilters);
    
    // Filter users based on selected criteria
    let filteredUsers = [...allUsers];
    
    // Apply gender filter
    if (updatedFilters.gender) {
      filteredUsers = filteredUsers.filter(user => 
        user.gender === updatedFilters.gender ||
        (user.preferences && user.preferences.preferredRoommateGender === 'any')
      );
    }
    
    // Apply budget filter
    if (updatedFilters.budget) {
      const budgetRanges = {
        'low': { min: 0, max: 1000 },
        'medium': { min: 1000, max: 2000 },
        'high': { min: 2000, max: Infinity }
      };
      
      const range = budgetRanges[updatedFilters.budget as keyof typeof budgetRanges];
      if (range) {
        filteredUsers = filteredUsers.filter(user => 
          user.preferences && 
          user.preferences.monthlyRentBudget !== undefined &&
          user.preferences.monthlyRentBudget >= range.min && 
          user.preferences.monthlyRentBudget <= range.max
        );
      }
    }
    
    // Apply pets filter
    if (updatedFilters.pets) {
      filteredUsers = filteredUsers.filter(user => 
        // Pet friendly users are those with low noise levels and high cleanliness
        user.preferences && 
        user.preferences.noiseLevel !== undefined &&
        user.preferences.cleanliness !== undefined &&
        user.preferences.noiseLevel <= 3 && 
        user.preferences.cleanliness >= 3
      );
    }
    
    // Apply cleanliness filter
    if (updatedFilters.cleanliness) {
      filteredUsers = filteredUsers.filter(user => 
        user.preferences && 
        user.preferences.cleanliness !== undefined &&
        user.preferences.cleanliness >= 4
      );
    }
    
    // Notify parent component about filter changes
    onFilterChange?.(updatedFilters, filteredUsers);
  };

  return (
    <View style={styles.container}>
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.filtersContainer}
      >
        {/* Gender filters */}
        <FilterChip 
          label="Any Gender" 
          icon="people-outline"
          isSelected={filters.gender === ''}
          onPress={() => handleGenderFilter('')}
        />
        <FilterChip 
          label="Male" 
          icon="male-outline"
          isSelected={filters.gender === 'male'}
          onPress={() => handleGenderFilter('male')}
        />
        <FilterChip 
          label="Female" 
          icon="female-outline"
          isSelected={filters.gender === 'female'}
          onPress={() => handleGenderFilter('female')}
        />
        
        {/* Budget filters */}
        <FilterChip 
          label="Any Budget" 
          icon="cash-outline"
          isSelected={filters.budget === ''}
          onPress={() => handleBudgetFilter('')}
        />
        <FilterChip 
          label="< $1000" 
          icon="cash-outline"
          isSelected={filters.budget === 'low'}
          onPress={() => handleBudgetFilter('low')}
        />
        <FilterChip 
          label="$1000-$2000" 
          icon="cash-outline"
          isSelected={filters.budget === 'medium'}
          onPress={() => handleBudgetFilter('medium')}
        />
        <FilterChip 
          label="> $2000" 
          icon="cash-outline"
          isSelected={filters.budget === 'high'}
          onPress={() => handleBudgetFilter('high')}
        />
        
        {/* Other filters */}
        <FilterChip 
          label="Pet Friendly" 
          icon="paw-outline"
          isSelected={filters.pets}
          onPress={() => toggleFilter('pets')}
        />
        <FilterChip 
          label="Very Clean" 
          icon="sparkles-outline"
          isSelected={filters.cleanliness}
          onPress={() => toggleFilter('cleanliness')}
        />
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    paddingTop: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  filtersContainer: {
    paddingHorizontal: 16,
    paddingVertical: 4,
    flexDirection: 'row',
    flexWrap: 'nowrap',
  },
});

export default FilterBar; 