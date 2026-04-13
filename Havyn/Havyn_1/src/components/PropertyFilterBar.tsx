import React, { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  ScrollView,
  Modal,
  FlatList
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ChicagoArea } from '../models/Listing';

interface PropertyFilterBarProps {
  onFilterChange: (filters: PropertyFilters) => void;
  areas: ChicagoArea[];
}

export interface PropertyFilters {
  minPrice?: number;
  maxPrice?: number;
  area?: ChicagoArea;
  minRooms?: number;
}

const priceRanges = [
  { label: 'Any', min: undefined, max: undefined },
  { label: 'Under $1,500', min: undefined, max: 1500 },
  { label: '$1,500 - $2,000', min: 1500, max: 2000 },
  { label: '$2,000 - $3,000', min: 2000, max: 3000 },
  { label: '$3,000+', min: 3000, max: undefined },
];

const roomOptions = [
  { label: 'Any', value: undefined },
  { label: '1+', value: 1 },
  { label: '2+', value: 2 },
  { label: '3+', value: 3 },
  { label: '4+', value: 4 },
];

const PropertyFilterBar: React.FC<PropertyFilterBarProps> = ({ onFilterChange, areas }) => {
  const [filters, setFilters] = useState<PropertyFilters>({});
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [showPriceModal, setShowPriceModal] = useState(false);
  const [showAreaModal, setShowAreaModal] = useState(false);
  const [showRoomsModal, setShowRoomsModal] = useState(false);

  // Apply price filter
  const handlePriceSelect = (min?: number, max?: number) => {
    const newFilters = { ...filters, minPrice: min, maxPrice: max };
    setFilters(newFilters);
    onFilterChange(newFilters);
    setShowPriceModal(false);
  };

  // Apply area filter
  const handleAreaSelect = (area?: ChicagoArea) => {
    const newFilters = { ...filters, area };
    setFilters(newFilters);
    onFilterChange(newFilters);
    setShowAreaModal(false);
  };

  // Apply rooms filter
  const handleRoomsSelect = (minRooms?: number) => {
    const newFilters = { ...filters, minRooms };
    setFilters(newFilters);
    onFilterChange(newFilters);
    setShowRoomsModal(false);
  };

  // Reset all filters
  const resetFilters = () => {
    const newFilters = {};
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  // Format price range for display
  const getPriceRangeText = () => {
    const { minPrice, maxPrice } = filters;
    if (!minPrice && !maxPrice) return 'Any Price';
    if (minPrice && !maxPrice) return `$${minPrice}+`;
    if (!minPrice && maxPrice) return `Under $${maxPrice}`;
    return `$${minPrice} - $${maxPrice}`;
  };

  // Format rooms for display
  const getRoomsText = () => {
    const { minRooms } = filters;
    if (!minRooms) return 'Any Rooms';
    return `${minRooms}+ Rooms`;
  };

  // Get the number of active filters
  const getActiveFilterCount = () => {
    let count = 0;
    if (filters.minPrice !== undefined || filters.maxPrice !== undefined) count++;
    if (filters.area !== undefined) count++;
    if (filters.minRooms !== undefined) count++;
    return count;
  };

  const activeFilterCount = getActiveFilterCount();

  return (
    <View style={styles.container}>
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <TouchableOpacity 
          style={[
            styles.filterButton, 
            (filters.minPrice !== undefined || filters.maxPrice !== undefined) && styles.activeFilterButton
          ]}
          onPress={() => setShowPriceModal(true)}
        >
          <Ionicons 
            name="cash-outline" 
            size={18} 
            color={(filters.minPrice !== undefined || filters.maxPrice !== undefined) ? '#fff' : '#555'} 
          />
          <Text 
            style={[
              styles.filterButtonText, 
              (filters.minPrice !== undefined || filters.maxPrice !== undefined) && styles.activeFilterButtonText
            ]}
          >
            {getPriceRangeText()}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[
            styles.filterButton, 
            filters.area !== undefined && styles.activeFilterButton
          ]}
          onPress={() => setShowAreaModal(true)}
        >
          <Ionicons 
            name="location-outline" 
            size={18} 
            color={filters.area !== undefined ? '#fff' : '#555'} 
          />
          <Text 
            style={[
              styles.filterButtonText, 
              filters.area !== undefined && styles.activeFilterButtonText
            ]}
          >
            {filters.area || 'Any Area'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[
            styles.filterButton, 
            filters.minRooms !== undefined && styles.activeFilterButton
          ]}
          onPress={() => setShowRoomsModal(true)}
        >
          <Ionicons 
            name="bed-outline" 
            size={18} 
            color={filters.minRooms !== undefined ? '#fff' : '#555'} 
          />
          <Text 
            style={[
              styles.filterButtonText, 
              filters.minRooms !== undefined && styles.activeFilterButtonText
            ]}
          >
            {getRoomsText()}
          </Text>
        </TouchableOpacity>

        {activeFilterCount > 0 && (
          <TouchableOpacity 
            style={styles.resetButton}
            onPress={resetFilters}
          >
            <Ionicons name="close-circle" size={18} color="#e74c3c" />
            <Text style={styles.resetButtonText}>Reset ({activeFilterCount})</Text>
          </TouchableOpacity>
        )}
      </ScrollView>

      {/* Price Range Modal */}
      <Modal
        visible={showPriceModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowPriceModal(false)}
      >
        <TouchableOpacity 
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setShowPriceModal(false)}
        >
          <View style={styles.modalContent} onStartShouldSetResponder={() => true}>
            <Text style={styles.modalTitle}>Price Range</Text>
            {priceRanges.map((range, index) => (
              <TouchableOpacity
                key={index}
                style={styles.modalItem}
                onPress={() => handlePriceSelect(range.min, range.max)}
              >
                <Text style={styles.modalItemText}>{range.label}</Text>
                {(filters.minPrice === range.min && filters.maxPrice === range.max) && (
                  <Ionicons name="checkmark" size={20} color="#3498db" />
                )}
              </TouchableOpacity>
            ))}
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Area Modal */}
      <Modal
        visible={showAreaModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowAreaModal(false)}
      >
        <TouchableOpacity 
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setShowAreaModal(false)}
        >
          <View style={styles.modalContent} onStartShouldSetResponder={() => true}>
            <Text style={styles.modalTitle}>Neighborhood</Text>
            <TouchableOpacity
              style={styles.modalItem}
              onPress={() => handleAreaSelect(undefined)}
            >
              <Text style={styles.modalItemText}>Any Area</Text>
              {filters.area === undefined && (
                <Ionicons name="checkmark" size={20} color="#3498db" />
              )}
            </TouchableOpacity>
            <ScrollView style={styles.areaScrollView}>
              {areas.map((area, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.modalItem}
                  onPress={() => handleAreaSelect(area)}
                >
                  <Text style={styles.modalItemText}>{area}</Text>
                  {filters.area === area && (
                    <Ionicons name="checkmark" size={20} color="#3498db" />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Rooms Modal */}
      <Modal
        visible={showRoomsModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowRoomsModal(false)}
      >
        <TouchableOpacity 
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setShowRoomsModal(false)}
        >
          <View style={styles.modalContent} onStartShouldSetResponder={() => true}>
            <Text style={styles.modalTitle}>Bedrooms</Text>
            {roomOptions.map((option, index) => (
              <TouchableOpacity
                key={index}
                style={styles.modalItem}
                onPress={() => handleRoomsSelect(option.value)}
              >
                <Text style={styles.modalItemText}>{option.label}</Text>
                {filters.minRooms === option.value && (
                  <Ionicons name="checkmark" size={20} color="#3498db" />
                )}
              </TouchableOpacity>
            ))}
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  scrollContent: {
    paddingHorizontal: 16,
  },
  filterButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    marginRight: 12,
  },
  activeFilterButton: {
    backgroundColor: '#3498db',
  },
  filterButtonText: {
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 6,
    color: '#555',
  },
  activeFilterButtonText: {
    color: 'white',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffeeee',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  resetButtonText: {
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 6,
    color: '#e74c3c',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    width: '80%',
    maxHeight: '70%',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
    textAlign: 'center',
  },
  modalItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalItemText: {
    fontSize: 16,
    color: '#333',
  },
  areaScrollView: {
    maxHeight: 300,
  },
});

export default PropertyFilterBar; 