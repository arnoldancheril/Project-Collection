import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  Animated,
  Dimensions,
  Modal,
  Alert,
  ScrollView,
  Switch,
  TextInput,
  Platform
} from 'react-native';
import { Ionicons, MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import { useProfile } from '../../contexts/ProfileContext';
import { COLORS } from '../../utils/theme';
import SwipeCard from '../../components/SwipeCard';
import SAMPLE_PROFILES from '../../utils/sampleProfiles';
import Slider from '@react-native-community/slider';

const { width, height } = Dimensions.get('window');
const CARD_OFFSET = 12; // Offset for stacked cards
const VISIBLE_CARDS = 3; // Number of cards visible in the stack

// Chicago neighborhoods for the filter
const CHICAGO_NEIGHBORHOODS = [
  'Lincoln Park', 'Wicker Park', 'Lakeview', 'Logan Square', 
  'West Loop', 'South Loop', 'River North', 'Old Town',
  'Uptown', 'Andersonville', 'Edgewater', 'Pilsen'
];

const SwipeScreen = ({ navigation }) => {
  const { profile, likeProfile, dislikeProfile } = useProfile(); // Use the enhanced profile context
  const [profiles, setProfiles] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [showEmptyState, setShowEmptyState] = useState(false);
  const cardRefs = useRef(new Array());
  
  // Filter state
  const [filters, setFilters] = useState({
    budget: { min: 500, max: 3000 },
    locations: [],
    propertyType: [],
    gender: 'any',
    ageRange: { min: 18, max: 50 },
    lifestyle: {
      smoking: 'any',
      pets: 'any',
      drinking: 'any',
      cleanliness: 'any'
    },
    moveInDate: null
  });

  // Selected neighborhoods
  const [selectedNeighborhoods, setSelectedNeighborhoods] = useState([]);

  useEffect(() => {
    // Use sample profiles as our data source
    setProfiles(SAMPLE_PROFILES);
    setLoading(false);
  }, []);

  useEffect(() => {
    // Check if we need to show the empty state
    setShowEmptyState(currentIndex >= profiles.length);
  }, [currentIndex, profiles.length]);

  const handleSwipeRight = (index) => {
    // Get the profile at the specified index (default to current)
    const swipedIndex = index !== undefined ? index : currentIndex;
    if (swipedIndex >= profiles.length) return;
    
    const currentProfile = profiles[swipedIndex];
    
    // Use the likeProfile function from context and check if it's a match
    const { isMatch } = likeProfile(currentProfile);
    
    // Show match alert if it's a match
    if (isMatch) {
      setTimeout(() => {
        Alert.alert(
          "It's a Match!",
          `You and ${currentProfile.firstName} liked each other.`,
          [
            {
              text: "Keep Browsing",
              style: "cancel"
            },
            {
              text: "Send Message",
              onPress: () => navigation.navigate('Matches')
            }
          ]
        );
      }, 500);
    }
    
    // Move to next profile
    setCurrentIndex(prevIndex => prevIndex + 1);
  };

  const handleSwipeLeft = (index) => {
    // Get the profile at the specified index (default to current)
    const swipedIndex = index !== undefined ? index : currentIndex;
    if (swipedIndex >= profiles.length) return;
    
    const currentProfile = profiles[swipedIndex];
    
    // Use the dislikeProfile function from context
    dislikeProfile(currentProfile);
    
    // Move to next profile
    setCurrentIndex(prevIndex => prevIndex + 1);
  };

  const navigateToDetailedProfile = (index) => {
    const viewedIndex = index !== undefined ? index : currentIndex;
    if (viewedIndex >= profiles.length) return;
    
    const currentProfile = profiles[viewedIndex];
    
    // Create a serializable copy of the profile
    const serializedProfile = { ...currentProfile };
    
    // Convert any Date objects to strings
    if (serializedProfile.moveInDate) {
      if (serializedProfile.moveInDate instanceof Date) {
        serializedProfile.moveInDate = serializedProfile.moveInDate.toISOString();
      }
    }
    
    // Navigate to the DetailedProfile screen inside the SwipeStack
    navigation.navigate('DetailedProfile', { profile: serializedProfile });
  };

  // Function to toggle neighborhood selection
  const toggleNeighborhood = (neighborhood) => {
    if (selectedNeighborhoods.includes(neighborhood)) {
      setSelectedNeighborhoods(selectedNeighborhoods.filter(n => n !== neighborhood));
    } else {
      setSelectedNeighborhoods([...selectedNeighborhoods, neighborhood]);
    }
  };

  // Function to apply filters (this would actually filter the results in a real app)
  const applyFilters = () => {
    // Update filters with selected neighborhoods
    setFilters({
      ...filters,
      locations: selectedNeighborhoods
    });
    
    // In a real app, you would filter profiles here
    // For now, just close the modal
    setShowFilterModal(false);
    
    // Provide feedback that filters were applied
    Alert.alert("Filters Applied", "Your preferences have been saved.");
  };

  // Function to reset all filters
  const resetFilters = () => {
    setFilters({
      budget: { min: 500, max: 3000 },
      locations: [],
      propertyType: [],
      gender: 'any',
      ageRange: { min: 18, max: 50 },
      lifestyle: {
        smoking: 'any',
        pets: 'any',
        drinking: 'any',
        cleanliness: 'any'
      },
      moveInDate: null
    });
    setSelectedNeighborhoods([]);
  };

  const renderCards = () => {
    if (loading) {
      return (
        <View style={styles.emptyStateContainer}>
          <Ionicons name="sync" size={60} color={COLORS.primary} />
          <Text style={styles.emptyStateText}>Loading profiles...</Text>
        </View>
      );
    }

    if (showEmptyState) {
      return (
        <View style={styles.emptyStateContainer}>
          <Ionicons name="search" size={60} color={COLORS.primary} />
          <Text style={styles.emptyStateTitle}>No more profiles</Text>
          <Text style={styles.emptyStateText}>We'll notify you when new roommates match your criteria.</Text>
          <TouchableOpacity 
            style={styles.refreshButton}
            onPress={() => setCurrentIndex(0)}
          >
            <Text style={styles.refreshButtonText}>Start Over</Text>
          </TouchableOpacity>
        </View>
      );
    }

    // Show a stack of cards (visible cards or remaining cards, whichever is less)
    const visibleCards = Math.min(VISIBLE_CARDS, profiles.length - currentIndex);
    const cards = [];

    for (let i = 0; i < visibleCards; i++) {
      const profileIndex = currentIndex + i;
      const profile = profiles[profileIndex];
      const isTopCard = i === 0;

      // Use scale and translateY to create stacked effect
      const scale = 1 - (i * 0.05); // Each card is 5% smaller than the one above
      const translateY = i * CARD_OFFSET; // Each card is offset vertically
      
      cards.push(
        <SwipeCard
          key={profile.id}
          ref={el => cardRefs.current[i] = el}
          profile={profile}
          onSwipeLeft={() => handleSwipeLeft(profileIndex)}
          onSwipeRight={() => handleSwipeRight(profileIndex)}
          onPress={() => navigateToDetailedProfile(profileIndex)}
          isActive={isTopCard}
          style={{
            transform: [{ scale }, { translateY }],
            zIndex: visibleCards - i, // Higher cards have higher z-index
          }}
        />
      );
    }

    return (
      <View style={styles.cardContainer}>
        {/* Render cards in reverse order so first card is on top */}
        {[...cards].reverse().map((card, index) => (
          <React.Fragment key={`card-fragment-${index}`}>
            {card}
          </React.Fragment>
        ))}
      </View>
    );
  };

  // Filter modal content
  const renderFilterModal = () => (
    <Modal
      visible={showFilterModal}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowFilterModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Filter Roommates</Text>
            <TouchableOpacity onPress={() => setShowFilterModal(false)}>
              <Ionicons name="close" size={28} color={COLORS.text} />
            </TouchableOpacity>
          </View>
          
          <ScrollView style={styles.filtersScrollView} showsVerticalScrollIndicator={false}>
            {/* Budget Range Filter */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Budget Range</Text>
              <Text style={styles.filterValue}>
                ${filters.budget.min} - ${filters.budget.max}
              </Text>
              <View style={styles.sliderContainer}>
                <Slider
                  style={styles.slider}
                  minimumValue={500}
                  maximumValue={5000}
                  step={50}
                  minimumTrackTintColor={COLORS.primary}
                  maximumTrackTintColor="#ddd"
                  thumbTintColor={COLORS.primary}
                  value={filters.budget.min}
                  onValueChange={(value) => setFilters({
                    ...filters,
                    budget: { ...filters.budget, min: value }
                  })}
                />
                <Slider
                  style={styles.slider}
                  minimumValue={500}
                  maximumValue={5000}
                  step={50}
                  minimumTrackTintColor={COLORS.primary}
                  maximumTrackTintColor="#ddd"
                  thumbTintColor={COLORS.primary}
                  value={filters.budget.max}
                  onValueChange={(value) => setFilters({
                    ...filters,
                    budget: { ...filters.budget, max: value }
                  })}
                />
              </View>
            </View>
            
            {/* Location Filter */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Chicago Neighborhoods</Text>
              <View style={styles.neighborhoodsContainer}>
                {CHICAGO_NEIGHBORHOODS.map((neighborhood) => (
                  <TouchableOpacity
                    key={neighborhood}
                    style={[
                      styles.neighborhoodChip,
                      selectedNeighborhoods.includes(neighborhood) && styles.selectedNeighborhoodChip
                    ]}
                    onPress={() => toggleNeighborhood(neighborhood)}
                  >
                    <Text 
                      style={[
                        styles.neighborhoodText,
                        selectedNeighborhoods.includes(neighborhood) && styles.selectedNeighborhoodText
                      ]}
                    >
                      {neighborhood}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            
            {/* Property Type Filter */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Property Type</Text>
              <View style={styles.propertyTypeContainer}>
                {['Apartment', 'House', 'Condo', 'Studio'].map((type) => (
                  <TouchableOpacity
                    key={type}
                    style={[
                      styles.propertyTypeChip,
                      filters.propertyType.includes(type) && styles.selectedPropertyTypeChip
                    ]}
                    onPress={() => {
                      if (filters.propertyType.includes(type)) {
                        setFilters({
                          ...filters,
                          propertyType: filters.propertyType.filter(t => t !== type)
                        });
                      } else {
                        setFilters({
                          ...filters,
                          propertyType: [...filters.propertyType, type]
                        });
                      }
                    }}
                  >
                    <Text 
                      style={[
                        styles.propertyTypeText,
                        filters.propertyType.includes(type) && styles.selectedPropertyTypeText
                      ]}
                    >
                      {type}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            
            {/* Gender Preference */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Gender Preference</Text>
              <View style={styles.radioOptionsContainer}>
                {['Any', 'Male', 'Female', 'Non-binary'].map((gender) => (
                  <TouchableOpacity
                    key={gender}
                    style={styles.radioOption}
                    onPress={() => setFilters({
                      ...filters,
                      gender: gender.toLowerCase()
                    })}
                  >
                    <View style={styles.radioCircle}>
                      {filters.gender === gender.toLowerCase() && (
                        <View style={styles.selectedRadioCircle} />
                      )}
                    </View>
                    <Text style={styles.radioText}>{gender}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            
            {/* Age Range Filter */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Age Range</Text>
              <Text style={styles.filterValue}>
                {filters.ageRange.min} - {filters.ageRange.max} years
              </Text>
              <View style={styles.sliderContainer}>
                <Slider
                  style={styles.slider}
                  minimumValue={18}
                  maximumValue={65}
                  step={1}
                  minimumTrackTintColor={COLORS.primary}
                  maximumTrackTintColor="#ddd"
                  thumbTintColor={COLORS.primary}
                  value={filters.ageRange.min}
                  onValueChange={(value) => setFilters({
                    ...filters,
                    ageRange: { ...filters.ageRange, min: value }
                  })}
                />
                <Slider
                  style={styles.slider}
                  minimumValue={18}
                  maximumValue={65}
                  step={1}
                  minimumTrackTintColor={COLORS.primary}
                  maximumTrackTintColor="#ddd"
                  thumbTintColor={COLORS.primary}
                  value={filters.ageRange.max}
                  onValueChange={(value) => setFilters({
                    ...filters,
                    ageRange: { ...filters.ageRange, max: value }
                  })}
                />
              </View>
            </View>
            
            {/* Lifestyle Preferences */}
            <View style={styles.filterSection}>
              <Text style={styles.filterTitle}>Lifestyle Preferences</Text>
              
              {/* Smoking Preference */}
              <View style={styles.lifestyleOption}>
                <View style={styles.lifestyleIconContainer}>
                  <MaterialIcons name="smoking-rooms" size={24} color={COLORS.text} />
                </View>
                <View style={styles.lifestyleTextContainer}>
                  <Text style={styles.lifestyleLabel}>Smoking</Text>
                </View>
                <View style={styles.lifestyleSelectContainer}>
                  <TouchableOpacity
                    style={styles.lifestyleSelectButton}
                    onPress={() => {
                      const nextValue = filters.lifestyle.smoking === 'any' ? 'no' :
                                        filters.lifestyle.smoking === 'no' ? 'yes' : 'any';
                      setFilters({
                        ...filters,
                        lifestyle: { ...filters.lifestyle, smoking: nextValue }
                      });
                    }}
                  >
                    <Text style={styles.lifestyleSelectText}>
                      {filters.lifestyle.smoking === 'any' ? 'Any' :
                       filters.lifestyle.smoking === 'no' ? 'No' : 'Yes'}
                    </Text>
                    <Ionicons name="chevron-down" size={16} color={COLORS.textSecondary} />
                  </TouchableOpacity>
                </View>
              </View>
              
              {/* Pets Preference */}
              <View style={styles.lifestyleOption}>
                <View style={styles.lifestyleIconContainer}>
                  <MaterialIcons name="pets" size={24} color={COLORS.text} />
                </View>
                <View style={styles.lifestyleTextContainer}>
                  <Text style={styles.lifestyleLabel}>Pets</Text>
                </View>
                <View style={styles.lifestyleSelectContainer}>
                  <TouchableOpacity
                    style={styles.lifestyleSelectButton}
                    onPress={() => {
                      const nextValue = filters.lifestyle.pets === 'any' ? 'no' :
                                        filters.lifestyle.pets === 'no' ? 'yes' : 'any';
                      setFilters({
                        ...filters,
                        lifestyle: { ...filters.lifestyle, pets: nextValue }
                      });
                    }}
                  >
                    <Text style={styles.lifestyleSelectText}>
                      {filters.lifestyle.pets === 'any' ? 'Any' :
                       filters.lifestyle.pets === 'no' ? 'No' : 'Yes'}
                    </Text>
                    <Ionicons name="chevron-down" size={16} color={COLORS.textSecondary} />
                  </TouchableOpacity>
                </View>
              </View>
              
              {/* Drinking Preference */}
              <View style={styles.lifestyleOption}>
                <View style={styles.lifestyleIconContainer}>
                  <FontAwesome5 name="glass-martini-alt" size={20} color={COLORS.text} />
                </View>
                <View style={styles.lifestyleTextContainer}>
                  <Text style={styles.lifestyleLabel}>Drinking</Text>
                </View>
                <View style={styles.lifestyleSelectContainer}>
                  <TouchableOpacity
                    style={styles.lifestyleSelectButton}
                    onPress={() => {
                      const nextValue = filters.lifestyle.drinking === 'any' ? 'no' :
                                        filters.lifestyle.drinking === 'no' ? 'yes' : 'any';
                      setFilters({
                        ...filters,
                        lifestyle: { ...filters.lifestyle, drinking: nextValue }
                      });
                    }}
                  >
                    <Text style={styles.lifestyleSelectText}>
                      {filters.lifestyle.drinking === 'any' ? 'Any' :
                       filters.lifestyle.drinking === 'no' ? 'No' : 'Yes'}
                    </Text>
                    <Ionicons name="chevron-down" size={16} color={COLORS.textSecondary} />
                  </TouchableOpacity>
                </View>
              </View>
              
              {/* Cleanliness Preference */}
              <View style={styles.lifestyleOption}>
                <View style={styles.lifestyleIconContainer}>
                  <MaterialIcons name="cleaning-services" size={24} color={COLORS.text} />
                </View>
                <View style={styles.lifestyleTextContainer}>
                  <Text style={styles.lifestyleLabel}>Cleanliness</Text>
                </View>
                <View style={styles.lifestyleSelectContainer}>
                  <TouchableOpacity
                    style={styles.lifestyleSelectButton}
                    onPress={() => {
                      const nextValue = filters.lifestyle.cleanliness === 'any' ? 'clean' :
                                        filters.lifestyle.cleanliness === 'clean' ? 'moderate' :
                                        filters.lifestyle.cleanliness === 'moderate' ? 'relaxed' : 'any';
                      setFilters({
                        ...filters,
                        lifestyle: { ...filters.lifestyle, cleanliness: nextValue }
                      });
                    }}
                  >
                    <Text style={styles.lifestyleSelectText}>
                      {filters.lifestyle.cleanliness === 'any' ? 'Any' :
                       filters.lifestyle.cleanliness === 'clean' ? 'Clean' :
                       filters.lifestyle.cleanliness === 'moderate' ? 'Moderate' : 'Relaxed'}
                    </Text>
                    <Ionicons name="chevron-down" size={16} color={COLORS.textSecondary} />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
            
            {/* Filter Actions */}
            <View style={styles.filterActionsContainer}>
              <TouchableOpacity 
                style={styles.resetButton}
                onPress={resetFilters}
              >
                <Text style={styles.resetButtonText}>Reset All</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={styles.applyButton}
                onPress={applyFilters}
              >
                <Text style={styles.applyButtonText}>Apply Filters</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Discover</Text>
        <TouchableOpacity 
          style={styles.filterButton}
          onPress={() => setShowFilterModal(true)}
        >
          <Ionicons name="options-outline" size={24} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      {renderCards()}
      {renderFilterModal()}

      {/* Add padding at the bottom to account for the tab bar */}
      <View style={styles.bottomSpacer} />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  filterButton: {
    backgroundColor: COLORS.filterButton,
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  emptyStateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyStateTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: 20,
    marginBottom: 10,
  },
  emptyStateText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  refreshButton: {
    backgroundColor: COLORS.primary,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 25,
    marginTop: 20,
  },
  refreshButtonText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 16,
  },
  bottomSpacer: {
    height: 100, // Add extra space at the bottom for the tab bar
  },
  
  // Filter Modal Styles
  modalContainer: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    backgroundColor: 'white',
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    paddingTop: 20,
    height: height * 0.85, // 85% of screen height
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -3 },
        shadowOpacity: 0.1,
        shadowRadius: 6,
      },
      android: {
        elevation: 10,
      },
    }),
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  filtersScrollView: {
    flex: 1,
  },
  filterSection: {
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  filterTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 12,
  },
  filterValue: {
    fontSize: 16,
    color: COLORS.primary,
    marginBottom: 10,
  },
  sliderContainer: {
    paddingVertical: 10,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  
  // Neighborhood Styles
  neighborhoodsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  neighborhoodChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
    marginRight: 8,
    marginBottom: 8,
  },
  selectedNeighborhoodChip: {
    backgroundColor: COLORS.primary + '20', // 20% opacity
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  neighborhoodText: {
    fontSize: 14,
    color: COLORS.text,
  },
  selectedNeighborhoodText: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  
  // Property Type Styles
  propertyTypeContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  propertyTypeChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
    marginRight: 8,
    marginBottom: 8,
  },
  selectedPropertyTypeChip: {
    backgroundColor: COLORS.primary + '20',
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  propertyTypeText: {
    fontSize: 14,
    color: COLORS.text,
  },
  selectedPropertyTypeText: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  
  // Radio Button Styles
  radioOptionsContainer: {
    marginTop: 5,
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  radioCircle: {
    height: 20,
    width: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  selectedRadioCircle: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.primary,
  },
  radioText: {
    fontSize: 16,
    color: COLORS.text,
  },
  
  // Lifestyle Preferences Styles
  lifestyleOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  lifestyleIconContainer: {
    width: 40,
    alignItems: 'center',
  },
  lifestyleTextContainer: {
    flex: 1,
    marginLeft: 10,
  },
  lifestyleLabel: {
    fontSize: 16,
    color: COLORS.text,
  },
  lifestyleSelectContainer: {
    marginLeft: 10,
  },
  lifestyleSelectButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  lifestyleSelectText: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginRight: 5,
  },
  
  // Filter Actions Styles
  filterActionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingVertical: 20,
  },
  resetButton: {
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.text,
    marginRight: 10,
  },
  resetButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.text,
  },
  applyButton: {
    flex: 1,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 10,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
  },
  applyButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
});

export default SwipeScreen; 