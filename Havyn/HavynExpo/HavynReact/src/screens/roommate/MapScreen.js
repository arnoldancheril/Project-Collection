import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ScrollView, Alert, Platform, Image, SafeAreaView, Modal } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';
import * as Location from 'expo-location';
import SAMPLE_PROFILES from '../../utils/sampleProfiles';

// Try to import MapView components
let MapView, Marker, PROVIDER_GOOGLE;
try {
  const Maps = require('react-native-maps');
  MapView = Maps.default;
  Marker = Maps.Marker;
  PROVIDER_GOOGLE = Maps.PROVIDER_GOOGLE;
} catch (error) {
  console.warn('Could not load react-native-maps:', error);
}

// Filter options for the map view
const FILTER_OPTIONS = [
  { id: 'all', label: 'All Areas', icon: 'globe-outline' },
  { id: 'downtown', label: 'Downtown', icon: 'business-outline' },
  { id: 'north', label: 'North Side', icon: 'compass-outline' },
  { id: 'west', label: 'West Side', icon: 'navigate-outline' },
];

// Convert sample profiles to property listings
const convertProfilesToListings = () => {
  return SAMPLE_PROFILES.filter(profile => 
    profile.property && profile.location
  ).map(profile => ({
    id: profile.id,
    title: `${profile.property.type || 'Property'} in ${profile.location.neighborhood || 'Chicago'}`,
    description: profile.bio,
    rent: profile.property.rent || '$1000-$2000/month',
    images: profile.images,
    coordinate: {
      latitude: 41.8781 + (Math.random() * 0.1 - 0.05),  // Random Chicago area coordinates
      longitude: -87.6298 + (Math.random() * 0.1 - 0.05)
    },
    type: profile.property.type || 'Apartment',
    neighborhood: profile.location.neighborhood || 'Unknown',
    amenities: Array.isArray(profile.property.amenities) ? profile.property.amenities : [profile.property.amenities],
    bedrooms: profile.property.rooms || 1,
    bathrooms: profile.property.bathrooms || 1,
    profileName: profile.firstName,
    profileAge: profile.age,
    verified: profile.verified
  }));
};

// Filter pill component
const FilterPill = ({ label, icon, isActive, onPress }) => (
  <TouchableOpacity 
    style={[styles.filterPill, isActive && styles.activePill]} 
    onPress={onPress}
  >
    <Ionicons 
      name={icon} 
      size={16} 
      color={isActive ? 'white' : COLORS.primary} 
    />
    <Text style={[styles.filterPillText, isActive && styles.activePillText]}>{label}</Text>
  </TouchableOpacity>
);

const MapScreen = ({ navigation }) => {
  const [region, setRegion] = useState({
    latitude: 41.8781, // Chicago latitude
    longitude: -87.6298, // Chicago longitude
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  });
  const [selectedProperty, setSelectedProperty] = useState(null);
  const [activeFilter, setActiveFilter] = useState('all');
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [userLocation, setUserLocation] = useState(null);
  const [mapError, setMapError] = useState(true); // Assume MapView is not available initially
  const [viewMode, setViewMode] = useState('map'); // 'map' or 'list'
  const [mapStyle, setMapStyle] = useState('standard'); // 'standard', '3d', 'satellite'

  useEffect(() => {
    // Convert profiles to property listings
    const listings = convertProfilesToListings();
    setProperties(listings);
    
    // Request location permissions and get current position
    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission denied', 'Location access is required to show your position on the map.');
          setLoading(false);
          return;
        }

        const location = await Location.getCurrentPositionAsync({});
        const { latitude, longitude } = location.coords;
        
        setUserLocation({ latitude, longitude });
        setRegion({
          latitude: 41.8781, // Always center on Chicago
          longitude: -87.6298,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        });
      } catch (error) {
        console.error('Error getting location:', error);
        Alert.alert('Location error', 'Could not determine your location. Using default map view.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Filter properties based on selected filter
  const filteredProperties = () => {
    if (activeFilter === 'all') return properties;
    
    // Filtering logic based on neighborhoods
    const filterByNeighborhood = (property) => {
      const neighborhoodMap = {
        'downtown': ['Loop', 'River North', 'Streeterville', 'West Loop'],
        'north': ['Lincoln Park', 'Lakeview', 'Wrigleyville', 'Uptown'],
        'west': ['Wicker Park', 'Bucktown', 'Logan Square', 'Ukrainian Village']
      };
      
      return neighborhoodMap[activeFilter]?.includes(property.neighborhood);
    };
    
    return properties.filter(filterByNeighborhood);
  };

  const handleMarkerPress = (property) => {
    setSelectedProperty(property);
  };

  const handleViewPropertyDetails = () => {
    if (selectedProperty) {
      navigation.navigate('PropertyDetails', { property: selectedProperty });
      setSelectedProperty(null);
    }
  };

  // Render filter bar
  const renderFilterBar = () => (
    <ScrollView 
      horizontal 
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.filterBarContent}
      style={styles.filterBar}
    >
      {FILTER_OPTIONS.map((filter) => (
        <FilterPill 
          key={filter.id}
          label={filter.label} 
          icon={filter.icon} 
          isActive={activeFilter === filter.id}
          onPress={() => setActiveFilter(filter.id)} 
        />
      ))}
    </ScrollView>
  );

  // Render a fallback UI when MapView cannot be loaded
  const renderListView = () => (
    <View style={styles.fallbackContainer}>
      <View style={styles.listHeaderContainer}>
        <Ionicons name="map" size={60} color={COLORS.primary} style={styles.fallbackIcon} />
        <Text style={styles.fallbackTitle}>Property Listings</Text>
        <Text style={styles.fallbackText}>
          Browse available properties in Chicago:
        </Text>
      </View>
      <ScrollView style={styles.propertiesList}>
        {filteredProperties().map((property) => (
          <TouchableOpacity 
            key={property.id}
            style={styles.propertyListItem}
            onPress={() => navigation.navigate('PropertyDetails', { property })}
          >
            <Image 
              source={property.images[0]}
              style={styles.propertyListImage}
            />
            <View style={styles.propertyListInfo}>
              <Text style={styles.propertyListTitle}>{property.title}</Text>
              <Text style={styles.propertyListRent}>{property.rent}</Text>
              <Text style={styles.propertyListDetails}>
                {property.bedrooms} bed • {property.bathrooms} bath • {property.neighborhood}
              </Text>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {renderFilterBar()}
      
      {loading ? (
        <View style={styles.loadingContainer}>
          <Ionicons name="sync" size={80} color={COLORS.primary} style={styles.loadingIcon} />
          <Text style={styles.loadingText}>Loading properties...</Text>
        </View>
      ) : (
        renderListView()
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  mapContainer: {
    flex: 1,
    position: 'relative',
  },
  map: {
    width: '100%',
    height: '100%',
  },
  filterBar: {
    maxHeight: 60,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  filterBarContent: {
    flexDirection: 'row',
    paddingVertical: 10,
    paddingHorizontal: 15,
  },
  filterPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F0F6FF',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
    marginRight: 10,
    borderWidth: 1,
    borderColor: COLORS.primary,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  activePill: {
    backgroundColor: COLORS.primary,
  },
  filterPillText: {
    color: COLORS.primary,
    marginLeft: 6,
    fontWeight: '600',
  },
  activePillText: {
    color: 'white',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  loadingIcon: {
    marginBottom: 20,
    opacity: 0.7,
  },
  loadingText: {
    fontSize: 18,
    color: COLORS.textSecondary,
    marginTop: 10,
  },
  fallbackContainer: {
    flex: 1,
    padding: 10,
    backgroundColor: '#fff',
  },
  listHeaderContainer: {
    alignItems: 'center',
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
    marginBottom: 10,
  },
  fallbackIcon: {
    marginTop: 10,
    marginBottom: 10,
    opacity: 0.7,
  },
  fallbackTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 5,
  },
  fallbackText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginBottom: 10,
  },
  propertiesList: {
    width: '100%',
  },
  propertyListItem: {
    flexDirection: 'row',
    marginBottom: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    overflow: 'hidden',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  propertyListImage: {
    width: 120,
    height: 120,
  },
  propertyListInfo: {
    flex: 1,
    padding: 10,
    justifyContent: 'center',
  },
  propertyListTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  propertyListRent: {
    fontSize: 14,
    color: COLORS.primary,
    fontWeight: '600',
    marginBottom: 5,
  },
  propertyListDetails: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
});

export default MapScreen; 