import React, { useState, useEffect, useRef } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Dimensions, 
  TouchableOpacity, 
  ActivityIndicator,
  Alert
} from 'react-native';
import MapView, { PROVIDER_GOOGLE, Region, Marker } from 'react-native-maps';
import { Ionicons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import { 
  PropertyMarker, 
  PropertyDetailModal, 
  CreatePropertyGroupModal,
  PropertyFilterBar
} from '../../src/components';
import { 
  generateSampleListings, 
  filterPropertiesByArea, 
  filterPropertiesByPrice,
  filterPropertiesByRooms
} from '../../src/services/mapService';
import { Listing, ChicagoArea } from '../../src/models/Listing';
import { PropertyFilters } from '../../src/components/PropertyFilterBar';

const { width, height } = Dimensions.get('window');

// Initial map region - Chicago
const INITIAL_REGION: Region = {
  latitude: 41.8781,
  longitude: -87.6298,
  latitudeDelta: 0.1,
  longitudeDelta: 0.1,
};

export default function MapScreen() {
  const [listings, setListings] = useState<Listing[]>([]);
  const [filteredListings, setFilteredListings] = useState<Listing[]>([]);
  const [selectedListing, setSelectedListing] = useState<Listing | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showCreateGroupModal, setShowCreateGroupModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [currentRegion, setCurrentRegion] = useState<Region>(INITIAL_REGION);
  const [userLocation, setUserLocation] = useState<Location.LocationObject | null>(null);
  const [locationPermission, setLocationPermission] = useState<boolean>(false);
  
  const mapRef = useRef<MapView>(null);

  // Get unique areas from listings
  const getUniqueAreas = (): ChicagoArea[] => {
    const areas = listings.map(listing => listing.area);
    return [...new Set(areas)] as ChicagoArea[];
  };

  // Request location permission and get user's location
  useEffect(() => {
    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        const hasPermission = status === 'granted';
        setLocationPermission(hasPermission);
        
        if (hasPermission) {
          const location = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.Balanced,
          });
          setUserLocation(location);
          
          // Center map on user's location, but only if it's in the Chicago area
          const isInChicagoArea = 
            location.coords.latitude > 41.6 && 
            location.coords.latitude < 42.1 && 
            location.coords.longitude > -88.0 && 
            location.coords.longitude < -87.5;
            
          if (isInChicagoArea && mapRef.current) {
            mapRef.current.animateToRegion({
              latitude: location.coords.latitude,
              longitude: location.coords.longitude,
              latitudeDelta: 0.02,
              longitudeDelta: 0.02,
            }, 1000);
          }
        }
      } catch (error) {
        console.error('Error getting location:', error);
      }
    })();
  }, []);

  // Load sample property listings
  useEffect(() => {
    const loadListings = async () => {
      try {
        // In a real app, we would fetch this data from Firestore
        const sampleListings = generateSampleListings(20);
        setListings(sampleListings);
        setFilteredListings(sampleListings);
      } catch (error) {
        console.error('Error loading listings:', error);
        Alert.alert('Error', 'Failed to load property listings');
      } finally {
        setLoading(false);
      }
    };
    
    loadListings();
  }, []);

  // Handle property marker press
  const handleMarkerPress = (listing: Listing) => {
    setSelectedListing(listing);
    setShowDetailModal(true);
    
    // Center the map on the selected property
    if (mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: listing.location.latitude,
        longitude: listing.location.longitude,
        latitudeDelta: 0.02,
        longitudeDelta: 0.02,
      }, 500);
    }
  };

  // Handle property contact button press
  const handleContactOwner = (listing: Listing) => {
    setShowDetailModal(false);
    
    // In a real app, we would navigate to a chat or contact screen
    Alert.alert(
      'Contact Owner',
      `We'll connect you with the owner of this property: ${listing.address}`,
      [{ text: 'OK' }]
    );
  };

  // Handle create group button press
  const handleCreateGroup = (listing: Listing) => {
    setShowDetailModal(false);
    setSelectedListing(listing);
    setShowCreateGroupModal(true);
  };

  // Handle group creation form submission
  const handleGroupCreated = (groupName: string, description: string, preferredRoommates: number) => {
    setShowCreateGroupModal(false);
    
    // In a real app, we would create a group in Firestore
    Alert.alert(
      'Group Created!',
      `Your housing group "${groupName}" has been created for ${selectedListing?.address}. Looking for ${preferredRoommates} roommates.`,
      [{ text: 'Go to Groups', onPress: () => console.log('Navigate to groups tab') }]
    );
  };

  // Handle filter changes
  const handleFilterChange = (filters: PropertyFilters) => {
    let filtered = [...listings];
    
    // Filter by area
    if (filters.area) {
      filtered = filterPropertiesByArea(filtered, filters.area);
    }
    
    // Filter by price range
    filtered = filterPropertiesByPrice(filtered, filters.minPrice, filters.maxPrice);
    
    // Filter by number of rooms
    filtered = filterPropertiesByRooms(filtered, filters.minRooms);
    
    setFilteredListings(filtered);
  };

  // Zoom out to see all properties
  const handleZoomOut = () => {
    if (mapRef.current) {
      mapRef.current.animateToRegion(INITIAL_REGION, 500);
    }
  };

  // Center map on user's location
  const handleCenterOnUser = () => {
    if (!userLocation || !mapRef.current) return;
    
    mapRef.current.animateToRegion({
      latitude: userLocation.coords.latitude,
      longitude: userLocation.coords.longitude,
      latitudeDelta: 0.02,
      longitudeDelta: 0.02,
    }, 500);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading properties...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Property Filter Bar */}
      <PropertyFilterBar 
        onFilterChange={handleFilterChange}
        areas={getUniqueAreas()}
      />
      
      {/* Map */}
      <MapView
        ref={mapRef}
        style={styles.map}
        provider={PROVIDER_GOOGLE}
        initialRegion={INITIAL_REGION}
        onRegionChangeComplete={setCurrentRegion}
        showsUserLocation={locationPermission}
        showsMyLocationButton={false}
        showsCompass={true}
      >
        {/* Property Markers */}
        {filteredListings.map((listing) => (
          <PropertyMarker
            key={listing.id}
            listing={listing}
            onPress={handleMarkerPress}
          />
        ))}
      </MapView>

      {/* Map Controls */}
      <View style={styles.mapControls}>
        {locationPermission && (
          <TouchableOpacity 
            style={[styles.controlButton, { marginBottom: 12 }]} 
            onPress={handleCenterOnUser}
          >
            <Ionicons name="locate" size={24} color="#3498db" />
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.controlButton} onPress={handleZoomOut}>
          <Ionicons name="expand-outline" size={24} color="#333" />
        </TouchableOpacity>
      </View>

      {/* Status Bar - Property Count */}
      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {filteredListings.length} {filteredListings.length === 1 ? 'property' : 'properties'} found
        </Text>
      </View>

      {/* Property Detail Modal */}
      <PropertyDetailModal
        listing={selectedListing}
        visible={showDetailModal}
        onClose={() => setShowDetailModal(false)}
        onContactOwner={handleContactOwner}
        onCreateGroup={handleCreateGroup}
      />

      {/* Create Property Group Modal */}
      <CreatePropertyGroupModal
        listing={selectedListing}
        visible={showCreateGroupModal}
        onClose={() => setShowCreateGroupModal(false)}
        onCreateGroup={handleGroupCreated}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  map: {
    flex: 1,
    width: '100%',
  },
  mapControls: {
    position: 'absolute',
    right: 16,
    bottom: 80,
  },
  controlButton: {
    backgroundColor: 'white',
    borderRadius: 40,
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  statusBar: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
}); 