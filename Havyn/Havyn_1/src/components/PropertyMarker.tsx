import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Marker } from 'react-native-maps';
import { Listing } from '../models/Listing';
import { Ionicons } from '@expo/vector-icons';

interface PropertyMarkerProps {
  listing: Listing;
  onPress: (listing: Listing) => void;
}

const PropertyMarker: React.FC<PropertyMarkerProps> = ({ listing, onPress }) => {
  const { location, homeDetails } = listing;
  const { rent } = homeDetails;
  
  // Format rent as a monthly price
  const formattedRent = `$${rent.toLocaleString()}/mo`;
  
  return (
    <Marker
      coordinate={{
        latitude: location.latitude,
        longitude: location.longitude,
      }}
      onPress={() => onPress(listing)}
    >
      <TouchableOpacity style={styles.markerContainer} onPress={() => onPress(listing)}>
        <View style={styles.priceTag}>
          <Ionicons name="home" size={16} color="#fff" style={styles.icon} />
          <Text style={styles.priceText}>{formattedRent}</Text>
        </View>
      </TouchableOpacity>
    </Marker>
  );
};

const styles = StyleSheet.create({
  markerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  priceTag: {
    backgroundColor: '#3498db',
    borderRadius: 20,
    paddingVertical: 6,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  icon: {
    marginRight: 4,
  },
  priceText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
});

export default PropertyMarker; 