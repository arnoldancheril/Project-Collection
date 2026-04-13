import { Listing, ChicagoArea } from '../models/Listing';
import { Timestamp, GeoPoint } from 'firebase/firestore';

// Sample coordinates for Chicago neighborhoods
const chicagoAreaCoordinates: Record<ChicagoArea, { lat: number, lng: number }> = {
  'Wicker Park': { lat: 41.9088, lng: -87.6796 },
  'Logan Square': { lat: 41.9231, lng: -87.7093 },
  'Lincoln Park': { lat: 41.9214, lng: -87.6513 },
  'Lakeview': { lat: 41.9442, lng: -87.6634 },
  'River North': { lat: 41.8924, lng: -87.6341 },
  'West Loop': { lat: 41.8854, lng: -87.6627 },
  'South Loop': { lat: 41.8735, lng: -87.6285 },
  'Hyde Park': { lat: 41.7943, lng: -87.5917 },
  'Pilsen': { lat: 41.8562, lng: -87.6612 },
  'Bucktown': { lat: 41.9227, lng: -87.6772 },
  'Old Town': { lat: 41.9117, lng: -87.6377 },
  'Uptown': { lat: 41.9665, lng: -87.6533 },
  'Rogers Park': { lat: 42.0095, lng: -87.6768 },
  'Edgewater': { lat: 41.9833, lng: -87.6683 },
  'Andersonville': { lat: 41.9834, lng: -87.6688 },
  'Ravenswood': { lat: 41.9690, lng: -87.6740 },
  'Bridgeport': { lat: 41.8367, lng: -87.6486 },
  'Ukrainian Village': { lat: 41.8960, lng: -87.6760 },
  'Gold Coast': { lat: 41.9000, lng: -87.6270 },
  'Other': { lat: 41.8781, lng: -87.6298 }  // Downtown Chicago as default
};

// Create a function to generate sample property listings
export const generateSampleListings = (count: number = 15): Listing[] => {
  const listings: Listing[] = [];
  const areas = Object.keys(chicagoAreaCoordinates) as ChicagoArea[];
  
  for (let i = 0; i < count; i++) {
    const areaIndex = Math.floor(Math.random() * areas.length);
    const area = areas[areaIndex];
    const baseCoordinates = chicagoAreaCoordinates[area];
    
    // Add a small random offset to the coordinates to spread out properties
    const latOffset = (Math.random() - 0.5) * 0.01;
    const lngOffset = (Math.random() - 0.5) * 0.01;
    
    const location = new GeoPoint(
      baseCoordinates.lat + latOffset,
      baseCoordinates.lng + lngOffset
    );
    
    // Generate a random rent between $1000 and $4500
    const rent = Math.floor(Math.random() * 3500) + 1000;
    
    // Generate random details for the property
    const rooms = Math.floor(Math.random() * 4) + 1;
    const bathrooms = Math.floor(Math.random() * 3) + 1;
    const furnished = Math.random() > 0.5;
    const petsAllowed = Math.random() > 0.3;
    
    const amenities = [];
    if (Math.random() > 0.5) amenities.push('In-unit Laundry');
    if (Math.random() > 0.5) amenities.push('Dishwasher');
    if (Math.random() > 0.6) amenities.push('Central AC');
    if (Math.random() > 0.7) amenities.push('Balcony');
    if (Math.random() > 0.8) amenities.push('Gym');
    if (Math.random() > 0.8) amenities.push('Pool');
    if (Math.random() > 0.7) amenities.push('Parking');
    
    const moveInDate = new Timestamp(
      Math.floor(Date.now() / 1000) + Math.floor(Math.random() * 7776000), // Random date within 90 days
      0
    );
    
    const leaseLength = [6, 12, 18][Math.floor(Math.random() * 3)];
    
    // Generate a sample address
    const streetNumber = Math.floor(Math.random() * 9000) + 1000;
    const streets = ['Main St', 'Park Ave', 'Oak St', 'Maple Rd', 'Washington Blvd', 'Lake St', 'Division St'];
    const street = streets[Math.floor(Math.random() * streets.length)];
    const address = `${streetNumber} ${street}`;
    
    // Generate a zipcode based on the area
    const zipCode = `606${Math.floor(Math.random() * 40) + 10}`;
    
    const descriptions = [
      'Spacious and modern apartment with great natural light.',
      'Charming unit in historic building with hardwood floors.',
      'Newly renovated apartment with stainless steel appliances.',
      'Cozy home with large windows and updated kitchen.',
      'Luxury unit with high ceilings and premium finishes.',
      'Contemporary space with open floor plan and city views.',
      'Classic Chicago apartment with vintage details and modern updates.',
      'Stylish unit in prime location with excellent amenities.',
      'Bright and airy apartment with recent renovations.'
    ];
    
    const description = descriptions[Math.floor(Math.random() * descriptions.length)];
    
    // Generate sample property images - in a real app, these would come from Firebase Storage
    const propertyImageUrls = [
      'https://via.placeholder.com/800x600/4a90e2/ffffff?text=Property+Photo+1',
      'https://via.placeholder.com/800x600/27ae60/ffffff?text=Property+Photo+2',
      'https://via.placeholder.com/800x600/e74c3c/ffffff?text=Property+Photo+3'
    ];
    
    // Create the listing
    listings.push({
      id: `listing-${i + 1}`,
      ownerId: `user-${Math.floor(Math.random() * 30) + 1}`,
      address,
      city: 'Chicago',
      area,
      zipCode,
      location,
      homeDetails: {
        rooms,
        bathrooms,
        rent,
        moveInDate,
        leaseLength,
        furnished,
        petsAllowed,
        amenities
      },
      propertyImageUrls,
      description,
      createdAt: new Timestamp(Math.floor(Date.now() / 1000) - Math.floor(Math.random() * 2592000), 0), // Random date within last 30 days
      updatedAt: new Timestamp(Math.floor(Date.now() / 1000), 0),
      active: true
    });
  }
  
  return listings;
};

// Function to get specific property details
export const getPropertyById = (listings: Listing[], id: string): Listing | undefined => {
  return listings.find(listing => listing.id === id);
};

// Function to filter properties by area
export const filterPropertiesByArea = (listings: Listing[], area?: ChicagoArea): Listing[] => {
  if (!area) return listings;
  return listings.filter(listing => listing.area === area);
};

// Function to filter properties by price range
export const filterPropertiesByPrice = (
  listings: Listing[],
  minPrice?: number,
  maxPrice?: number
): Listing[] => {
  return listings.filter(listing => {
    const rent = listing.homeDetails.rent;
    if (minPrice && maxPrice) {
      return rent >= minPrice && rent <= maxPrice;
    } else if (minPrice) {
      return rent >= minPrice;
    } else if (maxPrice) {
      return rent <= maxPrice;
    }
    return true;
  });
};

// Function to filter properties by number of rooms
export const filterPropertiesByRooms = (
  listings: Listing[],
  minRooms?: number
): Listing[] => {
  if (!minRooms) return listings;
  return listings.filter(listing => listing.homeDetails.rooms >= minRooms);
}; 