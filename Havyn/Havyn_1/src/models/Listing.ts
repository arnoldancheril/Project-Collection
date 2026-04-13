import { Timestamp, GeoPoint } from 'firebase/firestore';

export type ChicagoArea = 
  'Wicker Park' | 'Logan Square' | 'Lincoln Park' | 'Lakeview' | 'River North' | 
  'West Loop' | 'South Loop' | 'Hyde Park' | 'Pilsen' | 'Bucktown' | 
  'Old Town' | 'Uptown' | 'Rogers Park' | 'Edgewater' | 'Andersonville' | 
  'Ravenswood' | 'Bridgeport' | 'Ukrainian Village' | 'Gold Coast' | 'Other';

export interface HomeDetails {
  rooms: number;
  bathrooms: number;
  rent: number;
  moveInDate?: Timestamp;
  leaseLength?: number; // in months
  furnished: boolean;
  petsAllowed: boolean;
  amenities: string[]; // Array of amenities
}

export interface Listing {
  id: string;
  ownerId: string; // Reference to the user who created the listing
  address: string;
  city: string; // Always "Chicago"
  area: ChicagoArea;
  zipCode: string;
  location: GeoPoint; // For map querying
  homeDetails: HomeDetails;
  propertyImageUrls: string[]; // Array of URLs from Firebase Storage
  description: string;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  active: boolean; // Whether the listing is active or not
} 