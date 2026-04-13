import { Timestamp } from 'firebase/firestore';

export interface Coordinates {
  latitude: number;
  longitude: number;
}

export interface Amenity {
  id: string;
  name: string;
  icon?: string; // Icon name for displaying
}

export interface PropertyFeature {
  id: string;
  name: string;
  icon?: string; // Icon name for displaying
}

export interface PropertyHomeDetails {
  bedrooms: number;
  bathrooms: number;
  squareFeet: number;
  rent: number; // Monthly rent in USD
  depositAmount: number;
  availableDate: Timestamp;
  leaseLength: number; // In months
  furnished: boolean;
  petsAllowed: boolean;
  utilities: string[]; // List of included utilities (e.g., "water", "electricity")
  utilitiesCost?: number; // Estimated monthly utilities cost
}

export interface PropertyContact {
  name: string;
  email: string;
  phone?: string;
  preferredContactMethod: 'email' | 'phone' | 'either';
  availableHours?: string;
}

export interface Property {
  id: string;
  ownerId: string; // User ID of the property owner
  title: string;
  description: string;
  propertyType: 'apartment' | 'house' | 'condo' | 'townhouse' | 'room';
  location: {
    address: string;
    city: string;
    state: string;
    zipCode: string;
    coordinates: Coordinates;
    neighborhood: string;
  };
  homeDetails: PropertyHomeDetails;
  amenities: Amenity[]; // Property amenities like gym, pool, etc.
  features: PropertyFeature[]; // Property features like hardwood floors, balcony, etc.
  contact: PropertyContact;
  images: string[]; // Array of image URLs
  active: boolean; // Whether the listing is active
  createdAt: Timestamp;
  updatedAt: Timestamp;
  featured?: boolean; // Whether this is a featured listing
  numberOfRoommates?: number; // For room listings, how many roommates already live there
  maxOccupancy?: number; // Maximum number of occupants allowed
}

export interface PropertyGroup {
  id: string;
  propertyId: string;
  creatorId: string;
  name: string;
  description: string;
  members: string[]; // Array of user IDs
  interestedUsers: string[]; // Array of user IDs interested in joining
  createdAt: Timestamp;
  updatedAt: Timestamp;
  moveInDate?: Timestamp;
  maxMembers: number;
  isOpen: boolean; // Whether the group is still accepting new members
}

export interface PropertyGroupMember {
  userId: string;
  name: string;
  profileImageUrl?: string;
  joinedAt: Timestamp;
  isCreator: boolean;
  status: 'active' | 'pending' | 'left';
}

export interface PropertyFilters {
  minPrice?: number;
  maxPrice?: number;
  bedrooms?: number;
  neighborhood?: string[];
  propertyType?: string[];
  petsAllowed?: boolean;
  furnished?: boolean;
} 