export default class PropertyListing {
  constructor(data = {}) {
    this.id = data.id || '';
    this.ownerId = data.ownerId || '';
    this.title = data.title || '';
    this.description = data.description || '';
    this.propertyType = data.propertyType || 'apartment'; // apartment, house, condo, room
    this.address = data.address || {
      street: '',
      city: '',
      state: '',
      zipCode: '',
      country: 'USA',
    };
    this.location = data.location || {
      latitude: 0,
      longitude: 0,
    };
    this.price = data.price || 0;
    this.bedrooms = data.bedrooms || 1;
    this.bathrooms = data.bathrooms || 1;
    this.totalRooms = data.totalRooms || 1;
    this.availableRooms = data.availableRooms || 1;
    this.squareFeet = data.squareFeet || 0;
    this.amenities = data.amenities || [];
    this.photos = data.photos || [];
    this.utilitiesIncluded = data.utilitiesIncluded || false;
    this.availableFrom = data.availableFrom || new Date();
    this.availableTo = data.availableTo || null; // null means indefinite
    this.leaseLength = data.leaseLength || '12 months';
    this.petPolicy = data.petPolicy || 'No pets allowed';
    this.smokingPolicy = data.smokingPolicy || 'No smoking';
    this.parkingAvailable = data.parkingAvailable || false;
    this.furnished = data.furnished || false;
    this.accessibility = data.accessibility || [];
    this.securityDeposit = data.securityDeposit || 0;
    this.applicationFee = data.applicationFee || 0;
    
    // For roommate matching
    this.currentRoommates = data.currentRoommates || []; // Array of user IDs
    this.roommatePreferences = data.roommatePreferences || {
      gender: 'Any',
      ageRange: { min: 18, max: 99 },
      cleanliness: 0, // 0 means no preference
      lifestyle: '', // e.g., "Student preferred"
    };
    
    // Application and management
    this.interestedUsers = data.interestedUsers || []; // Array of user IDs
    this.rejectedUsers = data.rejectedUsers || []; // Array of user IDs
    this.approvedUsers = data.approvedUsers || []; // Array of user IDs
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
    this.status = data.status || 'active'; // active, pending, rented, expired
    this.virtualTourLink = data.virtualTourLink || '';
    this.videoLink = data.videoLink || '';
  }
  
  toJSON() {
    return {
      id: this.id,
      ownerId: this.ownerId,
      title: this.title,
      description: this.description,
      propertyType: this.propertyType,
      address: this.address,
      location: this.location,
      price: this.price,
      bedrooms: this.bedrooms,
      bathrooms: this.bathrooms,
      totalRooms: this.totalRooms,
      availableRooms: this.availableRooms,
      squareFeet: this.squareFeet,
      amenities: this.amenities,
      photos: this.photos,
      utilitiesIncluded: this.utilitiesIncluded,
      availableFrom: this.availableFrom,
      availableTo: this.availableTo,
      leaseLength: this.leaseLength,
      petPolicy: this.petPolicy,
      smokingPolicy: this.smokingPolicy,
      parkingAvailable: this.parkingAvailable,
      furnished: this.furnished,
      accessibility: this.accessibility,
      securityDeposit: this.securityDeposit,
      applicationFee: this.applicationFee,
      currentRoommates: this.currentRoommates,
      roommatePreferences: this.roommatePreferences,
      interestedUsers: this.interestedUsers,
      rejectedUsers: this.rejectedUsers,
      approvedUsers: this.approvedUsers,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      status: this.status,
      virtualTourLink: this.virtualTourLink,
      videoLink: this.videoLink,
    };
  }
} 