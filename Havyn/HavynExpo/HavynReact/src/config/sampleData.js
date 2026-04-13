import UserProfile from '../models/UserProfile';
import PropertyListing from '../models/PropertyListing';

// Sample user profiles for roommate matching
export const sampleProfiles = [
  new UserProfile({
    id: 'user1',
    email: 'alex@example.com',
    fullName: 'Alex Johnson',
    birthdate: new Date(1995, 5, 15),
    gender: 'Male',
    phoneNumber: '555-123-4567',
    bio: 'Graduate student at State University. I enjoy hiking, reading, and occasional gaming. Looking for a quiet and clean living space.',
    profileImageUrl: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1600486913747-55e5470d6f40?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
      'https://images.unsplash.com/photo-1560250097-0b93528c311a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80'
    ],
    occupation: 'Graduate Student',
    school: 'State University',
    cleanliness: 4,
    noise: 2,
    guestsFrequency: 'Rarely',
    wakeTime: '7:00 AM',
    sleepTime: '11:00 PM',
    smoking: false,
    drinking: 'Occasionally',
    pets: false,
    dietaryRestrictions: ['Vegetarian'],
    preferredGender: 'Any',
    ageRangeMin: 21,
    ageRangeMax: 35,
    budget: { min: 800, max: 1200 },
    desiredLocations: ['Downtown', 'University District'],
    currentLocation: {
      latitude: 37.7749,
      longitude: -122.4194
    },
    moveInDate: new Date(2023, 8, 1),
    accountType: 'roommate'
  }),
  
  new UserProfile({
    id: 'user2',
    email: 'taylor@example.com',
    fullName: 'Taylor Smith',
    birthdate: new Date(1993, 2, 10),
    gender: 'Female',
    phoneNumber: '555-234-5678',
    bio: 'Software engineer working remotely. I love cooking, yoga, and exploring new restaurants. Looking for a roommate with similar interests.',
    profileImageUrl: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1534528741775-53994a69daeb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=764&q=80',
      'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=776&q=80'
    ],
    occupation: 'Software Engineer',
    school: 'Tech University',
    cleanliness: 5,
    noise: 3,
    guestsFrequency: 'Occasionally',
    wakeTime: '8:00 AM',
    sleepTime: '12:00 AM',
    smoking: false,
    drinking: 'Socially',
    pets: true,
    dietaryRestrictions: [],
    preferredGender: 'Female',
    ageRangeMin: 25,
    ageRangeMax: 35,
    budget: { min: 1000, max: 1500 },
    desiredLocations: ['West Side', 'South Bay'],
    currentLocation: {
      latitude: 37.7833,
      longitude: -122.4167
    },
    moveInDate: new Date(2023, 9, 15),
    accountType: 'roommate'
  }),
  
  new UserProfile({
    id: 'user3',
    email: 'jordan@example.com',
    fullName: 'Jordan Wilson',
    birthdate: new Date(1989, 8, 5),
    gender: 'Non-binary',
    phoneNumber: '555-345-6789',
    bio: 'Artist and part-time barista. I\'m creative, neat, and respect personal space. Looking for a laid-back household where I can set up a small studio space.',
    profileImageUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1540569014015-19a7be504e3a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80',
      'https://images.unsplash.com/photo-1507152832244-10d45c7eda57?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80'
    ],
    occupation: 'Artist & Barista',
    school: 'Art Institute',
    cleanliness: 3,
    noise: 2,
    guestsFrequency: 'Occasionally',
    wakeTime: '9:00 AM',
    sleepTime: '1:00 AM',
    smoking: false,
    drinking: 'Rarely',
    pets: true,
    dietaryRestrictions: ['Gluten-free'],
    preferredGender: 'Any',
    ageRangeMin: 25,
    ageRangeMax: 40,
    budget: { min: 700, max: 1100 },
    desiredLocations: ['Arts District', 'Downtown'],
    currentLocation: {
      latitude: 37.7694,
      longitude: -122.4862
    },
    moveInDate: new Date(2023, 7, 10),
    accountType: 'roommate'
  }),
  
  new UserProfile({
    id: 'user4',
    email: 'marcus@example.com',
    fullName: 'Marcus Chen',
    birthdate: new Date(1991, 4, 20),
    gender: 'Male',
    phoneNumber: '555-456-7890',
    bio: 'Medical resident with a chaotic schedule but clean habits. I like to exercise, play piano, and watch documentaries. Looking for understanding roommates.',
    profileImageUrl: 'https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=848&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1629084094878-cf1fda277dc1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
      'https://images.unsplash.com/photo-1590031905406-f18a426d772d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=844&q=80'
    ],
    occupation: 'Medical Resident',
    school: 'Medical University',
    cleanliness: 5,
    noise: 2,
    guestsFrequency: 'Rarely',
    wakeTime: 'Varies',
    sleepTime: 'Varies',
    smoking: false,
    drinking: 'Occasionally',
    pets: false,
    dietaryRestrictions: [],
    preferredGender: 'Any',
    ageRangeMin: 25,
    ageRangeMax: 40,
    budget: { min: 1000, max: 1800 },
    desiredLocations: ['Hospital District', 'Midtown'],
    currentLocation: {
      latitude: 37.7833,
      longitude: -122.4167
    },
    moveInDate: new Date(2023, 6, 1),
    accountType: 'roommate'
  }),
  
  new UserProfile({
    id: 'user5',
    email: 'sophia@example.com',
    fullName: 'Sophia Hernandez',
    birthdate: new Date(1994, 11, 18),
    gender: 'Female',
    phoneNumber: '555-567-8901',
    bio: 'Marketing professional who works from home 3 days a week. I enjoy cooking, running, and watching movies. Looking for a social yet respectful roommate.',
    profileImageUrl: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=776&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1567532939604-b6b5b0db2604?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
      'https://images.unsplash.com/photo-1557555187-23d685287bc3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=776&q=80'
    ],
    occupation: 'Marketing Manager',
    school: 'Business College',
    cleanliness: 4,
    noise: 3,
    guestsFrequency: 'Weekly',
    wakeTime: '7:30 AM',
    sleepTime: '11:30 PM',
    smoking: false,
    drinking: 'Socially',
    pets: false,
    dietaryRestrictions: [],
    preferredGender: 'Female',
    ageRangeMin: 24,
    ageRangeMax: 35,
    budget: { min: 1000, max: 1600 },
    desiredLocations: ['Downtown', 'East Side'],
    currentLocation: {
      latitude: 37.7594,
      longitude: -122.4334
    },
    moveInDate: new Date(2023, 8, 15),
    accountType: 'roommate'
  }),
  
  // Users with rooms to offer
  new UserProfile({
    id: 'user6',
    email: 'david@example.com',
    fullName: 'David Park',
    birthdate: new Date(1988, 3, 12),
    gender: 'Male',
    phoneNumber: '555-678-9012',
    bio: 'Architect who owns a modern 2-bedroom apartment. The spare room is available as I travel frequently for work. Looking for a tidy, respectful roommate.',
    profileImageUrl: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1616002411355-49593fd89721?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
      'https://images.unsplash.com/photo-1577880216142-8549e9488dad?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80'
    ],
    occupation: 'Architect',
    school: 'Design Institute',
    cleanliness: 5,
    noise: 2,
    guestsFrequency: 'Occasionally',
    wakeTime: '6:30 AM',
    sleepTime: '11:00 PM',
    smoking: false,
    drinking: 'Occasionally',
    pets: false,
    dietaryRestrictions: [],
    preferredGender: 'Any',
    ageRangeMin: 25,
    ageRangeMax: 45,
    budget: { min: 0, max: 0 }, // Not looking for a place
    accountType: 'has_room',
    property: 'property1'
  }),
  
  new UserProfile({
    id: 'user7',
    email: 'natalie@example.com',
    fullName: 'Natalie Kim',
    birthdate: new Date(1992, 7, 28),
    gender: 'Female',
    phoneNumber: '555-789-0123',
    bio: 'PhD student with a 3-bedroom house near campus. Two rooms are available, ideal for students or professionals. Shared kitchen and living areas.',
    profileImageUrl: 'https://images.unsplash.com/photo-1601412436009-d964bd02edbc?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
    additionalPhotos: [
      'https://images.unsplash.com/photo-1601412440979-3452424d81a4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=774&q=80',
      'https://images.unsplash.com/photo-1548142813-c348350df52b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=778&q=80'
    ],
    occupation: 'PhD Student',
    school: 'University Research Center',
    cleanliness: 4,
    noise: 3,
    guestsFrequency: 'Occasionally',
    wakeTime: '7:00 AM',
    sleepTime: '12:00 AM',
    smoking: false,
    drinking: 'Socially',
    pets: true,
    dietaryRestrictions: ['Vegetarian'],
    preferredGender: 'Female',
    ageRangeMin: 21,
    ageRangeMax: 35,
    budget: { min: 0, max: 0 }, // Not looking for a place
    accountType: 'has_room',
    property: 'property2'
  }),
  
  // Property managers
  new UserProfile({
    id: 'user8',
    email: 'michael@example.com',
    fullName: 'Michael Roberts',
    birthdate: new Date(1985, 9, 15),
    gender: 'Male',
    phoneNumber: '555-890-1234',
    bio: 'Property manager with multiple listings in the downtown area. Looking for responsible tenants for luxury apartments.',
    profileImageUrl: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80',
    occupation: 'Property Manager',
    accountType: 'property_manager'
  }),
  
  new UserProfile({
    id: 'user9',
    email: 'olivia@example.com',
    fullName: 'Olivia Thompson',
    birthdate: new Date(1990, 5, 22),
    gender: 'Female',
    phoneNumber: '555-901-2345',
    bio: 'Real estate agent specializing in roommate matching for professional shared housing. Multiple properties available.',
    profileImageUrl: 'https://images.unsplash.com/photo-1580489944761-15a19d654956?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1061&q=80',
    occupation: 'Real Estate Agent',
    accountType: 'property_manager'
  })
];

// Sample property listings
export const sampleProperties = [
  // Property owned by David (user6)
  new PropertyListing({
    id: 'property1',
    ownerId: 'user6',
    title: 'Modern 2-Bedroom Apartment with Great View',
    description: 'Beautiful 2-bedroom apartment in a high-rise building with panoramic city views. The available room is spacious with a private bathroom. Shared kitchen and living area. Perfect for professionals.',
    propertyType: 'apartment',
    address: {
      street: '123 Skyline Ave',
      city: 'San Francisco',
      state: 'CA',
      zipCode: '94110',
      country: 'USA',
    },
    location: {
      latitude: 37.7749,
      longitude: -122.4194,
    },
    price: 1200,
    bedrooms: 2,
    bathrooms: 2,
    totalRooms: 4,
    availableRooms: 1,
    squareFeet: 1100,
    amenities: ['Air Conditioning', 'In-unit Washer/Dryer', 'Dishwasher', 'Gym', 'Roof Deck', 'Elevator'],
    photos: [
      'https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1484154218962-a197022b5858?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1774&q=80'
    ],
    utilitiesIncluded: true,
    availableFrom: new Date(2023, 7, 1),
    leaseLength: '12 months',
    petPolicy: 'No pets allowed',
    smokingPolicy: 'No smoking',
    parkingAvailable: true,
    furnished: true,
    accessibility: ['Elevator', 'Wide doorways'],
    securityDeposit: 1200,
    applicationFee: 50,
    roommatePreferences: {
      gender: 'Any',
      ageRange: { min: 25, max: 45 },
      cleanliness: 4,
      lifestyle: 'Professional',
    },
    createdAt: new Date(2023, 6, 15),
    status: 'active',
  }),
  
  // Property owned by Natalie (user7)
  new PropertyListing({
    id: 'property2',
    ownerId: 'user7',
    title: 'Spacious 3-Bedroom House Near Campus',
    description: 'Charming 3-bedroom house within walking distance to the university. Two rooms available for rent. Shared kitchen, living room, and backyard. Perfect for students or young professionals.',
    propertyType: 'house',
    address: {
      street: '456 College Lane',
      city: 'Berkeley',
      state: 'CA',
      zipCode: '94704',
      country: 'USA',
    },
    location: {
      latitude: 37.8715,
      longitude: -122.2730,
    },
    price: 950,
    bedrooms: 3,
    bathrooms: 2,
    totalRooms: 6,
    availableRooms: 2,
    squareFeet: 1800,
    amenities: ['Backyard', 'Washer/Dryer', 'Dishwasher', 'Fireplace', 'Porch', 'Storage'],
    photos: [
      'https://images.unsplash.com/photo-1605276374104-dee2a0ed3cd6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1554995207-c18c203602cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1493809842364-78817add7ffb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80'
    ],
    utilitiesIncluded: false,
    availableFrom: new Date(2023, 8, 1),
    leaseLength: '9 months',
    petPolicy: 'Small pets considered',
    smokingPolicy: 'Outside only',
    parkingAvailable: true,
    furnished: false,
    accessibility: ['First floor bedroom'],
    securityDeposit: 950,
    applicationFee: 30,
    roommatePreferences: {
      gender: 'Female',
      ageRange: { min: 21, max: 35 },
      cleanliness: 3,
      lifestyle: 'Student preferred',
    },
    createdAt: new Date(2023, 7, 1),
    status: 'active',
  }),
  
  // Property owned by Michael (property manager - user8)
  new PropertyListing({
    id: 'property3',
    ownerId: 'user8',
    title: 'Luxury Downtown Condo - Roommate Wanted',
    description: 'High-end condo in the heart of downtown. Shared living arrangement with one existing tenant. Building features include pool, gym, and rooftop lounge. Perfect for a professional.',
    propertyType: 'condo',
    address: {
      street: '789 Financial District',
      city: 'San Francisco',
      state: 'CA',
      zipCode: '94111',
      country: 'USA',
    },
    location: {
      latitude: 37.7913,
      longitude: -122.3991,
    },
    price: 1800,
    bedrooms: 2,
    bathrooms: 2,
    totalRooms: 4,
    availableRooms: 1,
    squareFeet: 1300,
    amenities: ['Pool', 'Gym', 'Rooftop Lounge', 'Concierge', 'In-unit Washer/Dryer', 'Central AC'],
    photos: [
      'https://images.unsplash.com/photo-1567496898669-ee935f5f647a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1771&q=80',
      'https://images.unsplash.com/photo-1459535653751-d571815e906b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1512917774080-9991f1c4c750?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80'
    ],
    utilitiesIncluded: true,
    availableFrom: new Date(2023, 7, 15),
    leaseLength: '12 months',
    petPolicy: 'No pets',
    smokingPolicy: 'No smoking',
    parkingAvailable: true,
    furnished: true,
    accessibility: ['Elevator', 'Wheelchair accessible'],
    securityDeposit: 2000,
    applicationFee: 75,
    currentRoommates: ['existingTenant1'],
    roommatePreferences: {
      gender: 'Any',
      ageRange: { min: 25, max: 45 },
      cleanliness: 5,
      lifestyle: 'Professional',
    },
    createdAt: new Date(2023, 6, 20),
    status: 'active',
  }),
  
  // Property owned by Olivia (property manager - user9)
  new PropertyListing({
    id: 'property4',
    ownerId: 'user9',
    title: 'Shared Professional Housing - Private Rooms',
    description: 'Professionally managed shared house with 4 private bedrooms. Perfect for young professionals and graduate students. Common areas include kitchen, living room, and garden patio.',
    propertyType: 'house',
    address: {
      street: '321 Professional Row',
      city: 'Oakland',
      state: 'CA',
      zipCode: '94611',
      country: 'USA',
    },
    location: {
      latitude: 37.8044,
      longitude: -122.2711,
    },
    price: 1050,
    bedrooms: 4,
    bathrooms: 2,
    totalRooms: 7,
    availableRooms: 2,
    squareFeet: 2200,
    amenities: ['Garden', 'Washer/Dryer', 'Dishwasher', 'High-speed Internet', 'Weekly cleaning', 'Utilities included'],
    photos: [
      'https://images.unsplash.com/photo-1592595896616-c37162298647?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80',
      'https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1692&q=80'
    ],
    utilitiesIncluded: true,
    availableFrom: new Date(2023, 8, 1),
    leaseLength: '6 months',
    petPolicy: 'No pets',
    smokingPolicy: 'No smoking',
    parkingAvailable: true,
    furnished: true,
    accessibility: ['One ground floor bedroom'],
    securityDeposit: 1000,
    applicationFee: 50,
    currentRoommates: ['existingTenant2', 'existingTenant3'],
    roommatePreferences: {
      gender: 'Any',
      ageRange: { min: 23, max: 38 },
      cleanliness: 4,
      lifestyle: 'Professional/Graduate Student',
    },
    createdAt: new Date(2023, 7, 5),
    status: 'active',
  })
]; 