# Havyn Data Models Documentation

## Overview

This document provides comprehensive documentation for all data models used in the Havyn Chicago roommate finder application. The application uses Firebase Firestore as its primary database, with four main data collections: Users, Listings, Matches, and Messages.

## Database Architecture

The Havyn application follows a document-based NoSQL database structure using Firebase Firestore. The main collections are:

- **`users`**: Individual user profiles and preferences
- **`listings`**: Property and room listings 
- **`matches`**: Connections between users interested in each other
- **`messages`**: Chat messages between matched users

## Data Models

### 1. User Model

The User model represents individual users of the platform, including their personal information, preferences, and profile settings.

#### TypeScript Interface

```typescript
interface User {
  id: string;                    // Unique identifier (Firebase Auth UID)
  email: string;                 // User's email address
  name: string;                  // Full name
  birthday: Timestamp;           // Date of birth (Firebase Timestamp)
  age?: number;                  // Calculated age (computed field)
  gender: Gender;                // User's gender identity
  profileType: ProfileType;      // Type of user profile
  createdAt: Timestamp;          // Account creation timestamp
  updatedAt: Timestamp;          // Last profile update timestamp
  profileImageUrl?: string;      // URL to profile image in Firebase Storage
  preferences: UserPreferences;  // User's living preferences
  descriptions: string[];        // Array of 3 profile descriptions
  habitsSummary?: string;        // Summary of living habits
  lookingForSummary?: string;    // What user is looking for
}
```

#### Type Definitions

```typescript
// Gender options for inclusivity
export type Gender = 'male' | 'female' | 'non-binary' | 'prefer_not_to_say';

// Profile types define user's intent
export type ProfileType = 'looking_for_room' | 'have_room' | 'apartment_listing';

// Rating scales (1-5) for lifestyle preferences
export type CleanlinessLevel = 1 | 2 | 3 | 4 | 5;
export type NoiseLevel = 1 | 2 | 3 | 4 | 5;
export type SocialLevel = 1 | 2 | 3 | 4 | 5;

// Sleep schedule preferences
export type SleepSchedule = 'early_bird' | 'night_owl' | 'regular';
```

#### User Preferences

```typescript
interface UserPreferences {
  cleanliness: CleanlinessLevel;           // 1 = very messy, 5 = very clean
  noiseLevel: NoiseLevel;                  // 1 = very quiet, 5 = very loud
  socialLevel: SocialLevel;                // 1 = very private, 5 = very social
  sleepSchedule: SleepSchedule;            // Sleep pattern preference
  preferredRoommateGender?: Gender | 'any'; // Gender preference for roommates
  preferredAgeRange?: {                    // Age range for potential roommates
    min: number;
    max: number;
  };
  monthlyRentBudget?: number;              // Budget (required if looking_for_room)
}
```

#### Field Descriptions

- **`id`**: Primary key, matches Firebase Authentication UID
- **`email`**: User's login email, validated during registration
- **`name`**: Display name shown to other users
- **`birthday`**: Used for age calculation and age-based matching
- **`age`**: Computed field, automatically calculated from birthday
- **`gender`**: Self-identified gender, used for preference matching
- **`profileType`**: Determines user's role:
  - `looking_for_room`: Seeking accommodation
  - `have_room`: Has space to offer
  - `apartment_listing`: Listing entire apartment
- **`profileImageUrl`**: Optional profile photo stored in Firebase Storage
- **`descriptions`**: Exactly 3 short descriptive phrases (e.g., "Jazz musician", "Dog lover")
- **`preferences`**: Detailed compatibility factors for matching algorithm

#### Usage Examples

```typescript
// Creating a new user profile
const newUser: Omit<User, 'id' | 'createdAt' | 'updatedAt'> = {
  email: "user@example.com",
  name: "Alex Johnson",
  birthday: Timestamp.fromDate(new Date('1998-05-15')),
  gender: "non-binary",
  profileType: "looking_for_room",
  preferences: {
    cleanliness: 4,
    noiseLevel: 2,
    socialLevel: 3,
    sleepSchedule: "regular",
    preferredRoommateGender: "any",
    preferredAgeRange: { min: 22, max: 30 },
    monthlyRentBudget: 1200
  },
  descriptions: ["Software developer", "Coffee enthusiast", "Yoga practitioner"]
};
```

---

### 2. Listing Model

The Listing model represents available rooms or apartments in Chicago. Each listing is owned by a user and contains property details, location information, and rental terms.

#### TypeScript Interface

```typescript
interface Listing {
  id: string;                    // Unique listing identifier
  ownerId: string;               // Reference to User who created listing
  address: string;               // Street address
  city: string;                  // Always "Chicago"
  area: ChicagoArea;             // Neighborhood/area
  zipCode: string;               // ZIP code for location
  location: GeoPoint;            // Coordinates for map display
  homeDetails: HomeDetails;      // Property specifications
  propertyImageUrls: string[];   // Array of property photos
  description: string;           // Detailed listing description
  createdAt: Timestamp;          // Listing creation date
  updatedAt: Timestamp;          // Last modification date
  active: boolean;               // Whether listing is currently active
}
```

#### Chicago Areas

```typescript
export type ChicagoArea = 
  'Wicker Park' | 'Logan Square' | 'Lincoln Park' | 'Lakeview' | 'River North' | 
  'West Loop' | 'South Loop' | 'Hyde Park' | 'Pilsen' | 'Bucktown' | 
  'Old Town' | 'Uptown' | 'Rogers Park' | 'Edgewater' | 'Andersonville' | 
  'Ravenswood' | 'Bridgeport' | 'Ukrainian Village' | 'Gold Coast' | 'Other';
```

#### Home Details

```typescript
interface HomeDetails {
  rooms: number;                 // Number of bedrooms
  bathrooms: number;             // Number of bathrooms
  rent: number;                  // Monthly rent in USD
  moveInDate?: Timestamp;        // Available move-in date
  leaseLength?: number;          // Lease duration in months
  furnished: boolean;            // Whether space is furnished
  petsAllowed: boolean;          // Pet policy
  amenities: string[];           // List of amenities
}
```

#### Field Descriptions

- **`ownerId`**: Foreign key linking to the User who created the listing
- **`area`**: Predefined Chicago neighborhoods for consistent filtering
- **`location`**: GeoPoint enables radius-based searches and map integration
- **`homeDetails`**: Comprehensive property information for decision-making
- **`propertyImageUrls`**: Multiple photos stored in Firebase Storage
- **`active`**: Allows soft deletion - listings can be deactivated without removal

#### Usage Examples

```typescript
// Creating a new listing
const newListing: Omit<Listing, 'id' | 'createdAt' | 'updatedAt'> = {
  ownerId: "user123",
  address: "1234 N Milwaukee Ave",
  city: "Chicago",
  area: "Wicker Park",
  zipCode: "60622",
  location: new GeoPoint(41.9085, -87.6767),
  homeDetails: {
    rooms: 2,
    bathrooms: 1,
    rent: 1200,
    moveInDate: Timestamp.fromDate(new Date('2024-02-01')),
    leaseLength: 12,
    furnished: false,
    petsAllowed: true,
    amenities: ["Laundry in unit", "Dishwasher", "Balcony", "Near public transit"]
  },
  propertyImageUrls: [],
  description: "Bright 2-bedroom apartment in the heart of Wicker Park...",
  active: true
};
```

---

### 3. Match Model

The Match model represents connections between users who have expressed mutual interest. Matches enable users to communicate and can be related to specific listings.

#### TypeScript Interface

```typescript
interface Match {
  id: string;                    // Unique match identifier
  initiatorId: string;           // User who sent the match request
  recipientId: string;           // User who received the match request
  listingId?: string;            // Optional: specific listing that sparked match
  status: MatchStatus;           // Current state of the match
  createdAt: Timestamp;          // When match was initiated
  updatedAt: Timestamp;          // Last status change
  lastMessageTimestamp?: Timestamp; // Most recent message time
  hasUnreadMessages?: boolean;   // Unread message indicator
}
```

#### Match Status

```typescript
export type MatchStatus = 'pending' | 'accepted' | 'rejected' | 'archived';
```

#### Status Descriptions

- **`pending`**: Initial state when one user shows interest
- **`accepted`**: Both users have confirmed mutual interest
- **`rejected`**: One user declined the match
- **`archived`**: Match was closed or conversation ended

#### Field Descriptions

- **`initiatorId`** and **`recipientId`**: Define the two users in the match
- **`listingId`**: Optional reference to specific property of interest
- **`lastMessageTimestamp`**: Enables sorting matches by recent activity
- **`hasUnreadMessages`**: Powers notification system and UI indicators

#### Usage Examples

```typescript
// Creating a match when user shows interest
const newMatch: Omit<Match, 'id' | 'createdAt' | 'updatedAt'> = {
  initiatorId: "user123",
  recipientId: "user456",
  listingId: "listing789",  // Optional: if match is about specific listing
  status: "pending"
};

// Accepting a match
await updateMatchStatus(matchId, "accepted");
```

---

### 4. Message Model

The Message model stores all communications between matched users, supporting text messages and system notifications.

#### TypeScript Interface

```typescript
interface Message {
  id: string;                    // Unique message identifier
  matchId: string;               // Reference to the Match
  senderId: string;              // User who sent the message
  receiverId: string;            // User who will receive the message
  content: string;               // Message text content
  type: MessageType;             // Type of message
  createdAt: Timestamp;          // When message was sent
  read: boolean;                 // Whether recipient has read message
}
```

#### Message Types

```typescript
export type MessageType = 'text' | 'image' | 'system';
```

#### Type Descriptions

- **`text`**: Regular user-to-user text messages
- **`image`**: Image messages (future feature)
- **`system`**: Automated messages (e.g., "John accepted your match request")

#### Field Descriptions

- **`matchId`**: Groups messages within a conversation
- **`senderId`** and **`receiverId`**: Track message direction
- **`content`**: Message text or system message content
- **`read`**: Powers read receipts and unread message counts

#### Usage Examples

```typescript
// Sending a text message
const newMessage: Omit<Message, 'id'> = {
  matchId: "match123",
  senderId: "user123",
  receiverId: "user456",
  content: "Hi! I'm interested in your listing in Wicker Park.",
  type: "text",
  createdAt: Timestamp.now(),
  read: false
};

// System message for match acceptance
const systemMessage: Omit<Message, 'id'> = {
  matchId: "match123",
  senderId: "system",
  receiverId: "user123",
  content: "Sarah accepted your match request!",
  type: "system",
  createdAt: Timestamp.now(),
  read: false
};
```

## Database Relationships

### User → Listings (One-to-Many)
- One user can create multiple listings
- Each listing belongs to exactly one user
- Relationship: `listing.ownerId` → `user.id`

### User → Matches (Many-to-Many through Match)
- Users can have multiple matches
- Each match involves exactly two users
- Relationships: 
  - `match.initiatorId` → `user.id`
  - `match.recipientId` → `user.id`

### Match → Messages (One-to-Many)
- One match can have multiple messages
- Each message belongs to exactly one match
- Relationship: `message.matchId` → `match.id`

### Listing → Matches (One-to-Many, Optional)
- One listing can generate multiple matches
- Matches can exist without specific listings (general compatibility)
- Relationship: `match.listingId` → `listing.id`

## Security Rules

### Firestore Security Considerations

```javascript
// Example security rules for Users collection
match /users/{userId} {
  // Users can read/write their own profile
  allow read, write: if request.auth.uid == userId;
  
  // Users can read other profiles for matching
  allow read: if request.auth.uid != null;
}

// Example security rules for Matches collection
match /matches/{matchId} {
  // Only participants can access match
  allow read, write: if request.auth.uid in [
    resource.data.initiatorId, 
    resource.data.recipientId
  ];
}
```

## Data Validation

### User Profile Validation
- Email must be valid format and unique
- Name must be 2-50 characters
- Birthday must result in age 18-99
- Descriptions array must contain exactly 3 items
- Rent budget required only for "looking_for_room" profiles

### Listing Validation
- Address must be in Chicago
- Rent must be positive number
- Rooms and bathrooms must be positive integers
- At least one property image recommended

### Message Validation
- Content cannot be empty for text messages
- Sender must be participant in the match
- Match must be in "accepted" status for messaging

## Indexing Strategy

### Recommended Firestore Indexes

```javascript
// Users collection indexes
users: [
  { fields: ['profileType', 'createdAt'] },
  { fields: ['preferences.preferredAgeRange.min', 'preferences.preferredAgeRange.max'] },
  { fields: ['area', 'active'] }
]

// Listings collection indexes
listings: [
  { fields: ['area', 'active', 'createdAt'] },
  { fields: ['homeDetails.rent', 'active'] },
  { fields: ['ownerId', 'active'] }
]

// Matches collection indexes
matches: [
  { fields: ['initiatorId', 'updatedAt'] },
  { fields: ['recipientId', 'updatedAt'] },
  { fields: ['status', 'updatedAt'] }
]

// Messages collection indexes
messages: [
  { fields: ['matchId', 'createdAt'] },
  { fields: ['receiverId', 'read'] }
]
```

## Best Practices

### Data Consistency
- Always update `updatedAt` timestamp when modifying documents
- Use Firestore transactions for operations affecting multiple documents
- Implement optimistic locking for concurrent updates

### Performance Optimization
- Denormalize frequently accessed data (e.g., user names in messages)
- Use pagination for large data sets
- Implement proper caching strategies

### Privacy & Security
- Never store sensitive information in plain text
- Implement proper access controls
- Regular security audits of database rules

### Scalability Considerations
- Design for horizontal scaling
- Consider data partitioning strategies for large user bases
- Plan for archiving old data (messages, inactive listings)

## Migration Strategies

### Schema Evolution
- Use optional fields for new features
- Implement gradual rollouts for schema changes
- Maintain backward compatibility during transitions

### Data Cleanup
- Regular cleanup of inactive listings
- Archive old messages to reduce database size
- Implement user account deletion workflows 