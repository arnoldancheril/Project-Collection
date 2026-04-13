import { 
  doc, 
  setDoc, 
  getDoc, 
  updateDoc, 
  deleteDoc, 
  collection,
  query,
  where,
  getDocs,
  orderBy,
  limit,
  Timestamp,
  GeoPoint,
  addDoc,
  serverTimestamp,
  writeBatch
} from 'firebase/firestore';
import { 
  ref, 
  uploadBytes, 
  getDownloadURL, 
  deleteObject 
} from 'firebase/storage';
// AUTH FUNCTIONS TEMPORARILY DISABLED - Authentication is commented out in firebaseConfig
// import { 
//   createUserWithEmailAndPassword, 
//   signInWithEmailAndPassword, 
//   signOut, 
//   updateProfile,
//   sendPasswordResetEmail
// } from 'firebase/auth';

// import { firebaseAuthentication } from '../../firebaseConfig'; // Temporarily disabled
import { db, storage } from '../../firebaseConfig';
import { User, Listing, Match, Message, ProfileType } from '../models';
import { MatchStatus } from '../models/Match';

// ============ Auth Services ============
// ALL AUTH FUNCTIONS TEMPORARILY DISABLED

/**
 * Register a new user with email and password
 * TEMPORARILY DISABLED - Firebase Auth not available
 */
/*
export const registerUser = async (email: string, password: string) => {
  try {
    const userCredential = await createUserWithEmailAndPassword(firebaseAuthentication, email, password);
    return userCredential.user;
  } catch (error) {
    console.error('Error registering user:', error);
    throw error;
  }
};

export const loginUser = async (email: string, password: string) => {
  try {
    const userCredential = await signInWithEmailAndPassword(firebaseAuthentication, email, password);
    return userCredential.user;
  } catch (error) {
    console.error('Error logging in user:', error);
    throw error;
  }
};

export const logoutUser = async () => {
  try {
    await signOut(firebaseAuthentication);
  } catch (error) {
    console.error('Error logging out user:', error);
    throw error;
  }
};

export const resetPassword = async (email: string) => {
  try {
    await sendPasswordResetEmail(firebaseAuthentication, email);
  } catch (error) {
    console.error('Error sending password reset email:', error);
    throw error;
  }
};

export const updateUserDisplayName = async (displayName: string) => {
  try {
    const user = firebaseAuthentication.currentUser;
    if (user) {
      await updateProfile(user, { displayName });
    }
  } catch (error) {
    console.error('Error updating display name:', error);
    throw error;
  }
};
*/

// ============ User Profile Services ============

/**
 * Create a new user profile in Firestore with sequential ID
 */
export const createUserProfile = async (userData: Partial<User>): Promise<User> => {
  try {
    // Get next sequential user ID
    const nextUserId = await getNextSequentialUserId();
    
    const timestamp = Timestamp.now();
    const userProfile: User = {
      id: userData.id || `user-${Date.now()}`, // Fallback ID for non-auth cases
      userId: nextUserId, // Sequential ID (00001, 00002, etc.)
      email: userData.email || '',
      name: userData.name || '',
      birthday: userData.birthday || timestamp,
      age: userData.age || 18,
      gender: userData.gender || 'non-binary',
      profileType: userData.profileType || 'looking_for_room',
      profileImageUrl: userData.profileImageUrl || '',
      preferences: userData.preferences || {
        cleanliness: 3,
        noiseLevel: 3,
        socialLevel: 3,
        sleepSchedule: 'regular',
        preferredRoommateGender: 'any',
        preferredAgeRange: { min: 18, max: 35 },
        monthlyRentBudget: 1000
      },
      descriptions: userData.descriptions || [],
      habitsSummary: userData.habitsSummary || '',
      lookingForSummary: userData.lookingForSummary || '',
      createdAt: timestamp,
      updatedAt: timestamp,
      ...userData
    };

    const userRef = doc(db, 'users', userProfile.id);
    await setDoc(userRef, userProfile);
    
    return userProfile;
  } catch (error) {
    console.error('Error creating user profile:', error);
    throw error;
  }
};

/**
 * Create a user profile based on the user type
 */
export const createUserProfileByType = async (
  userData: Partial<User>,
  profileType: ProfileType
): Promise<User> => {
  try {
    const baseUserData: Partial<User> = {
      ...userData,
      profileType
    };
    
    // Add type-specific default data
    let enhancedUserData: Partial<User> = { ...baseUserData };
    
    switch (profileType) {
      case 'looking_for_room':
        // Defaults for someone looking for a room
        enhancedUserData = {
          ...enhancedUserData,
          preferences: {
            ...(userData.preferences || {}),
            monthlyRentBudget: userData.preferences?.monthlyRentBudget || 1000,
            preferredRoommateGender: userData.preferences?.preferredRoommateGender || 'any',
            preferredAgeRange: userData.preferences?.preferredAgeRange || { min: 18, max: 35 },
            cleanliness: userData.preferences?.cleanliness || 3,
            noiseLevel: userData.preferences?.noiseLevel || 3,
            socialLevel: userData.preferences?.socialLevel || 3,
            sleepSchedule: userData.preferences?.sleepSchedule || 'regular',
          },
          lookingForSummary: userData.lookingForSummary || 'Looking for a comfortable space with respectful roommates.'
        };
        break;
        
      case 'have_room':
        // Defaults for someone with a room
        enhancedUserData = {
          ...enhancedUserData,
          preferences: {
            ...(userData.preferences || {}),
            preferredRoommateGender: userData.preferences?.preferredRoommateGender || 'any',
            preferredAgeRange: userData.preferences?.preferredAgeRange || { min: 18, max: 35 },
            cleanliness: userData.preferences?.cleanliness || 3,
            noiseLevel: userData.preferences?.noiseLevel || 3,
            socialLevel: userData.preferences?.socialLevel || 3,
            sleepSchedule: userData.preferences?.sleepSchedule || 'regular',
          },
          lookingForSummary: userData.lookingForSummary || 'Looking for a responsible roommate for my place.'
        };
        break;
        
      case 'apartment_listing':
        // Defaults for property management/listings
        enhancedUserData = {
          ...enhancedUserData,
          preferences: {
            ...(userData.preferences || {}),
            cleanliness: userData.preferences?.cleanliness || 5, // Higher standard
            noiseLevel: userData.preferences?.noiseLevel || 2, // Quieter
            socialLevel: userData.preferences?.socialLevel || 3,
            sleepSchedule: userData.preferences?.sleepSchedule || 'regular',
          },
          lookingForSummary: userData.lookingForSummary || 'Professional property management offering quality housing options.'
        };
        break;
    }
    
    return await createUserProfile(enhancedUserData);
  } catch (error) {
    console.error('Error creating user profile by type:', error);
    throw error;
  }
};

/**
 * Get next sequential user ID
 */
export const getNextSequentialUserId = async (): Promise<string> => {
  try {
    const counterRef = doc(db, 'counters', 'userIds');
    const counterSnap = await getDoc(counterRef);
    
    let nextId = 1;
    
    if (counterSnap.exists()) {
      nextId = counterSnap.data().nextId;
      await updateDoc(counterRef, { nextId: nextId + 1 });
    } else {
      await setDoc(counterRef, { nextId: nextId + 1 });
    }
    
    // Format as 5-digit ID with leading zeros
    return nextId.toString().padStart(5, '0');
  } catch (error) {
    console.error('Error getting next sequential user ID:', error);
    throw error;
  }
};

/**
 * Get user profile by ID
 */
export const getUserProfile = async (userId: string): Promise<User | null> => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (userSnap.exists()) {
      return userSnap.data() as User;
    } else {
      return null;
    }
  } catch (error) {
    console.error('Error getting user profile:', error);
    throw error;
  }
};

/**
 * Get user profiles by type
 */
export const getUsersByType = async (profileType: ProfileType): Promise<User[]> => {
  try {
    const usersQuery = query(
      collection(db, 'users'),
      where('profileType', '==', profileType)
    );
    
    const querySnapshot = await getDocs(usersQuery);
    const users: User[] = [];
    
    querySnapshot.forEach((doc) => {
      users.push(doc.data() as User);
    });
    
    return users;
  } catch (error) {
    console.error(`Error getting users with profile type ${profileType}:`, error);
    throw error;
  }
};

/**
 * Update user profile
 */
export const updateUserProfile = async (userId: string, updates: Partial<User>): Promise<void> => {
  try {
    const userRef = doc(db, 'users', userId);
    await updateDoc(userRef, {
      ...updates,
      updatedAt: Timestamp.now()
    });
  } catch (error) {
    console.error('Error updating user profile:', error);
    throw error;
  }
};

/**
 * Upload profile image and update user profile
 * TEMPORARILY DISABLED - Auth required for user ID
 */
/*
export const uploadProfileImage = async (userId: string, imageUri: string): Promise<string> => {
  try {
    const response = await fetch(imageUri);
    const blob = await response.blob();
    
    const storageRef = ref(storage, `profileImages/${userId}`);
    await uploadBytes(storageRef, blob);
    
    const downloadURL = await getDownloadURL(storageRef);
    
    // Update user profile with the image URL
    await updateUserProfile(userId, { profileImageUrl: downloadURL });
    
    return downloadURL;
  } catch (error) {
    console.error('Error uploading profile image:', error);
    throw error;
  }
};
*/

// ============ Listing Services ============

/**
 * Create a new property listing
 */
export const createListing = async (listingData: Omit<Listing, 'id' | 'createdAt' | 'updatedAt'>): Promise<Listing> => {
  try {
    const timestamp = Timestamp.now();
    const listingRef = collection(db, 'listings');
    
    const listingWithTimestamps = {
      ...listingData,
      createdAt: timestamp,
      updatedAt: timestamp,
    };
    
    const docRef = await addDoc(listingRef, listingWithTimestamps);
    
    const listing: Listing = {
      id: docRef.id,
      ...listingWithTimestamps,
    };
    
    await updateDoc(docRef, { id: docRef.id });
    
    return listing;
  } catch (error) {
    console.error('Error creating listing:', error);
    throw error;
  }
};

/**
 * Get a specific listing by ID
 */
export const getListing = async (listingId: string): Promise<Listing | null> => {
  try {
    const listingRef = doc(db, 'listings', listingId);
    const listingSnap = await getDoc(listingRef);
    
    if (listingSnap.exists()) {
      return listingSnap.data() as Listing;
    } else {
      return null;
    }
  } catch (error) {
    console.error('Error getting listing:', error);
    throw error;
  }
};

/**
 * Get listings with filters
 */
export const getListings = async (filters?: {
  userId?: string;
  neighborhood?: string;
  minPrice?: number;
  maxPrice?: number;
  propertyType?: string;
  limit?: number;
}): Promise<Listing[]> => {
  try {
    let listingsQuery = query(collection(db, 'listings'), where('active', '==', true));
    
    if (filters) {
      // Apply filters
      if (filters.userId) {
        listingsQuery = query(listingsQuery, where('ownerId', '==', filters.userId));
      }
      
      if (filters.neighborhood) {
        listingsQuery = query(listingsQuery, where('area', '==', filters.neighborhood));
      }
      
      if (filters.propertyType) {
        listingsQuery = query(listingsQuery, where('propertyType', '==', filters.propertyType));
      }
      
      // Sort by created date (newest first)
      listingsQuery = query(listingsQuery, orderBy('createdAt', 'desc'));
      
      // Apply limit if provided
      if (filters.limit) {
        listingsQuery = query(listingsQuery, limit(filters.limit));
      }
    } else {
      // Default sort and limit
      listingsQuery = query(
        listingsQuery, 
        orderBy('createdAt', 'desc'),
        limit(20)
      );
    }
    
    const querySnapshot = await getDocs(listingsQuery);
    const listings: Listing[] = [];
    
    querySnapshot.forEach((doc) => {
      const listing = doc.data() as Listing;
      
      // Apply client-side price filters if provided
      if (
        (filters?.minPrice === undefined || listing.homeDetails.rent >= filters.minPrice) &&
        (filters?.maxPrice === undefined || listing.homeDetails.rent <= filters.maxPrice)
      ) {
        listings.push(listing);
      }
    });
    
    return listings;
  } catch (error) {
    console.error('Error getting listings:', error);
    throw error;
  }
};

/**
 * Get listings created by users with a room to offer
 */
export const getListingsFromRoomOwners = async (): Promise<{user: User, listing: Listing}[]> => {
  try {
    // First get all users with 'have_room' profile type
    const roomOwners = await getUsersByType('have_room');
    
    // Then get listings for each user
    const results: {user: User, listing: Listing}[] = [];
    
    for (const user of roomOwners) {
      const listings = await getListings({ userId: user.id });
      
      // For each listing, associate it with the user
      for (const listing of listings) {
        results.push({ user, listing });
      }
    }
    
    return results;
  } catch (error) {
    console.error('Error getting listings from room owners:', error);
    throw error;
  }
};

/**
 * Get listings created by apartment companies
 */
export const getApartmentListings = async (): Promise<{user: User, listing: Listing}[]> => {
  try {
    // First get all users with 'apartment_listing' profile type
    const apartmentOwners = await getUsersByType('apartment_listing');
    
    // Then get listings for each user
    const results: {user: User, listing: Listing}[] = [];
    
    for (const user of apartmentOwners) {
      const listings = await getListings({ userId: user.id });
      
      // For each listing, associate it with the user
      for (const listing of listings) {
        results.push({ user, listing });
      }
    }
    
    return results;
  } catch (error) {
    console.error('Error getting apartment listings:', error);
    throw error;
  }
};

/**
 * Update a listing
 */
export const updateListing = async (listingId: string, updates: Partial<Listing>): Promise<void> => {
  try {
    const listingRef = doc(db, 'listings', listingId);
    
    await updateDoc(listingRef, {
      ...updates,
      updatedAt: Timestamp.now()
    });
  } catch (error) {
    console.error('Error updating listing:', error);
    throw error;
  }
};

/**
 * Delete a listing
 */
export const deleteListing = async (listingId: string): Promise<void> => {
  try {
    const listingRef = doc(db, 'listings', listingId);
    await deleteDoc(listingRef);
    
    // Also delete associated images from storage
    // This would need to be expanded to handle multiple images
    try {
      const storageRef = ref(storage, `listings/${listingId}`);
      await deleteObject(storageRef);
    } catch (storageError) {
      console.warn('Error deleting listing images (may not exist):', storageError);
    }
  } catch (error) {
    console.error('Error deleting listing:', error);
    throw error;
  }
};

/**
 * Upload listing images
 */
export const uploadListingImages = async (listingId: string, imageUris: string[]): Promise<string[]> => {
  try {
    const downloadUrls: string[] = [];
    
    for (let i = 0; i < imageUris.length; i++) {
      const imageUri = imageUris[i];
      const response = await fetch(imageUri);
      const blob = await response.blob();
      
      const imageFileName = `image_${i + 1}_${Date.now()}`;
      const storageRef = ref(storage, `listings/${listingId}/${imageFileName}`);
      
      await uploadBytes(storageRef, blob);
      const downloadUrl = await getDownloadURL(storageRef);
      downloadUrls.push(downloadUrl);
    }
    
    // Update the listing with the image URLs
    await updateListing(listingId, {
      propertyImageUrls: downloadUrls
    });
    
    return downloadUrls;
  } catch (error) {
    console.error('Error uploading listing images:', error);
    throw error;
  }
};

// ============ Match Services ============

/**
 * Create a new match between users
 */
export const createMatch = async (initiatorId: string, recipientId: string): Promise<Match> => {
  try {
    const timestamp = Timestamp.now();
    
    // Check if a match already exists
    const existingMatch = await getExistingMatch(initiatorId, recipientId);
    if (existingMatch) {
      return existingMatch;
    }
    
    const matchRef = collection(db, 'matches');
    
    const matchData = {
      initiatorId,
      recipientId,
      status: 'pending' as MatchStatus,
      lastMessageTimestamp: undefined,
      createdAt: timestamp,
      updatedAt: timestamp
    };
    
    const docRef = await addDoc(matchRef, matchData);
    
    const match: Match = {
      id: docRef.id,
      ...matchData,
    };
    
    await updateDoc(docRef, { id: docRef.id });
    
    return match;
  } catch (error) {
    console.error('Error creating match:', error);
    throw error;
  }
};

/**
 * Get existing match between users if it exists
 */
export const getExistingMatch = async (user1Id: string, user2Id: string): Promise<Match | null> => {
  try {
    // Check both directions (user1->user2 and user2->user1)
    const matchQuery1 = query(
      collection(db, 'matches'),
      where('initiatorId', '==', user1Id),
      where('recipientId', '==', user2Id)
    );
    
    const matchQuery2 = query(
      collection(db, 'matches'),
      where('initiatorId', '==', user2Id),
      where('recipientId', '==', user1Id)
    );
    
    const querySnapshot1 = await getDocs(matchQuery1);
    const querySnapshot2 = await getDocs(matchQuery2);
    
    if (!querySnapshot1.empty) {
      return querySnapshot1.docs[0].data() as Match;
    }
    
    if (!querySnapshot2.empty) {
      return querySnapshot2.docs[0].data() as Match;
    }
    
    return null;
  } catch (error) {
    console.error('Error checking for existing match:', error);
    throw error;
  }
};

/**
 * Get all matches for a user
 */
export const getUserMatches = async (userId: string): Promise<Match[]> => {
  try {
    // Get matches where user is either initiator or recipient
    const initiatorQuery = query(
      collection(db, 'matches'),
      where('initiatorId', '==', userId),
      orderBy('updatedAt', 'desc')
    );
    
    const recipientQuery = query(
      collection(db, 'matches'),
      where('recipientId', '==', userId),
      orderBy('updatedAt', 'desc')
    );
    
    const initiatorSnapshot = await getDocs(initiatorQuery);
    const recipientSnapshot = await getDocs(recipientQuery);
    
    const matches: Match[] = [];
    
    initiatorSnapshot.forEach((doc) => {
      matches.push(doc.data() as Match);
    });
    
    recipientSnapshot.forEach((doc) => {
      // Avoid duplicates if both queries return the same match
      const match = doc.data() as Match;
      if (!matches.some(m => m.id === match.id)) {
        matches.push(match);
      }
    });
    
    // Sort by last message timestamp or creation date
    return matches.sort((a, b) => {
      const aTime = a.lastMessageTimestamp?.toMillis() || a.createdAt.toMillis();
      const bTime = b.lastMessageTimestamp?.toMillis() || b.createdAt.toMillis();
      return bTime - aTime; // Descending order (newest first)
    });
  } catch (error) {
    console.error('Error getting user matches:', error);
    throw error;
  }
};

/**
 * Update match status
 */
export const updateMatchStatus = async (matchId: string, status: 'pending' | 'accepted' | 'rejected'): Promise<void> => {
  try {
    const matchRef = doc(db, 'matches', matchId);
    
    await updateDoc(matchRef, {
      status,
      updatedAt: Timestamp.now()
    });
  } catch (error) {
    console.error('Error updating match status:', error);
    throw error;
  }
};

// ============ Message Services ============

/**
 * Send a message
 */
export const sendMessage = async (messageData: Omit<Message, 'id' | 'createdAt'>): Promise<Message> => {
  try {
    const timestamp = Timestamp.now();
    const messageRef = collection(db, 'messages');
    
    const messageWithTimestamp = {
      ...messageData,
      createdAt: timestamp
    };
    
    const docRef = await addDoc(messageRef, messageWithTimestamp);
    
    const message: Message = {
      id: docRef.id,
      ...messageWithTimestamp,
    };
    
    await updateDoc(docRef, { id: docRef.id });
    
    // Update the match with the last message timestamp
    await updateDoc(doc(db, 'matches', messageData.matchId), {
      lastMessageTimestamp: timestamp,
      updatedAt: timestamp
    });
    
    return message;
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

/**
 * Get messages for a match
 */
export const getMatchMessages = async (matchId: string): Promise<Message[]> => {
  try {
    const messagesQuery = query(
      collection(db, 'messages'),
      where('matchId', '==', matchId),
      orderBy('createdAt', 'asc')
    );
    
    const querySnapshot = await getDocs(messagesQuery);
    const messages: Message[] = [];
    
    querySnapshot.forEach((doc) => {
      messages.push(doc.data() as Message);
    });
    
    return messages;
  } catch (error) {
    console.error('Error getting match messages:', error);
    throw error;
  }
}; 