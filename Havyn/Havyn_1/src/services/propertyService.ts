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
  addDoc,
  writeBatch
} from 'firebase/firestore';
import { 
  ref, 
  uploadBytes, 
  getDownloadURL, 
  deleteObject,
  listAll
} from 'firebase/storage';
import { db, storage } from '../../firebaseConfig';
import { 
  Property, 
  PropertyGroup, 
  PropertyGroupMember, 
  PropertyFilters,
  Coordinates
} from '../models/Property';
import { User } from '../models';

/**
 * Create a new property listing
 */
export const createProperty = async (propertyData: Omit<Property, 'id' | 'createdAt' | 'updatedAt'>): Promise<Property> => {
  try {
    const timestamp = Timestamp.now();
    const propertiesRef = collection(db, 'properties');
    
    const propertyWithTimestamps = {
      ...propertyData,
      createdAt: timestamp,
      updatedAt: timestamp,
    };
    
    const docRef = await addDoc(propertiesRef, propertyWithTimestamps);
    
    const property: Property = {
      id: docRef.id,
      ...propertyWithTimestamps,
    };
    
    await updateDoc(docRef, { id: docRef.id });
    
    return property;
  } catch (error) {
    console.error('Error creating property:', error);
    throw error;
  }
};

/**
 * Get a specific property by ID
 */
export const getProperty = async (propertyId: string): Promise<Property | null> => {
  try {
    const propertyRef = doc(db, 'properties', propertyId);
    const propertySnap = await getDoc(propertyRef);
    
    if (propertySnap.exists()) {
      return propertySnap.data() as Property;
    } else {
      return null;
    }
  } catch (error) {
    console.error('Error getting property:', error);
    throw error;
  }
};

/**
 * Get properties with filters
 */
export const getProperties = async (filters?: PropertyFilters): Promise<Property[]> => {
  try {
    let propertiesQuery = query(collection(db, 'properties'), where('active', '==', true));
    
    if (filters) {
      // Apply filters that can be done with Firestore queries
      if (filters.propertyType && filters.propertyType.length > 0) {
        // Note: Firestore doesn't support OR queries directly, so for multiple property types
        // we'll need to filter client-side or use separate queries
        if (filters.propertyType.length === 1) {
          propertiesQuery = query(propertiesQuery, where('propertyType', '==', filters.propertyType[0]));
        }
      }
      
      if (filters.bedrooms) {
        propertiesQuery = query(propertiesQuery, where('homeDetails.bedrooms', '>=', filters.bedrooms));
      }
      
      // Sort by created date (newest first)
      propertiesQuery = query(propertiesQuery, orderBy('createdAt', 'desc'));
    } else {
      // Default sort
      propertiesQuery = query(propertiesQuery, orderBy('createdAt', 'desc'));
    }
    
    const querySnapshot = await getDocs(propertiesQuery);
    let properties: Property[] = [];
    
    querySnapshot.forEach((doc) => {
      properties.push(doc.data() as Property);
    });
    
    // Apply client-side filters that can't be done with Firestore queries
    if (filters) {
      // Filter by price range
      if (filters.minPrice !== undefined) {
        properties = properties.filter(property => property.homeDetails.rent >= filters.minPrice!);
      }
      
      if (filters.maxPrice !== undefined) {
        properties = properties.filter(property => property.homeDetails.rent <= filters.maxPrice!);
      }
      
      // Filter by neighborhood (array of neighborhood names)
      if (filters.neighborhood && filters.neighborhood.length > 0) {
        properties = properties.filter(property => 
          filters.neighborhood!.includes(property.location.neighborhood)
        );
      }
      
      // Filter by multiple property types
      if (filters.propertyType && filters.propertyType.length > 1) {
        properties = properties.filter(property => 
          filters.propertyType!.includes(property.propertyType)
        );
      }
      
      // Filter by pets allowed
      if (filters.petsAllowed !== undefined) {
        properties = properties.filter(property => 
          property.homeDetails.petsAllowed === filters.petsAllowed
        );
      }
      
      // Filter by furnished
      if (filters.furnished !== undefined) {
        properties = properties.filter(property => 
          property.homeDetails.furnished === filters.furnished
        );
      }
    }
    
    return properties;
  } catch (error) {
    console.error('Error getting properties:', error);
    throw error;
  }
};

/**
 * Get properties within a geographic area (for map view)
 */
export const getPropertiesInArea = async (
  center: Coordinates,
  radiusInKm: number
): Promise<Property[]> => {
  try {
    // For simplicity, we'll get all properties and filter by distance client-side
    // In a production app, you might use a geospatial database or Firestore geoqueries
    const properties = await getProperties();
    
    // Filter properties based on distance from center
    return properties.filter(property => {
      const distance = calculateDistance(
        center.latitude,
        center.longitude,
        property.location.coordinates.latitude,
        property.location.coordinates.longitude
      );
      
      return distance <= radiusInKm;
    });
  } catch (error) {
    console.error('Error getting properties in area:', error);
    throw error;
  }
};

/**
 * Calculate distance between two points using Haversine formula
 */
function calculateDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371; // Radius of the Earth in km
  const dLat = deg2rad(lat2 - lat1);
  const dLon = deg2rad(lon2 - lon1);
  const a =
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  const distance = R * c; // Distance in km
  return distance;
}

function deg2rad(deg: number): number {
  return deg * (Math.PI/180);
}

/**
 * Update a property
 */
export const updateProperty = async (propertyId: string, updates: Partial<Property>): Promise<void> => {
  try {
    const propertyRef = doc(db, 'properties', propertyId);
    
    await updateDoc(propertyRef, {
      ...updates,
      updatedAt: Timestamp.now()
    });
  } catch (error) {
    console.error('Error updating property:', error);
    throw error;
  }
};

/**
 * Delete a property
 */
export const deleteProperty = async (propertyId: string): Promise<void> => {
  try {
    const propertyRef = doc(db, 'properties', propertyId);
    await deleteDoc(propertyRef);
    
    // Also delete associated images from storage
    try {
      const storageRef = ref(storage, `properties/${propertyId}`);
      const files = await listAll(storageRef);
      
      // Delete all files in the property folder
      await Promise.all(files.items.map(fileRef => deleteObject(fileRef)));
    } catch (storageError) {
      console.warn('Error deleting property images (may not exist):', storageError);
    }
  } catch (error) {
    console.error('Error deleting property:', error);
    throw error;
  }
};

/**
 * Upload property images
 */
export const uploadPropertyImages = async (propertyId: string, imageUris: string[]): Promise<string[]> => {
  try {
    const downloadUrls: string[] = [];
    
    for (let i = 0; i < imageUris.length; i++) {
      const imageUri = imageUris[i];
      const response = await fetch(imageUri);
      const blob = await response.blob();
      
      const imageFileName = `image_${i + 1}_${Date.now()}`;
      const storageRef = ref(storage, `properties/${propertyId}/${imageFileName}`);
      
      await uploadBytes(storageRef, blob);
      const downloadUrl = await getDownloadURL(storageRef);
      downloadUrls.push(downloadUrl);
    }
    
    // Update the property with the image URLs
    await updateProperty(propertyId, {
      images: downloadUrls
    });
    
    return downloadUrls;
  } catch (error) {
    console.error('Error uploading property images:', error);
    throw error;
  }
};

/**
 * Create a property group for roommates
 */
export const createPropertyGroup = async (groupData: Omit<PropertyGroup, 'id' | 'createdAt' | 'updatedAt'>): Promise<PropertyGroup> => {
  try {
    const timestamp = Timestamp.now();
    const groupsRef = collection(db, 'propertyGroups');
    
    const groupWithTimestamps = {
      ...groupData,
      createdAt: timestamp,
      updatedAt: timestamp,
    };
    
    const docRef = await addDoc(groupsRef, groupWithTimestamps);
    
    const group: PropertyGroup = {
      id: docRef.id,
      ...groupWithTimestamps,
    };
    
    await updateDoc(docRef, { id: docRef.id });
    
    return group;
  } catch (error) {
    console.error('Error creating property group:', error);
    throw error;
  }
};

/**
 * Get property groups for a specific property
 */
export const getPropertyGroups = async (propertyId: string): Promise<PropertyGroup[]> => {
  try {
    const groupsQuery = query(
      collection(db, 'propertyGroups'),
      where('propertyId', '==', propertyId),
      where('isOpen', '==', true),
      orderBy('createdAt', 'desc')
    );
    
    const querySnapshot = await getDocs(groupsQuery);
    const groups: PropertyGroup[] = [];
    
    querySnapshot.forEach((doc) => {
      groups.push(doc.data() as PropertyGroup);
    });
    
    return groups;
  } catch (error) {
    console.error('Error getting property groups:', error);
    throw error;
  }
};

/**
 * Get property groups that a user is a member of
 */
export const getUserPropertyGroups = async (userId: string): Promise<PropertyGroup[]> => {
  try {
    const groupsQuery = query(
      collection(db, 'propertyGroups'),
      where('members', 'array-contains', userId),
      orderBy('updatedAt', 'desc')
    );
    
    const querySnapshot = await getDocs(groupsQuery);
    const groups: PropertyGroup[] = [];
    
    querySnapshot.forEach((doc) => {
      groups.push(doc.data() as PropertyGroup);
    });
    
    return groups;
  } catch (error) {
    console.error('Error getting user property groups:', error);
    throw error;
  }
};

/**
 * Join a property group
 */
export const joinPropertyGroup = async (groupId: string, userId: string): Promise<void> => {
  try {
    const groupRef = doc(db, 'propertyGroups', groupId);
    const groupSnap = await getDoc(groupRef);
    
    if (!groupSnap.exists()) {
      throw new Error('Property group not found');
    }
    
    const group = groupSnap.data() as PropertyGroup;
    
    // Check if user is already a member
    if (group.members.includes(userId)) {
      return; // Already a member, nothing to do
    }
    
    // Check if group is full
    if (group.members.length >= group.maxMembers) {
      throw new Error('Property group is already full');
    }
    
    // Add user to members array
    await updateDoc(groupRef, {
      members: [...group.members, userId],
      updatedAt: Timestamp.now()
    });
    
    // Create group member record
    const memberRef = doc(db, 'propertyGroups', groupId, 'members', userId);
    await setDoc(memberRef, {
      userId,
      joinedAt: Timestamp.now(),
      isCreator: false,
      status: 'active'
    });
  } catch (error) {
    console.error('Error joining property group:', error);
    throw error;
  }
};

/**
 * Express interest in a property group
 */
export const expressInterestInGroup = async (groupId: string, userId: string): Promise<void> => {
  try {
    const groupRef = doc(db, 'propertyGroups', groupId);
    const groupSnap = await getDoc(groupRef);
    
    if (!groupSnap.exists()) {
      throw new Error('Property group not found');
    }
    
    const group = groupSnap.data() as PropertyGroup;
    
    // Check if user is already interested
    if (group.interestedUsers.includes(userId)) {
      return; // Already interested, nothing to do
    }
    
    // Add user to interested users array
    await updateDoc(groupRef, {
      interestedUsers: [...group.interestedUsers, userId],
      updatedAt: Timestamp.now()
    });
  } catch (error) {
    console.error('Error expressing interest in property group:', error);
    throw error;
  }
};

/**
 * Get property group members
 */
export const getPropertyGroupMembers = async (groupId: string): Promise<PropertyGroupMember[]> => {
  try {
    const membersQuery = query(collection(db, 'propertyGroups', groupId, 'members'));
    const querySnapshot = await getDocs(membersQuery);
    
    const memberPromises = querySnapshot.docs.map(async (docSnapshot) => {
      const memberData = docSnapshot.data() as Omit<PropertyGroupMember, 'name' | 'profileImageUrl'>;
      
      // Get user profile for member
      const userRef = doc(db, 'users', memberData.userId);
      const userSnap = await getDoc(userRef);
      
      if (userSnap.exists()) {
        const userData = userSnap.data() as User;
        
        return {
          ...memberData,
          name: userData.name,
          profileImageUrl: userData.profileImageUrl
        };
      } else {
        // Fallback if user not found
        return {
          ...memberData,
          name: 'Unknown User',
        };
      }
    });
    
    const members = await Promise.all(memberPromises);
    return members;
  } catch (error) {
    console.error('Error getting property group members:', error);
    throw error;
  }
}; 