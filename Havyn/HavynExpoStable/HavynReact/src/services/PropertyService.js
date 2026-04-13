import { 
  collection, 
  query, 
  where, 
  getDocs, 
  getDoc, 
  doc, 
  updateDoc, 
  arrayUnion, 
  arrayRemove, 
  limit, 
  startAfter, 
  orderBy, 
  addDoc, 
  setDoc, 
  deleteDoc 
} from 'firebase/firestore';
import { getDownloadURL, ref, uploadBytes, deleteObject } from 'firebase/storage';
import { db, storage } from '../config/firebaseConfig';
import PropertyListing from '../models/PropertyListing';
import authService from './AuthService';
import userService from './UserService';

class PropertyService {
  // Get property by ID
  async getPropertyById(propertyId) {
    try {
      const propertyRef = doc(db, 'properties', propertyId);
      const propertySnap = await getDoc(propertyRef);
      
      if (propertySnap.exists()) {
        return new PropertyListing(propertySnap.data());
      } else {
        throw new Error('Property not found');
      }
    } catch (error) {
      console.error('Error getting property:', error);
      throw error;
    }
  }
  
  // Create a new property listing
  async createProperty(propertyData) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const newProperty = new PropertyListing({
        ...propertyData,
        ownerId: currentUserId,
        createdAt: new Date(),
        updatedAt: new Date(),
      });
      
      // Create a new document with auto-generated ID
      const propertyRef = doc(collection(db, 'properties'));
      newProperty.id = propertyRef.id;
      
      await setDoc(propertyRef, newProperty.toJSON());
      
      // If the user is of type 'has_room', update their profile with this property
      const currentUser = await authService.getCurrentUserProfile();
      if (currentUser.accountType === 'has_room') {
        await authService.updateUserProfile({
          property: newProperty.id
        });
      }
      
      return newProperty;
    } catch (error) {
      console.error('Error creating property:', error);
      throw error;
    }
  }
  
  // Update an existing property listing
  async updateProperty(propertyId, propertyData) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      // Verify ownership
      const property = await this.getPropertyById(propertyId);
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to update this property');
      }
      
      const propertyRef = doc(db, 'properties', propertyId);
      await updateDoc(propertyRef, {
        ...propertyData,
        updatedAt: new Date()
      });
      
      return true;
    } catch (error) {
      console.error('Error updating property:', error);
      throw error;
    }
  }
  
  // Delete a property listing
  async deleteProperty(propertyId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      // Verify ownership
      const property = await this.getPropertyById(propertyId);
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to delete this property');
      }
      
      // Delete property photos from storage
      if (property.photos && property.photos.length > 0) {
        for (const photoUrl of property.photos) {
          try {
            const photoRef = ref(storage, photoUrl);
            await deleteObject(photoRef);
          } catch (error) {
            console.error(`Error deleting photo ${photoUrl}:`, error);
          }
        }
      }
      
      // Delete property document
      await deleteDoc(doc(db, 'properties', propertyId));
      
      // If user is 'has_room', update their profile
      const currentUser = await authService.getCurrentUserProfile();
      if (currentUser.accountType === 'has_room' && currentUser.property === propertyId) {
        await authService.updateUserProfile({
          property: null
        });
      }
      
      return true;
    } catch (error) {
      console.error('Error deleting property:', error);
      throw error;
    }
  }
  
  // Get all properties for the current property manager
  async getMyProperties() {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const q = query(
        collection(db, 'properties'),
        where('ownerId', '==', currentUserId),
        orderBy('createdAt', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      const properties = [];
      
      querySnapshot.forEach(doc => {
        properties.push(new PropertyListing(doc.data()));
      });
      
      return properties;
    } catch (error) {
      console.error('Error getting properties:', error);
      throw error;
    }
  }
  
  // Search for properties based on criteria
  async searchProperties(filters, lastVisibleProperty = null, batchSize = 10) {
    try {
      let q = collection(db, 'properties');
      const queryConstraints = [];
      
      // Add filters
      if (filters.status) {
        queryConstraints.push(where('status', '==', filters.status));
      } else {
        queryConstraints.push(where('status', '==', 'active'));
      }
      
      if (filters.minPrice) {
        queryConstraints.push(where('price', '>=', filters.minPrice));
      }
      
      if (filters.maxPrice) {
        queryConstraints.push(where('price', '<=', filters.maxPrice));
      }
      
      if (filters.bedrooms) {
        queryConstraints.push(where('bedrooms', '>=', filters.bedrooms));
      }
      
      if (filters.bathrooms) {
        queryConstraints.push(where('bathrooms', '>=', filters.bathrooms));
      }
      
      if (filters.propertyType) {
        queryConstraints.push(where('propertyType', '==', filters.propertyType));
      }
      
      // Add ordering and pagination
      queryConstraints.push(orderBy('price', 'asc'));
      
      if (lastVisibleProperty) {
        queryConstraints.push(startAfter(lastVisibleProperty));
      }
      
      queryConstraints.push(limit(batchSize));
      
      q = query(q, ...queryConstraints);
      
      const querySnapshot = await getDocs(q);
      const properties = [];
      let lastVisible = null;
      
      querySnapshot.forEach(doc => {
        properties.push(new PropertyListing(doc.data()));
        lastVisible = doc;
      });
      
      return { properties, lastVisible };
    } catch (error) {
      console.error('Error searching properties:', error);
      throw error;
    }
  }
  
  // Upload property image
  async uploadPropertyImage(uri, fileName) {
    try {
      const response = await fetch(uri);
      const blob = await response.blob();
      
      const storageRef = ref(storage, `propertyImages/${fileName}`);
      await uploadBytes(storageRef, blob);
      
      const downloadURL = await getDownloadURL(storageRef);
      return downloadURL;
    } catch (error) {
      console.error('Error uploading property image:', error);
      throw error;
    }
  }
  
  // Express interest in a property
  async expressInterest(propertyId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      // Update the property with the interested user
      const propertyRef = doc(db, 'properties', propertyId);
      await updateDoc(propertyRef, {
        interestedUsers: arrayUnion(currentUserId)
      });
      
      return true;
    } catch (error) {
      console.error('Error expressing interest:', error);
      throw error;
    }
  }
  
  // Get interested users for a property
  async getInterestedUsers(propertyId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const property = await this.getPropertyById(propertyId);
      
      // Verify ownership
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to view interested users');
      }
      
      const interestedUsers = [];
      
      if (property.interestedUsers && property.interestedUsers.length > 0) {
        for (const userId of property.interestedUsers) {
          try {
            const user = await userService.getUserById(userId);
            interestedUsers.push(user);
          } catch (error) {
            console.error(`Error getting interested user ${userId}:`, error);
          }
        }
      }
      
      return interestedUsers;
    } catch (error) {
      console.error('Error getting interested users:', error);
      throw error;
    }
  }
  
  // Approve a user for a property
  async approveUser(propertyId, userId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const property = await this.getPropertyById(propertyId);
      
      // Verify ownership
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to approve users');
      }
      
      const propertyRef = doc(db, 'properties', propertyId);
      await updateDoc(propertyRef, {
        approvedUsers: arrayUnion(userId),
        interestedUsers: arrayRemove(userId)
      });
      
      return true;
    } catch (error) {
      console.error('Error approving user:', error);
      throw error;
    }
  }
  
  // Reject a user for a property
  async rejectUser(propertyId, userId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const property = await this.getPropertyById(propertyId);
      
      // Verify ownership
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to reject users');
      }
      
      const propertyRef = doc(db, 'properties', propertyId);
      await updateDoc(propertyRef, {
        rejectedUsers: arrayUnion(userId),
        interestedUsers: arrayRemove(userId)
      });
      
      return true;
    } catch (error) {
      console.error('Error rejecting user:', error);
      throw error;
    }
  }
  
  // Get property analytics
  async getPropertyAnalytics(propertyId) {
    try {
      const currentUserId = authService.getCurrentUserId();
      
      if (!currentUserId) {
        throw new Error('No authenticated user');
      }
      
      const property = await this.getPropertyById(propertyId);
      
      // Verify ownership
      if (property.ownerId !== currentUserId) {
        throw new Error('You do not have permission to view analytics');
      }
      
      // Calculate analytics
      return {
        views: property.views || 0,
        interestedCount: property.interestedUsers ? property.interestedUsers.length : 0,
        approvedCount: property.approvedUsers ? property.approvedUsers.length : 0,
        rejectedCount: property.rejectedUsers ? property.rejectedUsers.length : 0,
        daysListed: Math.floor((new Date() - new Date(property.createdAt)) / (1000 * 60 * 60 * 24)),
        status: property.status,
      };
    } catch (error) {
      console.error('Error getting property analytics:', error);
      throw error;
    }
  }
}

export default new PropertyService(); 