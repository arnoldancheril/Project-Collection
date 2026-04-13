import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  updateProfile,
  EmailAuthProvider,
  reauthenticateWithCredential,
  updatePassword,
  deleteUser,
  sendEmailVerification
} from 'firebase/auth';
import { doc, setDoc, getDoc, updateDoc, deleteDoc } from 'firebase/firestore';
import { auth, db } from '../config/firebaseConfig';
import UserProfile from '../models/UserProfile';

class AuthService {
  // Register a new user
  async register(email, password, userProfile) {
    try {
      // Create the user in Firebase Auth
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const { user } = userCredential;
      
      // Add the user ID to the profile
      userProfile.id = user.uid;
      
      // Set the profile image to the user's auth profile if available
      if (userProfile.profileImageUrl) {
        await updateProfile(user, {
          photoURL: userProfile.profileImageUrl
        });
      }
      
      // Set display name
      await updateProfile(user, {
        displayName: userProfile.fullName
      });
      
      // Send email verification
      await sendEmailVerification(user);
      
      // Store the profile in Firestore
      await setDoc(doc(db, 'users', user.uid), userProfile.toJSON());
      
      return userCredential;
    } catch (error) {
      console.error('Error registering user:', error);
      throw error;
    }
  }
  
  // Sign in an existing user
  async login(email, password) {
    try {
      // For development: bypass authentication
      console.log(`Development mode: Bypassing authentication for ${email}`);
      
      // Create a mock user credential
      const mockUser = {
        uid: 'dev-user-id',
        email: email,
        displayName: email === 'admin' ? 'Admin User' : 'Test User',
        photoURL: null,
        emailVerified: true
      };
      
      const userCredential = {
        user: mockUser
      };
      
      return userCredential;
    } catch (error) {
      console.error('Error signing in:', error);
      throw error;
    }
  }
  
  // Sign out the current user
  async logout() {
    try {
      if (auth.currentUser) {
        const userRef = doc(db, 'users', auth.currentUser.uid);
        await updateDoc(userRef, {
          lastActive: new Date()
        });
      }
      
      await signOut(auth);
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  }
  
  // Get the current user's profile
  async getCurrentUserProfile() {
    try {
      const user = auth.currentUser;
      
      if (!user) {
        return null;
      }
      
      // For development: Return a mock profile
      console.log('Development mode: Returning mock user profile');
      
      // Create a mock profile based on email
      const isAdmin = user.email === 'admin';
      
      return {
        id: user.uid,
        email: user.email,
        fullName: isAdmin ? 'Admin User' : 'Test User',
        profileImageUrl: null,
        userType: isAdmin ? 'propertyManager' : 'roommate',
        bio: 'This is a development account for testing',
        preferences: {
          cleanliness: 5,
          noise: 3,
          guests: 4,
          smoking: false,
          pets: true
        },
        toJSON: function() {
          return { ...this };
        }
      };
    } catch (error) {
      console.error('Error getting user profile:', error);
      throw error;
    }
  }
  
  // Update user profile
  async updateUserProfile(profileData) {
    try {
      const user = auth.currentUser;
      
      if (!user) {
        throw new Error('No user is signed in');
      }
      
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        ...profileData,
        updatedAt: new Date()
      });
      
      // Update display name and photo if provided
      const updateData = {};
      if (profileData.fullName) updateData.displayName = profileData.fullName;
      if (profileData.profileImageUrl) updateData.photoURL = profileData.profileImageUrl;
      
      if (Object.keys(updateData).length > 0) {
        await updateProfile(user, updateData);
      }
      
      return true;
    } catch (error) {
      console.error('Error updating profile:', error);
      throw error;
    }
  }
  
  // Reset password
  async resetPassword(email) {
    try {
      await sendPasswordResetEmail(auth, email);
      return true;
    } catch (error) {
      console.error('Error resetting password:', error);
      throw error;
    }
  }
  
  // Change password
  async changePassword(currentPassword, newPassword) {
    try {
      const user = auth.currentUser;
      
      if (!user) {
        throw new Error('No user is signed in');
      }
      
      // Re-authenticate the user
      const credential = EmailAuthProvider.credential(user.email, currentPassword);
      await reauthenticateWithCredential(user, credential);
      
      // Update the password
      await updatePassword(user, newPassword);
      
      return true;
    } catch (error) {
      console.error('Error changing password:', error);
      throw error;
    }
  }
  
  // Delete account
  async deleteAccount(password) {
    try {
      const user = auth.currentUser;
      
      if (!user) {
        throw new Error('No user is signed in');
      }
      
      // Re-authenticate the user
      const credential = EmailAuthProvider.credential(user.email, password);
      await reauthenticateWithCredential(user, credential);
      
      // Delete the user's profile document
      await deleteDoc(doc(db, 'users', user.uid));
      
      // Delete the user account
      await deleteUser(user);
      
      return true;
    } catch (error) {
      console.error('Error deleting account:', error);
      throw error;
    }
  }
  
  // Check if user is authenticated
  isAuthenticated() {
    // For development: Return true as long as the mock login was called
    return true;
  }
  
  // Get current user id
  getCurrentUserId() {
    // For development: Return the mock user ID
    return 'dev-user-id';
  }
}

export default new AuthService(); 